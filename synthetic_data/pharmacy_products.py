"""Справочник товаров аптечной сети: лекарства из открытого реестра ЖНВЛП + синтетические нелекарственные категории.

Лекарства берутся из Государственного реестра предельных отпускных цен (grls.rosminzdrav.ru, архив на
28.09.2026, лист «Действующие»): МНН, торговое наименование, лекформа, дозировка, фасовка, производитель,
АТХ, предельная цена производителя. Из реестра отбираются только позиции, которые правдоподобно продаются
в розничной аптеке (без инфузий, лиофилизатов, больничных упаковок, контрастных средств и т.п.).

Розничная цена, категории для кластеризации, признаки СТМ и минимального ассортимента - синтетика.
Нелекарственные товары (БАД, гигиена, мать и дитя, медизделия, оптика) и их бренды выдуманы.
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_ZIP = Path(__file__).parent / "raw" / "zhnvlp_2026-09-28.zip"

# --- АТХ: первый уровень -> категория для кластеризации, второй уровень -> подкатегория для ценового анализа ---
ATC1_CATEGORY = {
    "A": "Пищеварение и обмен веществ", "B": "Кровь и кроветворение", "C": "Сердечно-сосудистая система",
    "D": "Дерматология", "G": "Мочеполовая система и половые гормоны", "H": "Гормоны системного действия",
    "J": "Противомикробные", "L": "Иммуномодуляторы и онкология", "M": "Костно-мышечная система",
    "N": "Нервная система", "P": "Прочие препараты", "R": "Дыхательная система", "S": "Органы чувств",
    "V": "Прочие препараты",
}
ATC2_NAME = {
    "A01": "Стоматологические препараты", "A02": "Препараты при нарушениях кислотности",
    "A03": "Препараты при функциональных нарушениях ЖКТ", "A04": "Противорвотные",
    "A05": "Препараты для печени и желчевыводящих путей", "A06": "Слабительные",
    "A07": "Противодиарейные и кишечные противовоспалительные", "A09": "Ферментные препараты",
    "A10": "Препараты для лечения диабета", "A11": "Витамины", "A12": "Минеральные добавки",
    "A16": "Прочие препараты для ЖКТ и обмена веществ", "B01": "Антикоагулянты", "B02": "Гемостатики",
    "B03": "Антианемические", "C01": "Препараты для лечения заболеваний сердца", "C02": "Антигипертензивные",
    "C03": "Диуретики", "C04": "Периферические вазодилататоры", "C05": "Ангиопротекторы",
    "C07": "Бета-адреноблокаторы", "C08": "Блокаторы кальциевых каналов",
    "C09": "Средства, влияющие на ренин-ангиотензиновую систему", "C10": "Гиполипидемические",
    "D01": "Противогрибковые для кожи", "D02": "Смягчающие и защитные", "D03": "Препараты для лечения ран",
    "D04": "Противозудные", "D06": "Антибиотики для наружного применения",
    "D07": "Глюкокортикоиды для наружного применения", "D08": "Антисептики", "D10": "Препараты для лечения акне",
    "D11": "Прочие дерматологические", "G01": "Противомикробные для гинекологии", "G02": "Прочие гинекологические",
    "G03": "Половые гормоны", "G04": "Урологические препараты", "H01": "Гормоны гипофиза и гипоталамуса",
    "H02": "Кортикостероиды системного действия", "H03": "Препараты для щитовидной железы",
    "H05": "Регуляторы обмена кальция", "J01": "Антибактериальные системного действия",
    "J02": "Противогрибковые системного действия", "J05": "Противовирусные системного действия",
    "L02": "Гормональные противоопухолевые", "L03": "Иммуностимуляторы", "L04": "Иммунодепрессанты",
    "M01": "Противовоспалительные и противоревматические", "M02": "Местные средства при болях в мышцах и суставах",
    "M03": "Миорелаксанты", "M04": "Противоподагрические", "M05": "Препараты для лечения заболеваний костей",
    "M09": "Прочие препараты для костно-мышечной системы", "N02": "Анальгетики", "N03": "Противоэпилептические",
    "N04": "Противопаркинсонические", "N05": "Психолептики", "N06": "Психоаналептики",
    "N07": "Прочие препараты для нервной системы", "P01": "Противопротозойные", "P02": "Противогельминтные",
    "P03": "Средства против эктопаразитов", "R01": "Назальные препараты", "R02": "Препараты для лечения горла",
    "R03": "Препараты при обструктивных заболеваниях дыхательных путей", "R05": "Препараты от кашля и простуды",
    "R06": "Антигистаминные системного действия", "R07": "Прочие препараты для дыхательной системы",
    "S01": "Офтальмологические препараты", "S02": "Препараты для лечения уха",
    "S03": "Офтальмологические и отологические препараты", "V06": "Лечебное питание",
}

# лекформа (начало текста до первой запятой) -> укрупнённая группа; порядок важен, берётся первое совпадение
FORM_GROUPS = [
    (r"саше|пакетик", "саше"),
    (r"^таблетк|^драже", "таблетки"),
    (r"^капсул", "капсулы"),
    (r"^суппозитор", "суппозитории"),
    (r"^гранул|порошок для приготовления (?:раствора|суспензии) для приема внутрь|^порошок для приема внутрь", "порошок"),
    (r"^сироп|суспензия для приема внутрь|раствор для приема внутрь|эмульсия для приема внутрь|эликсир", "жидкость внутрь"),
    (r"^капли", "капли"),
    (r"^спрей|^аэрозоль|порошок для ингаляций|раствор для ингаляций|суспензия для ингаляций", "спрей и ингаляции"),
    (r"^мазь|^крем|^гель|^линимент|^паста|^пластырь|трансдермальн", "мазь, крем, гель"),
    (r"раствор для (?:местного|наружного)", "раствор наружный"),
    (r"раствор для (?:внутримышечного|подкожного) введения|раствор для инъекций", "раствор для инъекций"),
]
RETAIL_ATC1_WEIGHT = {"R": 3.0, "D": 4.0, "S": 3.0, "M": 2.0, "N": 1.5, "A": 1.5, "G": 2.0, "H": 1.0,
                      "C": 0.8, "J": 0.6, "B": 0.6, "L": 0.25, "P": 2.0, "V": 1.0}
COUNTABLE_FORMS = {"таблетки", "капсулы", "суппозитории", "саше"}
EXCLUDED_ATC2 = {"B05", "V01", "V03", "V04", "V07", "V08", "V09", "V10", "N01", "J04", "J06", "J07", "L01", "H04", "D09"}
DOSE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(мг|мкг|г|МЕ|ЕД|мл|%)", re.IGNORECASE)


def load_registry(path=RAW_ZIP) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    raw = pd.read_excel(io.BytesIO(z.read(z.infolist()[0])), "Действующие", header=2)
    raw.columns = ["mnn", "trade_name", "form_full", "owner", "atc", "pack_qty", "price_limit",
                   "price_primary", "reg_no", "price_reg_date", "ean13", "effective_date"]
    return raw


def parse_registry(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.dropna(subset=["trade_name", "form_full", "atc", "pack_qty", "price_limit"]).copy()
    d["atc"] = d["atc"].astype(str).str.strip()
    d["atc2"] = d["atc"].str[:3]
    d = d[d["atc2"].isin(ATC2_NAME) & ~d["atc2"].isin(EXCLUDED_ATC2)]

    form = d["form_full"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    d["form_name"] = form.str.split(",").str[0].str.strip().str.lower()
    d["form_group"] = None
    for pattern, group in FORM_GROUPS:
        free = d["form_group"].isna()
        d.loc[free & d["form_name"].str.contains(pattern, regex=True), "form_group"] = group
    d = d[d["form_group"].notna()]

    # больничные упаковки, огромные фасовки и очень дорогие позиции - не розница
    d = d[~form.loc[d.index].str.contains("стационар", case=False)]
    d = d[(d["pack_qty"] >= 1) & (d["pack_qty"] <= 120)]
    d = d[d["price_limit"] <= 4000]

    dose = d["form_full"].astype(str).str.extract(DOSE_RE)
    d["dose_value"] = pd.to_numeric(dose[0].str.replace(",", "."), errors="coerce")
    d["dose_unit"] = dose[1]
    d["dosage"] = (d["dose_value"].map(lambda v: f"{v:g}" if pd.notna(v) else "") + " " + d["dose_unit"].fillna("")).str.strip()
    d["pack_ambiguous"] = d["form_full"].astype(str).str.contains(r"в комплекте|комплект|набор", case=False)

    # владелец РУ: первый сегмент до «;», без префиксов ролей (Вл., Вып.к., Перв.Уп. ...) и номера в скобках в конце
    owner = d["owner"].astype(str).str.split(";").str[0]
    owner = owner.str.replace(r"^(?:Вл\.|Вып\.к\.|Перв\.Уп\.|Втор\.Уп\.|Пр\.)+", "", regex=True)
    owner = owner.str.replace(r"\s*\([^()]*\)\s*$", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    d["country"] = owner.str.extract(r",\s*([^,]+)$")[0].str.strip()
    full_name = owner.str.replace(r",\s*[^,]+$", "", regex=True)
    # если есть сокращение вида (ООО "Название") - берём его вместо полного наименования
    short_name = full_name.str.extract(r"\(((?:ООО|АО|ЗАО|ОАО|ПАО|ФГУП|НАО)[^()]*)\)")[0]
    d["manufacturer"] = short_name.fillna(full_name).str.replace(r",\s*[^,]+$", "", regex=True).str.strip().str.slice(0, 80)
    d["trade_name"] = d["trade_name"].astype(str).str.replace(r"[®™*]+", "", regex=True).str.strip()
    d["mnn"] = d["mnn"].astype(str).str.strip()

    # одна и та же позиция часто зарегистрирована несколькими строками (разные владельцы/штрихкоды) - схлопываем
    keys = ["trade_name", "mnn", "atc", "atc2", "form_group", "form_name", "dosage", "dose_value", "dose_unit", "pack_qty"]
    g = d.groupby(keys, dropna=False, as_index=False).agg(
        price_limit=("price_limit", "median"), manufacturer=("manufacturer", "first"),
        country=("country", "first"), pack_ambiguous=("pack_ambiguous", "max"), ean13=("ean13", "first"),
    )
    return g


def select_drugs(parsed: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Отбирает n позиций: чем больше у МНН торговых наименований в реестре, тем выше шанс попасть в выборку
    (распространённые молекулы в аптеке представлены шире), внутри - случайно."""
    weight = parsed.groupby("mnn")["trade_name"].transform("nunique") ** 0.7
    weight = weight * np.where(parsed["form_group"].isin(COUNTABLE_FORMS), 1.6, 1.0)
    # реестр ЖНВЛП перекошен в сторону рецептурных и госпитальных групп - выравниваем к розничному миксу аптеки
    weight = weight * parsed["atc"].str[0].map(RETAIL_ATC1_WEIGHT).fillna(1.0)
    idx = rng.choice(parsed.index, size=min(n, len(parsed)), replace=False, p=weight / weight.sum())
    d = parsed.loc[np.sort(idx)].copy()

    # розничная цена: предельная цена производителя + оптовая и розничная надбавки + НДС 10%, с разбросом
    markup = rng.uniform(1.25, 1.7, len(d))
    d["base_price"] = np.round(d["price_limit"] * markup * 1.1, 2)

    d["category"] = d["atc"].str[0].map(ATC1_CATEGORY)
    d["subcategory"] = d["atc2"].map(ATC2_NAME)
    d["product_class"] = "Лекарственные средства"
    d["doses_in_pack"] = np.where(d["form_group"].isin(COUNTABLE_FORMS), d["pack_qty"], np.nan)
    d["is_own_brand"] = False
    d["source"] = "ГРЛС, реестр ЖНВЛП"
    d["name"] = (d["trade_name"] + " " + d["form_name"] + " " + d["dosage"] + " №" + d["pack_qty"].astype(int).astype(str))
    d["name"] = d["name"].str.replace(r"\s+", " ", regex=True).str.strip()
    return d


# --- синтетические нелекарственные категории ---
# подкатегория: (формы, диапазон цены за упаковку, варианты фасовки, доля СТМ)
NON_DRUG = {
    "БАД и витамины": {
        "Витаминно-минеральные комплексы": (["таблетки", "капсулы", "саше"], (250, 1600), [30, 60, 90], 0.25),
        "Омега-3 и жирные кислоты": (["капсулы"], (350, 2500), [30, 60, 120], 0.25),
        "Пробиотики": (["капсулы", "саше", "порошок"], (300, 1400), [10, 20, 30], 0.2),
        "Для суставов и связок": (["таблетки", "саше", "порошок"], (500, 3200), [30, 60], 0.2),
        "Магний и минералы": (["таблетки", "порошок"], (200, 1100), [30, 50, 60], 0.3),
    },
    "Гигиена и косметика": {
        "Уход за полостью рта": (["паста", "ополаскиватель", "щетка"], (120, 700), [1], 0.3),
        "Лечебная косметика для лица": (["крем", "сыворотка", "гель"], (600, 4200), [1], 0.1),
        "Уход за телом": (["крем", "бальзам", "лосьон"], (250, 1800), [1], 0.2),
        "Средства для волос": (["шампунь", "бальзам", "маска"], (300, 2200), [1], 0.15),
        "Солнцезащитные средства": (["крем", "спрей"], (700, 3000), [1], 0.1),
    },
    "Мать и дитя": {
        "Детское питание": (["смесь", "каша", "пюре"], (90, 1900), [1], 0.0),
        "Подгузники и пеленки": (["подгузники", "пеленки"], (400, 2400), [1], 0.2),
        "Уход за малышом": (["крем", "присыпка", "масло"], (180, 900), [1], 0.2),
        "Для кормления": (["бутылочка", "соска", "молокоотсос"], (250, 4500), [1], 0.1),
    },
    "Медицинские изделия": {
        "Перевязочные средства": (["бинт", "пластырь", "салфетки"], (40, 450), [1, 10, 20], 0.35),
        "Медицинская техника": (["тонометр", "термометр", "ингалятор"], (350, 6500), [1], 0.1),
        "Тесты и диагностика": (["тест", "тест-полоски"], (120, 1700), [1, 2, 50], 0.1),
        "Ортопедия и компрессия": (["бандаж", "чулки", "стельки"], (600, 5500), [1], 0.05),
    },
    "Оптика": {
        "Контактные линзы": (["линзы"], (700, 3800), [6, 30], 0.1),
        "Растворы для линз": (["раствор"], (300, 1100), [1], 0.2),
        "Очки": (["очки"], (400, 3500), [1], 0.1),
    },
}
NON_GOODS = [("Пакет-майка", 8), ("Доставка заказа", 150), ("Подарочная упаковка", 60)]

BRAND_SYLLABLES = ["Вита", "Нео", "Био", "Эко", "Мед", "Гелио", "Фарма", "Лайт", "Норд", "Сана", "Аква",
                   "Дерма", "Бэби", "Орто", "Опти", "Актив", "Клин", "Прима", "Сибер", "Альфа"]
OWN_BRAND = "Аптечная марка"


def make_brands(rng, n):
    names = set()
    while len(names) < n:
        a, b = rng.choice(BRAND_SYLLABLES, 2, replace=False)
        names.add(a + b.lower())
    return sorted(names)


def generate_non_drugs(n: int, rng: np.random.Generator) -> pd.DataFrame:
    subcats = [(cat, sub, spec) for cat, subs in NON_DRUG.items() for sub, spec in subs.items()]
    brands = make_brands(rng, 60)
    rows = []
    for i in range(n):
        cat, sub, (forms, (lo, hi), packs, own_share) = subcats[i % len(subcats)]
        own = rng.random() < own_share
        brand = OWN_BRAND if own else rng.choice(brands)
        form = rng.choice(forms)
        pack = int(rng.choice(packs))
        price = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if own:
            price *= 0.75  # СТМ дешевле аналогов
        rows.append({
            "trade_name": f"{brand} {sub.split()[0].lower()}", "mnn": None, "atc": None, "atc2": None,
            "form_group": form if form in COUNTABLE_FORMS | {"порошок"} else "прочее", "form_name": form,
            "dosage": "", "dose_value": np.nan, "dose_unit": None, "pack_qty": pack,
            "price_limit": np.nan, "manufacturer": brand, "country": "Россия", "pack_ambiguous": False, "ean13": None,
            "base_price": round(price, 2), "category": cat, "subcategory": sub,
            "product_class": "Товары аптечного ассортимента",
            "doses_in_pack": pack if form in COUNTABLE_FORMS else np.nan,
            "is_own_brand": own, "source": "синтетика",
            "name": f"{brand} {form} ({sub.lower()})" + (f" №{pack}" if pack > 1 else ""),
        })
    return pd.DataFrame(rows)


def build_products(n_drugs: int, n_non_drugs: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parsed = parse_registry(load_registry())
    drugs = select_drugs(parsed, n_drugs, rng)
    non_drugs = generate_non_drugs(n_non_drugs, rng)
    non_goods = pd.DataFrame([{
        "trade_name": name, "name": name, "form_group": "прочее", "form_name": "услуга",
        "base_price": price, "category": "Нетоварные позиции", "subcategory": "Нетоварные позиции",
        "product_class": "Нетоварные позиции", "is_own_brand": False, "source": "синтетика", "pack_qty": 1,
        "pack_ambiguous": False,
    } for name, price in NON_GOODS])

    p = pd.concat([drugs, non_drugs, non_goods], ignore_index=True)

    # минимальный ассортимент (защищённые позиции для скоринга): таблетированные формы самых
    # распространённых МНН - условный аналог обязательного перечня, признак синтетический
    top_mnn = drugs["mnn"].value_counts().head(25).index
    p["is_min_assortment"] = p["mnn"].isin(top_mnn) & p["form_group"].isin(["таблетки", "капсулы"])

    p.insert(0, "apCode", [f"AP{100000 + i}" for i in range(len(p))])
    cols = ["apCode", "name", "trade_name", "mnn", "manufacturer", "country", "atc", "atc2", "product_class",
            "category", "subcategory", "form_group", "form_name", "dosage", "dose_value", "dose_unit", "pack_qty",
            "doses_in_pack", "pack_ambiguous", "price_limit", "base_price", "is_own_brand", "is_min_assortment",
            "ean13", "source"]
    p = p[cols]
    p["pack_qty"] = p["pack_qty"].astype(int)
    p["ean13"] = p["ean13"].map(lambda x: None if pd.isna(x) else str(int(x)) if isinstance(x, float) else str(x).strip())
    p["mnn"] = p["mnn"].where(p["mnn"].notna(), None)
    return p
