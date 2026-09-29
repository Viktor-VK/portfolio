"""Синтетическая аптечная сеть: точки продаж, чеки, остатки, текущая ассортиментная матрица.

В каждую точку заложен скрытый архетип (центральная аптека, аптека у дома, дискаунтер и т.д.) - он задаёт
площадь, трафик, ценовой уклон покупателей и категорийный микс. Архетип хранится отдельной таблицей
`store_archetypes` только для финальной проверки: ноутбуки его не используют при кластеризации.
"""
import numpy as np
import pandas as pd

# --- архетипы точек ---
# share - доля точек сети; square - медиана площади торгового зала, м2; traffic - чеков в день;
# lines - среднее число позиций в чеке; theta - ценовой уклон (>0 - берут дорогое внутри подкатегории);
# markup - уровень цен точки относительно базового; mix - множители к базовому категорийному миксу
ARCHETYPES = {
    "Центральная аптека": dict(share=0.08, square=190, traffic=430, lines=2.3, theta=0.15, markup=1.00,
                               oft=0.85, location={"центр города": 0.8, "ТЦ": 0.2}, mix={}),
    "Классическая аптека": dict(share=0.22, square=90, traffic=210, lines=1.9, theta=0.0, markup=1.00,
                                oft=0.45, location={"центр города": 0.35, "спальный район": 0.65}, mix={}),
    "Аптека у дома": dict(share=0.19, square=45, traffic=140, lines=1.6, theta=-0.25, markup=1.00,
                          oft=0.15, location={"спальный район": 1.0},
                          mix={"Дыхательная система": 1.9, "Нервная система": 1.6, "Костно-мышечная система": 1.6,
                               "Гигиена и косметика": 0.4, "Мать и дитя": 0.6, "Оптика": 0.5, "БАД и витамины": 0.7}),
    "Дискаунтер": dict(share=0.12, square=75, traffic=330, lines=1.9, theta=-0.9, markup=0.90,
                       oft=0.3, location={"спальный район": 0.6, "центр города": 0.4},
                       mix={"Сердечно-сосудистая система": 1.8, "Пищеварение и обмен веществ": 1.4,
                            "Кровь и кроветворение": 1.6, "Гигиена и косметика": 0.35, "БАД и витамины": 0.6}),
    "Премиум-аптека": dict(share=0.08, square=85, traffic=150, lines=2.1, theta=0.9, markup=1.12,
                           oft=0.9, location={"центр города": 0.6, "ТЦ": 0.4},
                           mix={"Гигиена и косметика": 3.5, "БАД и витамины": 2.0, "Мать и дитя": 1.3,
                                "Дерматология": 1.8, "Сердечно-сосудистая система": 0.5, "Кровь и кроветворение": 0.6}),
    "Аптека в ТЦ": dict(share=0.10, square=125, traffic=370, lines=1.7, theta=0.2, markup=1.03,
                        oft=0.95, location={"ТЦ": 1.0},
                        mix={"Гигиена и косметика": 1.8, "Мать и дитя": 3.2, "Медицинские изделия": 1.6,
                             "Сердечно-сосудистая система": 0.5}),
    "Сельская аптека": dict(share=0.12, square=35, traffic=70, lines=1.8, theta=-0.5, markup=0.97,
                            oft=0.05, location={"село": 1.0},
                            mix={"Сердечно-сосудистая система": 2.3, "Кровь и кроветворение": 1.8,
                                 "Пищеварение и обмен веществ": 1.3, "Гигиена и косметика": 0.3, "Оптика": 0.1,
                                 "Мать и дитя": 0.5, "БАД и витамины": 0.6}),
    "Аптека при ЛПУ": dict(share=0.06, square=30, traffic=110, lines=1.6, theta=0.1, markup=1.00,
                           oft=0.1, location={"при поликлинике": 1.0},
                           mix={"Противомикробные": 3.0, "Иммуномодуляторы и онкология": 3.0,
                                "Кровь и кроветворение": 1.8, "Гормоны системного действия": 2.0,
                                "Мочеполовая система и половые гормоны": 1.5, "Гигиена и косметика": 0.3,
                                "Мать и дитя": 0.4, "БАД и витамины": 0.6}),
    "Оптика": dict(share=0.03, square=40, traffic=35, lines=1.3, theta=0.3, markup=1.05, type_id=4,
                   oft=1.0, location={"ТЦ": 0.5, "центр города": 0.5},
                   mix={"Оптика": 60.0}),
}

# базовый категорийный микс (вероятность, что позиция чека из этой категории) до множителей архетипа
BASE_MIX = {
    "Дыхательная система": 0.10, "Нервная система": 0.08, "Пищеварение и обмен веществ": 0.10,
    "Сердечно-сосудистая система": 0.09, "Костно-мышечная система": 0.06, "Противомикробные": 0.05,
    "Кровь и кроветворение": 0.04, "Мочеполовая система и половые гормоны": 0.03, "Дерматология": 0.03,
    "Органы чувств": 0.02, "Гормоны системного действия": 0.01, "Иммуномодуляторы и онкология": 0.005,
    "Прочие препараты": 0.005, "БАД и витамины": 0.10, "Гигиена и косметика": 0.08, "Мать и дитя": 0.05,
    "Медицинские изделия": 0.06, "Оптика": 0.01,
}

BRANDS_BY_ARCHETYPE = {  # вероятности бренда сети для архетипа: бренды частично совпадают с форматом
    "Дискаунтер": {"Бренд 2": 0.8, "Бренд 1": 0.2},
    "Сельская аптека": {"Бренд 2": 0.5, "Бренд 1": 0.3, "Бренд 4": 0.2},
    "Премиум-аптека": {"Бренд 3": 0.85, "Бренд 1": 0.15},
    "Оптика": {"Бренд 5": 1.0},
}
DEFAULT_BRANDS = {"Бренд 1": 0.55, "Бренд 4": 0.3, "Бренд 3": 0.15}
SALE_TYPES = {"Розница": 0.87, "Интернет-заказ (самовывоз)": 0.08, "Маркетплейс": 0.035, "Опт": 0.015}


def _pick(rng, options: dict, size=None):
    keys = list(options)
    p = np.array(list(options.values()), dtype=float)
    return rng.choice(keys, size=size, p=p / p.sum())


def generate_stores(n: int, calc_date: pd.Timestamp, rng: np.random.Generator):
    names = list(ARCHETYPES)
    shares = np.array([ARCHETYPES[a]["share"] for a in names])
    counts = np.floor(shares / shares.sum() * n).astype(int)
    counts[np.argsort(-shares)[: n - counts.sum()]] += 1  # добиваем до n крупнейшими архетипами
    archetype = np.repeat(names, counts)
    rng.shuffle(archetype)

    rows = []
    for i, a in enumerate(archetype):
        spec = ARCHETYPES[a]
        square = float(np.clip(rng.lognormal(np.log(spec["square"]), 0.22), 12, 450))
        open_date = calc_date - pd.Timedelta(days=int(rng.uniform(60, 12 * 365)))
        close_date = pd.NaT
        if rng.random() < 0.05:  # часть точек уже закрыта - их отсекает фильтр действующих
            close_date = calc_date - pd.Timedelta(days=int(rng.uniform(10, 300)))
        # бизнес-единицы - условные регионы; сельские точки сосредоточены в БЕ 4-6
        bu = f"БЕ {rng.integers(4, 7) if a == 'Сельская аптека' else rng.integers(1, 7)}"
        rows.append({
            "TradePointId": 1001 + i,
            "TradePointCode": f"TP{1001 + i:04d}",
            "BrandName": _pick(rng, BRANDS_BY_ARCHETYPE.get(a, DEFAULT_BRANDS)),
            "BusinessUnit": bu,
            "TradePointTypeId": spec.get("type_id", 1),
            "TradePointType": "Оптика" if spec.get("type_id") == 4 else "Аптека",
            "TradeSquare": round(square, 1),
            "Layout": "ОФТ" if rng.random() < spec["oft"] else "ЗФТ",
            "Location": _pick(rng, spec["location"]),
            "OpenDate": open_date.normalize(),
            "CloseDate": close_date,
            "IsVip": a == "Премиум-аптека" or (a == "Центральная аптека" and rng.random() < 0.3),
            "IsDiscounter": a == "Дискаунтер",
        })
    stores = pd.DataFrame(rows)
    stores["StoreName"] = stores["TradePointType"] + " " + stores["TradePointCode"]

    # текущая категория точки (действующая классификация сети): просто по площади, 5 уровней
    stores["CurrentCategory"] = pd.qcut(stores["TradeSquare"].rank(method="first"), 5,
                                        labels=["E", "D", "C", "B", "A"]).astype(str)
    truth = pd.DataFrame({"TradePointId": stores["TradePointId"], "archetype": archetype})
    return stores, truth


def product_behaviour(products: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Скрытые свойства товаров: популярность (длинный хвост) и относительная цена внутри подкатегории."""
    p = products[["apCode", "category", "subcategory", "base_price", "product_class"]].copy()
    p["popularity"] = rng.lognormal(0, 1.1, len(p))
    med = p.groupby("subcategory")["base_price"].transform("median")
    p["rel_price"] = np.log(p["base_price"] / med).clip(-2.5, 2.5)
    p["cost"] = p["base_price"] / rng.uniform(1.25, 1.5, len(p))  # закупочная цена
    return p


def generate_cheques(stores, truth, products, behaviour, date_from, days, cheque_frac, rng):
    """Чеки (построчно). Для каждой точки: число чеков ~ трафик x дни x доля выборки, в чеке 1+Пуассон позиций,
    товар выбирается по категорийному миксу архетипа, популярности и ценовому уклону покупателей."""
    goods = behaviour[behaviour["product_class"] != "Нетоварные позиции"].reset_index(drop=True)
    non_goods = behaviour[behaviour["product_class"] == "Нетоварные позиции"].reset_index(drop=True)
    cats = goods["category"].to_numpy()
    arche = dict(zip(truth["TradePointId"], truth["archetype"]))

    frames, cheque_id = [], 5_000_000
    for store in stores.itertuples():
        spec = ARCHETYPES[arche[store.TradePointId]]
        # точка открылась позже начала периода или закрылась раньше - продаёт только в свои дни
        start = max(pd.Timestamp(date_from), store.OpenDate)
        end = pd.Timestamp(date_from) + pd.Timedelta(days=days - 1)
        if pd.notna(store.CloseDate):
            end = min(end, store.CloseDate)
        active_days = (end - start).days + 1
        if active_days <= 0:
            continue
        traffic = spec["traffic"] * rng.lognormal(0, 0.25) * (store.TradeSquare / spec["square"]) ** 0.4
        n_cheques = rng.poisson(traffic * active_days * cheque_frac)
        if n_cheques == 0:
            continue

        mix = {c: BASE_MIX.get(c, 0.0) * spec["mix"].get(c, 1.0) for c in BASE_MIX}
        cat_w = np.array([mix.get(c, 0.0) for c in cats]) * rng.lognormal(0, 0.15, len(cats))
        carry = rng.random(len(goods)) < np.clip(0.25 + 0.5 * (store.TradeSquare / 150) ** 0.5
                                                 + 0.15 * np.log1p(goods["popularity"]), 0.05, 1.0)
        w = goods["popularity"].to_numpy() * np.exp(spec["theta"] * goods["rel_price"].to_numpy()) * carry
        # нормируем внутри категории, затем умножаем на долю категории в миксе
        cat_sum = pd.Series(w).groupby(cats).transform("sum").to_numpy()
        prob = np.where(cat_sum > 0, w / np.where(cat_sum > 0, cat_sum, 1) * cat_w, 0.0)
        prob = prob / prob.sum()

        n_lines = 1 + rng.poisson(spec["lines"] - 1, n_cheques)
        line_cheque = np.repeat(np.arange(n_cheques), n_lines)
        idx = rng.choice(len(goods), size=len(line_cheque), p=prob)
        amount = rng.choice([1, 1, 1, 1, 1, 1, 2, 2, 3], size=len(idx))
        day = rng.integers(0, active_days, n_cheques)
        sec = rng.normal(15 * 3600, 3.2 * 3600, n_cheques).clip(8 * 3600, 22 * 3600 - 1).astype(int)
        sale_type = _pick(rng, SALE_TYPES, n_cheques)

        f = pd.DataFrame({
            "ChequeId": cheque_id + line_cheque,
            "TradePointId": store.TradePointId,
            "apCode": goods["apCode"].to_numpy()[idx],
            "Date": (start + pd.to_timedelta(day[line_cheque], unit="D")).normalize(),
            "Time": pd.to_datetime(sec[line_cheque], unit="s").strftime("%H:%M:%S"),
            "Amount": amount,
            "price": goods["base_price"].to_numpy()[idx] * spec["markup"] * rng.normal(1, 0.03, len(idx)),
            "cost": goods["cost"].to_numpy()[idx],
            "SaleType": sale_type[line_cheque],
        })
        # нетоварные позиции (пакет, доставка) в части чеков - их отсекают бизнес-фильтры
        extra = rng.random(n_cheques) < 0.04
        if extra.any() and len(non_goods):
            k = rng.integers(0, len(non_goods), extra.sum())
            ng = pd.DataFrame({
                "ChequeId": cheque_id + np.flatnonzero(extra), "TradePointId": store.TradePointId,
                "apCode": non_goods["apCode"].to_numpy()[k],
                "Date": (start + pd.to_timedelta(day[extra], unit="D")).normalize(),
                "Time": pd.to_datetime(sec[extra], unit="s").strftime("%H:%M:%S"),
                "Amount": 1, "price": non_goods["base_price"].to_numpy()[k], "cost": 0.0,
                "SaleType": sale_type[extra],
            })
            f = pd.concat([f, ng], ignore_index=True)
        frames.append(f)
        cheque_id += n_cheques

    df = pd.concat(frames, ignore_index=True).sort_values(["ChequeId"], kind="stable")
    df["ChequePos"] = df.groupby("ChequeId").cumcount() + 1
    df["Sum_Price"] = (df["price"] * df["Amount"]).round(2)
    discount = np.where(rng.random(len(df)) < 0.3, rng.uniform(0.03, 0.15, len(df)), 0.0)
    df["Sum_Fact"] = (df["Sum_Price"] * (1 - discount)).round(2)
    df["Sum_Purch"] = (df["cost"] * df["Amount"]).round(2)
    # остаточный срок годности на момент продажи: у части позиций меньше 6 месяцев (уценка) - их отсекают фильтры
    df["ShelfLifeMonths"] = np.where(rng.random(len(df)) < 0.03, rng.integers(1, 6, len(df)), rng.integers(6, 36, len(df)))
    cols = ["ChequeId", "ChequePos", "TradePointId", "apCode", "Date", "Time", "Amount",
            "Sum_Price", "Sum_Fact", "Sum_Purch", "SaleType", "ShelfLifeMonths"]
    return df[cols].reset_index(drop=True)


def generate_stock(stores, cheques, behaviour, snapshot_date, rng, max_rows=None):
    """Снимок остатков на дату: по каждой паре точка-товар, которая продавалась, плюс немного непродававшегося
    товара (залежи). Количество - несколько дней продаж со случайным страховым запасом."""
    sold = cheques.merge(behaviour[["apCode", "product_class"]], on="apCode")
    sold = sold[sold["product_class"] != "Нетоварные позиции"]
    s = sold.groupby(["TradePointId", "apCode"], as_index=False)["Amount"].sum()
    s["Amount"] = np.maximum(1, np.round(s["Amount"] * rng.uniform(0.5, 3.0, len(s)))).astype(int)

    goods = behaviour[behaviour["product_class"] != "Нетоварные позиции"]
    dead = []
    for tp in stores["TradePointId"]:
        k = rng.integers(3, 15)
        dead.append(pd.DataFrame({"TradePointId": tp, "apCode": rng.choice(goods["apCode"], k, replace=False),
                                  "Amount": rng.integers(1, 5, k)}))
    s = pd.concat([s] + dead, ignore_index=True).drop_duplicates(["TradePointId", "apCode"])
    if max_rows and len(s) > max_rows:
        s = s.sample(max_rows, random_state=int(rng.integers(1e9)))
    s = s.merge(behaviour[["apCode", "base_price", "cost"]], on="apCode")
    s["Sum_Purch"] = (s["Amount"] * s["cost"]).round(2)
    s["Sum_Price"] = (s["Amount"] * s["base_price"]).round(2)
    s["Date"] = pd.Timestamp(snapshot_date)
    return s[["Date", "TradePointId", "apCode", "Amount", "Sum_Purch", "Sum_Price"]].sort_values(
        ["TradePointId", "apCode"]).reset_index(drop=True)


def generate_fact_matrix(stores, products, behaviour, rng):
    """Текущая (фактическая) ассортиментная матрица сети: по действующей категории точки A-E.
    Чем крупнее категория, тем шире матрица; в матрицу попадают в основном популярные товары, но с шумом -
    действующая матрица неидеальна, поэтому новому расчёту есть что улучшать."""
    goods = behaviour[behaviour["product_class"] != "Нетоварные позиции"].reset_index(drop=True)
    width = {"A": 0.55, "B": 0.45, "C": 0.37, "D": 0.3, "E": 0.22}
    rows = []
    noisy_score = np.log(goods["popularity"]) + rng.normal(0, 1.4, len(goods))
    for cat, share in width.items():
        k = int(len(goods) * share)
        top = goods["apCode"].to_numpy()[np.argsort(-noisy_score.to_numpy())[:k]]
        rows.append(pd.DataFrame({"CurrentCategory": cat, "apCode": top}))
    m = pd.concat(rows, ignore_index=True)
    mins = products.loc[products["is_min_assortment"], "apCode"]
    extra = pd.DataFrame([(c, a) for c in width for a in mins], columns=["CurrentCategory", "apCode"])
    return pd.concat([m, extra]).drop_duplicates().sort_values(["CurrentCategory", "apCode"]).reset_index(drop=True)
