"""Синтетическая сеть электроники в структуре выгрузок 1С - для проекта «Восстановление истории остатков».

Каталог - смартфоны Apple (реальные названия моделей, цены условные). Движение товара моделируется по дням:
спрос покупателей, заказы филиалов у своего РРЦ со сроком доставки, пополнение РРЦ с центрального склада,
перебои у поставщика, выход новой линейки в сентябре. Из симуляции получаются ровно те таблицы, которые
отдаёт 1С: журнал движений (счёт 41), реализация (счёт 90) и текущий остаток. Истинная дневная история
остатков и спрос сохраняются отдельно - только для проверки восстановления в ноутбуке.
"""
import uuid

import numpy as np
import pandas as pd

PERIOD_START = pd.Timestamp("2025-08-01")
NEW_LINE_RELEASE = pd.Timestamp("2025-09-26")  # поступление новой линейки в продажу

# модель: (статус, цена за базовую память, варианты памяти и надбавки, цвета, относительный спрос)
CATALOG = {
    "iPhone 15": ("Выводимый", 62000, {"128GB": 0, "256GB": 9000}, ["Black", "Blue", "Pink"], 0.8),
    "iPhone 15 Plus": ("Выводимый", 70000, {"128GB": 0}, ["Black", "Green"], 0.35),
    "iPhone 16e": ("Основной", 56000, {"128GB": 0, "256GB": 10000}, ["Black", "White"], 0.9),
    "iPhone 16": ("Основной", 76000, {"128GB": 0, "256GB": 10000, "512GB": 29000}, ["Black", "White", "Ultramarine"], 1.4),
    "iPhone 16 Plus": ("Основной", 86000, {"128GB": 0, "256GB": 10000}, ["Black", "Teal"], 0.45),
    "iPhone 16 Pro": ("Основной", 101000, {"128GB": 0, "256GB": 11000, "512GB": 30000}, ["Black Titanium", "Desert Titanium"], 1.0),
    "iPhone 16 Pro Max": ("Основной", 121000, {"256GB": 0, "512GB": 20000}, ["Black Titanium", "Desert Titanium"], 0.9),
    "iPhone 17": ("Новинка", 92000, {"256GB": 0, "512GB": 20000}, ["Black", "Lavender", "Sage"], 1.5),
    "iPhone Air": ("Новинка", 117000, {"256GB": 0, "512GB": 20000}, ["Space Black", "Sky Blue"], 0.6),
    "iPhone 17 Pro": ("Новинка", 127000, {"256GB": 0, "512GB": 20000, "1TB": 50000}, ["Cosmic Orange", "Deep Blue", "Silver"], 1.3),
    "iPhone 17 Pro Max": ("Новинка", 142000, {"256GB": 0, "512GB": 20000, "1TB": 50000}, ["Cosmic Orange", "Deep Blue"], 1.2),
}
CITIES = ["Москва", "Санкт-Петербург", "Екатеринбург", "Новосибирск", "Казань", "Нижний Новгород", "Самара",
          "Краснодар", "Ростов-на-Дону", "Воронеж", "Пермь", "Уфа"]


def _ref(rng):
    """Ссылка в стиле 1С - GUID."""
    return str(uuid.UUID(int=int(rng.integers(0, 2 ** 63)) << 64 | int(rng.integers(0, 2 ** 63))))


def build_products(n_max, rng):
    rows = []
    for model, (status, base, mem, colors, pop) in CATALOG.items():
        for m, extra in mem.items():
            for c in colors:
                # у старших объёмов памяти спрос ниже
                rows.append(dict(model=model, Товар=f"Смартфон Apple {model} {m} {c}", Ассортиментный_Статус=status,
                                 price=base + extra, pop=pop * (0.55 if extra >= 20000 else 1.0) * rng.lognormal(0, 0.25)))
    p = pd.DataFrame(rows)
    if n_max and len(p) > n_max:
        p = p.sort_values("pop", ascending=False).head(n_max)
    p = p.reset_index(drop=True)
    p["ТоварСсылка"] = [_ref(rng) for _ in range(len(p))]
    p["Код"] = [f"{5_100_000 + i * 17}" for i in range(len(p))]
    p["Категория1"], p["Категория2"], p["Категория3"] = "Смартфоны и гаджеты", "Смартфоны", "Смартфоны по брендам"
    p["Категория4"] = "Apple Смартфоны"
    return p


def build_branches(n_stores, rng):
    n_rrc = max(1, n_stores // 12)
    rows = [dict(Филиал="Центральный склад", Вид_филиала="Склад", size=0)]
    for r in range(n_rrc):
        rows.append(dict(Филиал=f"РРЦ {CITIES[r % len(CITIES)]}", Вид_филиала="РРЦ", size=1.2))
    for i in range(n_stores):
        kind = "Дисконт центр" if i % 9 == 4 else "Магазин"
        city = CITIES[int(rng.integers(0, len(CITIES)))]
        rows.append(dict(Филиал=f"{kind} {city} {i + 1:03d}", Вид_филиала=kind,
                         size=float(rng.lognormal(0, 0.45)) * (0.6 if kind == "Дисконт центр" else 1.0)))
    rows.append(dict(Филиал="Офис продаж B2B", Вид_филиала="Офис", size=0))
    b = pd.DataFrame(rows)
    b["ФилиалСсылка"] = [_ref(rng) for _ in range(len(b))]
    rrc = b[b.Вид_филиала == "РРЦ"]["Филиал"].tolist()
    b["ПриписанК_Складу"] = [
        "Центральный склад" if k == "РРЦ" else (rrc[int(rng.integers(0, len(rrc)))] if k in ("Магазин", "Дисконт центр") else None)
        for k in b.Вид_филиала]
    return b


def simulate(products, branches, days, rng):
    dates = pd.date_range(PERIOD_START, periods=days, freq="D")
    P, B = len(products), len(branches)
    kind = branches.Вид_филиала.to_numpy()
    is_store = np.isin(kind, ["Магазин", "Дисконт центр", "РРЦ"])
    wh = int(np.flatnonzero(kind == "Склад")[0])
    rrc_idx = {n: i for i, n in enumerate(branches.Филиал)}
    parent = np.array([rrc_idx.get(x, -1) if x else -1 for x in branches.ПриписанК_Складу])

    status = products.Ассортиментный_Статус.to_numpy()
    price = products.price.to_numpy()
    cost = np.round(price * 0.82, -1)
    base = 0.18 * np.outer(branches["size"].to_numpy(), products["pop"].to_numpy())   # спрос шт/день
    release_day = (NEW_LINE_RELEASE - PERIOD_START).days

    # политика запасов филиала (s, S): точка заказа ~ 4 дня спроса, максимум ~ 10 дней; РРЦ держит больше
    s_lvl = np.ceil(base * 6 + 1)
    S_lvl = np.ceil(base * 14 + 2)
    # РРЦ держит запас на свои продажи и на пополнение приписанных магазинов
    for r in np.flatnonzero(kind == "РРЦ"):
        kids = parent == r
        s_lvl[r] = np.ceil((base[r] + base[kids].sum(axis=0)) * 9 + 2)
        S_lvl[r] = np.ceil((base[r] + base[kids].sum(axis=0)) * 20 + 4)
    stock = np.where(is_store[:, None], np.round(S_lvl * rng.uniform(0.4, 1.0, (B, P))), 0).astype(int)
    stock[:, status == "Новинка"] = 0
    stock[wh] = 10 ** 5  # центральный склад: условно неограниченный запас
    pipeline = []  # (день прибытия, филиал, товар, количество)

    # перебои у поставщика: у части товаров центральный склад пуст несколько недель
    outage = np.zeros((days, P), dtype=bool)
    for p in rng.choice(P, size=max(1, P // 6), replace=False):
        st = int(rng.integers(5, days - 10))
        outage[st: st + int(rng.integers(6, 15)), p] = True

    true_rows, true_demand, moves, sales90 = [], [], [], []
    for t, d in enumerate(dates):
        # спрос: новинки появляются в день релиза с ажиотажем, выводимые модели постепенно затухают
        mult = np.ones(P)
        new = status == "Новинка"
        if t < release_day:
            mult[new] = 0
        else:
            mult[new] = 1 + 1.5 * np.exp(-(t - release_day) / 10)
            mult[status == "Выводимый"] = 0.55
            mult[status == "Основной"] = 0.85
        weekly = 1.25 if d.dayofweek >= 5 else 0.93
        demand = rng.poisson(base * mult * weekly) * is_store[:, None]

        true_rows.append(stock.copy())  # остаток на начало дня
        true_demand.append(demand.copy())

        arrivals = np.zeros((B, P), dtype=int)
        rest = []
        for day, b, p, q in pipeline:
            if day == t:
                arrivals[b, p] += q
            else:
                rest.append((day, b, p, q))
        pipeline = rest
        for b, p in zip(*np.nonzero(arrivals)):
            moves.append((d, 0, p, b, arrivals[b, p]))
        stock += arrivals

        sold = np.minimum(demand, np.where(is_store[:, None], stock, 0))
        stock -= sold
        for b, p in zip(*np.nonzero(sold)):
            moves.append((d, 1, p, b, sold[b, p]))
            sales90.append((d, p, b, sold[b, p]))
        # редкие списания (брак, витринный образец)
        wo = (rng.random((B, P)) < 0.0015) & (stock > 0)
        for b, p in zip(*np.nonzero(wo)):
            moves.append((d, 1, p, b, 1))
        stock -= wo

        # заказы: сначала РРЦ у центрального склада, затем магазины у своего РРЦ (перемещение со сроком в пути)
        in_transit = np.zeros((B, P), dtype=int)
        for day, b, p, q in pipeline:
            in_transit[b, p] += q
        position = stock + in_transit
        # новую линейку начинают завозить за неделю до старта продаж, чтобы к релизу она была на полках
        allowed = np.ones(P, dtype=bool) if t >= release_day - 8 else (status != "Новинка")
        for b in np.flatnonzero(is_store):
            src = wh if kind[b] == "РРЦ" else parent[b]
            need = np.flatnonzero((position[b] <= s_lvl[b]) & allowed & (status != "Выводимый") | (
                (position[b] <= 0) & (status == "Выводимый") & (rng.random(P) < 0.05)))
            for p in need:
                q = int(S_lvl[b, p] - position[b, p])
                if src == wh and outage[t, p]:
                    continue
                q = min(q, int(stock[src, p]))
                if q <= 0:
                    continue
                lead = int(rng.integers(5, 9)) if src == wh else int(rng.integers(1, 4))
                stock[src, p] -= q
                moves.append((d, 1, p, src, q))
                pipeline.append((t + lead, b, p, q))
                position[b, p] += q

    true_rows = np.stack(true_rows)  # дни x филиалы x товары
    true_demand = np.stack(true_demand)
    in_transit = np.zeros((B, P), dtype=int)
    for day, b, p, q in pipeline:
        in_transit[b, p] += q
    return dates, true_rows, true_demand, stock, in_transit, moves, sales90, cost


def generate(n_stores, n_products, days, seed):
    rng = np.random.default_rng(seed)
    products = build_products(n_products, rng)
    branches = build_branches(n_stores, rng)
    dates, true_stock, true_demand, stock_end, in_transit, moves, sales90, cost = simulate(products, branches, days, rng)
    pref, bref = products.ТоварСсылка.to_numpy(), branches.ФилиалСсылка.to_numpy()

    m = pd.DataFrame(moves, columns=["Дата", "ВидДвижения", "p", "b", "Количество"])
    m["Сумма"] = (m.Количество * cost[m.p]).round(2)
    m = m.groupby(["Дата", "ВидДвижения", "p", "b"], as_index=False)[["Количество", "Сумма"]].sum()
    m["Номенклатура"], m["Филиал"] = pref[m.p], bref[m.b]
    schet_41 = m[["Дата", "ВидДвижения", "Номенклатура", "Филиал", "Количество", "Сумма"]]

    s = pd.DataFrame(sales90, columns=["Дата", "p", "b", "Количество"])
    s["СуммаСебес"] = (s.Количество * cost[s.p]).round(2)
    s["Номенклатура"], s["Филиал"] = pref[s.p], bref[s.b]
    schet_90 = s[["Дата", "Номенклатура", "Филиал", "Количество", "СуммаСебес"]]

    bb, pp = np.nonzero((stock_end > 0) | (in_transit > 0))
    now = pd.DataFrame({"Номенклатура": pref[pp], "Филиал": bref[bb], "Остаток": stock_end[bb, pp],
                        "Транзит": in_transit[bb, pp]})
    rng2 = np.random.default_rng(seed + 1)
    now["Резерв"] = np.minimum(now.Остаток, rng2.binomial(1, 0.15, len(now)))
    now["МягкийРезерв"] = np.minimum(now.Остаток - now.Резерв, rng2.binomial(1, 0.08, len(now)))
    now["Путь"] = 0
    now["ДатаОстатка"] = dates[-1]
    stock_now = now[["ДатаОстатка", "Номенклатура", "Филиал", "Остаток", "Резерв", "МягкийРезерв", "Транзит", "Путь"]]

    # цены: приоритет ФЦ ОРП -> РФЦ обычная -> РФЦ от себестоимости, часть видов цен не заполнена
    price = products.price.to_numpy()
    prices = pd.DataFrame({
        "Номенклатура": pref,
        "ФЦ_ОРП": np.where(rng.random(len(pref)) < 0.7, price, np.nan),
        "РФЦ_обычная": np.where(rng.random(len(pref)) < 0.8, price * 1.03, np.nan),
        "РФЦ_от_себестоимости": np.round(price * 0.82 * 1.18, -1),
    })

    D, B, P = true_stock.shape
    idx = np.nonzero((true_stock > 0) | (true_demand > 0))
    truth = pd.DataFrame({"Дата": dates[idx[0]], "Филиал": bref[idx[1]], "Номенклатура": pref[idx[2]],
                          "ОстатокНаНачалоДня": true_stock[idx], "СпросИстинный": true_demand[idx]})
    truth = truth[branches.Вид_филиала.to_numpy()[idx[1]] != "Склад"]

    prod_out = products[["ТоварСсылка", "Код", "Товар", "Категория1", "Категория2", "Категория3", "Категория4",
                         "Ассортиментный_Статус"]]
    branch_out = branches[["ФилиалСсылка", "Филиал", "Вид_филиала", "ПриписанК_Складу"]]
    return {"products": prod_out, "branches": branch_out, "prices": prices, "schet_41": schet_41,
            "schet_90": schet_90, "stock_now": stock_now, "stock_daily_true": truth.reset_index(drop=True)}
