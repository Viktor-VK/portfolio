# -*- coding: utf-8 -*-
"""Синтетическая цепь поставок сети электроники - для проекта «Сквозной анализ цепи поставок».

Схема движения товара: поставщик -> склады -> магазины.
- Закуп раз в неделю (понедельник): закупщик заказывает поставщику товар на каждый склад, поставщик
  подтверждает объём (при дефиците - меньше заказанного), товар приходит на склад через 2 дня.
- Склад пополняет свои магазины каждый день по их заявкам (политика «точка заказа - максимум»),
  товар приходит в магазин на следующий день.
- Покупатели приходят в магазины; если товара нет, продажа теряется.

В симуляцию специально заложены проблемы на каждом этапе цепочки (таблица truth_events):
дефицит у поставщика, недозаказ и перезаказ закупщиком, перекос закупа между складами, пропуски пополнения
магазина и перезатарка отдельных магазинов. Ноутбук должен найти их по «учётным» таблицам - заказам
поставщику, журналу движений и текущему остатку, - истина используется только для проверки.
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

LEAD_SUPPLIER = 2   # дней от поставщика до склада
LEAD_STORE = 1      # дней от склада до магазина
ORDER_WEEKDAY = 0   # закуп по понедельникам
STORE_MIN, STORE_MAX = 2, 5  # магазин заказывает при запасе на 2 дня продаж, до 5 дней
COVER_DAYS = 10     # закупщик держит на складе запас на 10 дней отгрузок: неделя цикла + доставка + страховой

# склады и города их магазинов
WAREHOUSES = {
    "Склад Москва": ["Москва", "Санкт-Петербург", "Воронеж", "Нижний Новгород", "Краснодар", "Ростов-на-Дону"],
    "Склад Екатеринбург": ["Екатеринбург", "Пермь", "Уфа", "Казань", "Самара"],
    "Склад Новосибирск": ["Новосибирск", "Омск", "Красноярск"],
}
WH_SHARE = [0.45, 0.35, 0.20]  # доля магазинов по складам


def build_branches(n_stores, rng):
    rows = [dict(Филиал=w, Вид_филиала="Склад", ПриписанК_Складу=None, size=0.0) for w in WAREHOUSES]
    whs = list(WAREHOUSES)
    counts = np.maximum(1, np.round(np.array(WH_SHARE) * n_stores)).astype(int)
    counts[0] += n_stores - counts.sum()
    i = 0
    for w, n in zip(whs, counts):
        for _ in range(n):
            i += 1
            city = WAREHOUSES[w][int(rng.integers(0, len(WAREHOUSES[w])))]
            rows.append(dict(Филиал=f"Магазин {city} {i:03d}", Вид_филиала="Магазин", ПриписанК_Складу=w,
                             size=float(rng.lognormal(0, 0.45))))
    b = pd.DataFrame(rows)
    b["ФилиалСсылка"] = [_ref(rng) for _ in range(len(b))]
    return b


def plan_events(W, M, P, days, dates, status, parent, rng):
    """Заранее разыгрывает заложенные проблемы. Возвращает множители и список событий для truth_events."""
    ev = []
    mondays = [t for t in range(days) if dates[t].dayofweek == ORDER_WEEKDAY]
    release = (NEW_LINE_RELEASE - PERIOD_START).days
    buy_mult = np.ones((days, W, P))      # множитель к «правильному» заказу закупщика
    supply_frac = np.ones((days, P))      # доля заказа, которую поставщик может подтвердить
    skip = np.zeros((days, M, P), bool)   # магазин не заказывает (пропуск пополнения)
    s_mult = np.ones((days, M, P))        # множитель к максимуму запаса магазина (перезатарка)
    favor = np.full((days, P), -1)       # склад, которому закупщик отдал дефицитный товар в первую очередь
    active = np.ones(P, bool)  # проблемы закладываются для любых моделей, статус не важен
    outages = []

    # поставщик: у части товаров 1-3 недели дефицита; новая линейка первые 2 недели - квота
    for p in rng.choice(np.flatnonzero(active & (status != "Новинка")), size=max(1, P // 4), replace=False):
        t0 = mondays[int(rng.integers(1, len(mondays) - 3))]
        t1 = t0 + 7 * int(rng.integers(1, 4))
        f = float(rng.uniform(0, 0.4))
        supply_frac[t0:t1, p] = f
        ev.append(("Поставщик", "дефицит у поставщика", None, p, t0, t1 - 1, f))
        outages.append((p, t0, t1))
    for p in np.flatnonzero(status == "Новинка"):
        t0 = max(m for m in mondays if m <= release)
        supply_frac[t0:t0 + 14, p] = 0.75
        ev.append(("Поставщик", "квота на новинку", None, p, t0, t0 + 13, 0.75))

    # закуп: случайные недо- и перезаказы по складу, и перекос между складами в одну неделю
    for t in mondays[1:]:
        for w in range(W):
            for p in np.flatnonzero(active):
                r = rng.random()
                if r < 0.05:
                    k = float(rng.uniform(0.2, 0.45))
                    buy_mult[t, w, p] = k
                    ev.append(("Закуп", "недозаказ", w, p, t, t, k))
                elif r < 0.08:
                    k = float(rng.uniform(1.8, 2.5))
                    buy_mult[t, w, p] = k
                    ev.append(("Закуп", "перезаказ", w, p, t, t, k))
        if rng.random() < 0.5:
            p = int(rng.choice(np.flatnonzero(active)))
            a, b = rng.choice(W, size=2, replace=False)
            buy_mult[t, a, p], buy_mult[t, b, p] = 2.3, 0.25
            ev.append(("Закуп", "перекос между складами: перезаказ", int(a), p, t, t, 2.3))
            ev.append(("Закуп", "перекос между складами: недозаказ", int(b), p, t, t, 0.25))

    # перекос закупа при дефиците: в первую неделю перебоя один склад заказывают с запасом и отдают ему
    # подтверждённый объём в первую очередь, остальным складам достаётся остаток
    for p, t0, t1 in outages:
        if rng.random() < 0.7:
            w = int(rng.integers(0, W))
            buy_mult[t0, :, p] = 1.0
            buy_mult[t0, w, p] = 2.5
            favor[t0, p] = w
            supply_frac[t0, p] = max(supply_frac[t0, p], 0.6)  # умеренный дефицит: поставщик даёт больше половины
            ev.append(("Закуп", "перекос закупа при дефиците", w, p, t0, t0, 2.5))

    # магазины: пропуски пополнения и перезатарка отдельных пар «магазин-товар»
    n_pairs = max(2, int(M * P * 0.03))
    for _ in range(n_pairs):
        m, p = int(rng.integers(0, M)), int(rng.integers(0, P))
        t0 = int(rng.integers(5, days - 14))
        t1 = t0 + int(rng.integers(8, 16))
        skip[t0:t1, m, p] = True
        ev.append(("Распределение", "пропуск пополнения магазина", m, p, t0, t1 - 1, None))
    for _ in range(n_pairs):
        m, p = int(rng.integers(0, M)), int(rng.integers(0, P))
        t0 = int(rng.integers(5, days - 21))
        t1 = t0 + int(rng.integers(14, 21))
        s_mult[t0:t1, m, p] = 3.0
        ev.append(("Распределение", "перезатарка магазина", m, p, t0, t1 - 1, 3.0))
    return buy_mult, supply_frac, favor, skip, s_mult, ev


def simulate(products, branches, days, rng):
    dates = pd.date_range(PERIOD_START, periods=days, freq="D")
    P = len(products)
    wh_names = branches.loc[branches.Вид_филиала == "Склад", "Филиал"].tolist()
    stores = branches[branches.Вид_филиала == "Магазин"].reset_index(drop=True)
    W, M = len(wh_names), len(stores)
    parent = stores.ПриписанК_Складу.map({w: i for i, w in enumerate(wh_names)}).to_numpy()
    status = products.Ассортиментный_Статус.to_numpy()
    new = status == "Новинка"
    release = (NEW_LINE_RELEASE - PERIOD_START).days

    base = 0.45 * np.outer(stores["size"].to_numpy(), products["pop"].to_numpy())  # спрос шт/день
    # плановый спрос, по которому магазин держит запас: новинку планируют без ажиотажа
    def plan_mult(t):
        m = np.ones(P)
        if t < release:
            m[new] = 0
        else:
            m[status == "Выводимый"], m[status == "Основной"] = 0.55, 0.85
        return m

    def demand_mult(t, d):
        m = plan_mult(t)
        if t >= release:
            m[new] = 1 + 1.5 * np.exp(-(t - release) / 10)
        return m * (1.25 if d.dayofweek >= 5 else 0.93)

    buy_mult, supply_frac, favor, skip, s_mult, events = plan_events(W, M, P, days, dates, status, parent, rng)

    # стартовые остатки: магазины наполовину-полностью, склады - на неделю, новинок нет
    plan0 = base * np.where(new, 0, 1)
    st_s = np.where(new, 0, np.round(np.ceil(plan0 * 7 + 1) * rng.uniform(0.5, 1.0, (M, P)))).astype(int)
    st_w = np.zeros((W, P), dtype=int)
    for w in range(W):
        st_w[w] = np.round(plan0[parent == w].sum(axis=0) * 8).astype(int)

    pipe_w, pipe_s = [], []            # (день прибытия, склад/магазин, товар, шт.)
    moves = []                          # (день, вид филиала, индекс, товар, операция, шт.)
    orders = []                         # заказы поставщику
    shipped_hist = np.zeros((days, W, P))  # отгрузки склада в магазины - основа прогноза закупщика
    true_stock_w, true_stock_s, true_demand = [], [], []

    for t, d in enumerate(dates):
        true_stock_w.append(st_w.copy())
        true_stock_s.append(st_s.copy())

        # утро: приходы
        rest = []
        for day, w, p, q in pipe_w:
            if day == t:
                st_w[w, p] += q
                moves.append((t, "w", w, p, "Поступление от поставщика", q))
            else:
                rest.append((day, w, p, q))
        pipe_w = rest
        rest = []
        for day, m, p, q in pipe_s:
            if day == t:
                st_s[m, p] += q
                moves.append((t, "s", m, p, "Поступление со склада", q))
            else:
                rest.append((day, m, p, q))
        pipe_s = rest

        # день: продажи
        demand = rng.poisson(base * demand_mult(t, d))
        true_demand.append(demand)
        sold = np.minimum(demand, st_s)
        st_s -= sold
        for m, p in zip(*np.nonzero(sold)):
            moves.append((t, "s", m, p, "Продажа", int(sold[m, p])))

        # вечер: заявки магазинов, склад отгружает в пределах своего остатка
        plan = base * plan_mult(t)
        if t >= release - 3:
            plan_new_ok = np.ones(P, bool)
        else:
            plan_new_ok = ~new
        s_lvl = np.ceil(plan * STORE_MIN)
        S_lvl = np.ceil(plan * STORE_MAX * s_mult[t])
        in_tr = np.zeros((M, P), dtype=int)
        for day, m, p, q in pipe_s:
            in_tr[m, p] += q
        pos = st_s + in_tr
        req = np.where((pos <= s_lvl) & plan_new_ok & ~skip[t], S_lvl - pos, 0).astype(int)
        if t >= release - 3 and t < release:  # первая завозка новинки до старта продаж
            req[:, new] = np.maximum(req[:, new], (S_lvl[:, new] - pos[:, new]).astype(int))
        for m in rng.permutation(M):          # при нехватке на складе кто первый, тот и получил
            w = parent[m]
            for p in np.flatnonzero(req[m] > 0):
                q = int(min(req[m, p], st_w[w, p]))
                if q <= 0:
                    continue
                st_w[w, p] -= q
                shipped_hist[t, w, p] += q
                moves.append((t, "w", w, p, "Отгрузка в магазин", q))
                pipe_s.append((t + LEAD_STORE, m, p, q))

        # понедельник: закуп на следующую неделю
        if d.dayofweek == ORDER_WEEKDAY:
            in_w = np.zeros((W, P), dtype=int)
            for day, w, p, q in pipe_w:
                in_w[w, p] += q
            qty = np.zeros((W, P), dtype=int)
            for w in range(W):
                kids = parent == w
                # прогноз закупщика: средние отгрузки склада за 3 недели, для новинки до истории - план
                hist = shipped_hist[max(0, t - 21):t, w].mean(axis=0) if t > 0 else np.zeros(P)
                plan_w = (base[kids] * plan_mult(max(t, release))).sum(axis=0)
                plan_w[new] *= 2.0  # на старт новинки закупщик закладывает ажиотажный спрос
                fc = np.where(new & (t < release + 14), np.maximum(hist, plan_w), hist)
                if t < 14:  # в первые недели истории отгрузок мало - закупают по плану
                    fc = np.maximum(fc, plan_w * ~new)
                # пустой склад ничего не отгружает, и прогноз по отгрузкам падает; ниже половины плана закупщик
                # его не опускает - иначе заказы прекратились бы навсегда
                fc = np.maximum(fc, 0.5 * (base[kids] * plan_mult(t)).sum(axis=0))
                need = np.maximum(0, fc * COVER_DAYS - st_w[w] - in_w[w])
                if t < release - 7:
                    need[new] = 0
                k = buy_mult[t, w]
                # перезаказ: сверх нужного докупают ещё на (k - 1) недели отгрузок - даже если склад полон
                qty[w] = np.round(np.where(k > 1, need + (k - 1) * fc * 7, need * k)).astype(int)
            # поставщик подтверждает долю от суммы заказов; обычно она делится между складами пропорционально,
            # при перекосе выбранный склад получает свой заказ целиком, остальные - остаток
            conf = np.floor(qty * supply_frac[t]).astype(int)
            for p in np.flatnonzero(favor[t] >= 0):
                w, total = favor[t, p], int(np.floor(qty[:, p].sum() * supply_frac[t, p]))
                conf[w, p] = min(qty[w, p], total)
                rest, others = total - conf[w, p], [x for x in range(W) if x != w]
                share = qty[others, p] / max(1, qty[others, p].sum())
                conf[others, p] = np.floor(rest * share).astype(int)
            for w, p in zip(*np.nonzero(qty > 0)):
                orders.append((t, w, p, int(qty[w, p]), int(conf[w, p])))
                if conf[w, p] > 0:
                    pipe_w.append((t + LEAD_SUPPLIER, w, p, int(conf[w, p])))

    in_w = np.zeros((W, P), dtype=int)
    for day, w, p, q in pipe_w:
        in_w[w, p] += q
    in_s = np.zeros((M, P), dtype=int)
    for day, m, p, q in pipe_s:
        in_s[m, p] += q
    return dict(dates=dates, wh_names=wh_names, stores=stores, parent=parent, moves=moves, orders=orders,
                st_w=st_w, st_s=st_s, in_w=in_w, in_s=in_s, events=events,
                true_w=np.stack(true_stock_w), true_s=np.stack(true_stock_s), true_d=np.stack(true_demand))


def generate(n_stores, n_products, days, seed):
    rng = np.random.default_rng(seed)
    products = build_products(None, rng)
    if n_products and len(products) > n_products:
        # отладочная выборка: самые популярные модели, но обязательно с несколькими выводимыми
        old = products.Ассортиментный_Статус == "Выводимый"
        k = max(2, n_products // 6)
        keep = pd.concat([products[old].nlargest(k, "pop"), products[~old].nlargest(n_products - k, "pop")])
        products = products.loc[sorted(keep.index)].reset_index(drop=True)
    branches = build_branches(n_stores, rng)
    r = simulate(products, branches, days, rng)
    dates, pref = r["dates"], products.ТоварСсылка.to_numpy()
    bref = branches.set_index("Филиал").ФилиалСсылка
    wref = bref[r["wh_names"]].to_numpy()
    sref = bref[r["stores"].Филиал].to_numpy()
    price = products.price.to_numpy()
    cost = np.round(price * 0.82, -1)

    def ref(kind, i):
        return wref[i] if kind == "w" else sref[i]

    mv = pd.DataFrame(r["moves"], columns=["t", "kind", "i", "p", "Операция", "Количество"])
    mv["Дата"] = dates[mv.t]
    mv["Филиал"] = [ref(k, i) for k, i in zip(mv.kind, mv.i)]
    mv["Номенклатура"] = pref[mv.p]
    mv["ВидДвижения"] = np.where(mv.Операция.str.startswith("Поступление"), 0, 1)  # 0 - приход, 1 - расход
    mv["Сумма"] = (mv.Количество * cost[mv.p]).round(2)
    journal = mv[["Дата", "Филиал", "Номенклатура", "ВидДвижения", "Операция", "Количество", "Сумма"]]

    o = pd.DataFrame(r["orders"], columns=["t", "w", "p", "Заказано", "Подтверждено"])
    o["ДатаЗаказа"] = dates[o.t]
    o["ДатаПоступления"] = o.ДатаЗаказа + pd.Timedelta(days=LEAD_SUPPLIER)
    o["Склад"], o["Номенклатура"] = wref[o.w], pref[o.p]
    o["Поставщик"] = "Дистрибьютор Apple"
    supplier_orders = o[["ДатаЗаказа", "Поставщик", "Склад", "Номенклатура", "Заказано", "Подтверждено",
                         "ДатаПоступления"]]

    rows = []
    for kind, st, tr, refs in (("w", r["st_w"], r["in_w"], wref), ("s", r["st_s"], r["in_s"], sref)):
        ii, pp = np.nonzero((st > 0) | (tr > 0))
        rows.append(pd.DataFrame({"Филиал": refs[ii], "Номенклатура": pref[pp], "Остаток": st[ii, pp],
                                  "Транзит": tr[ii, pp]}))
    stock_now = pd.concat(rows, ignore_index=True)
    stock_now.insert(0, "ДатаОстатка", dates[-1])

    prices = pd.DataFrame({"Номенклатура": pref, "ЦенаПродажи": price.astype(float), "Себестоимость": cost})

    # истина: остатки на начало дня (склады и магазины) и спрос покупателей
    D = len(dates)
    tw, ts, td = r["true_w"], r["true_s"], r["true_d"]
    t_, i_, p_ = np.nonzero(tw >= 0)
    truth_w = pd.DataFrame({"Дата": dates[t_], "Филиал": wref[i_], "Номенклатура": pref[p_],
                            "ОстатокНаНачалоДня": tw[t_, i_, p_], "СпросИстинный": 0})
    t_, i_, p_ = np.nonzero((ts > 0) | (td > 0))
    truth_s = pd.DataFrame({"Дата": dates[t_], "Филиал": sref[i_], "Номенклатура": pref[p_],
                            "ОстатокНаНачалоДня": ts[t_, i_, p_], "СпросИстинный": td[t_, i_, p_]})
    truth_daily = pd.concat([truth_w, truth_s], ignore_index=True)

    ev = pd.DataFrame(r["events"], columns=["Этап", "Проблема", "i", "p", "t0", "t1", "Параметр"])
    ev["Филиал"] = [None if i is None or pd.isna(i) else (wref[int(i)] if e == "Закуп" else sref[int(i)])
                    for e, i in zip(ev.Этап, ev.i)]
    ev["Номенклатура"] = pref[ev.p]
    ev["ДатаС"], ev["ДатаПо"] = dates[ev.t0], dates[np.minimum(ev.t1, D - 1)]
    truth_events = ev[["Этап", "Проблема", "Филиал", "Номенклатура", "ДатаС", "ДатаПо", "Параметр"]]

    prod_out = products[["ТоварСсылка", "Код", "Товар", "model", "Ассортиментный_Статус"]].rename(
        columns={"model": "Модель"})
    branch_out = branches[["ФилиалСсылка", "Филиал", "Вид_филиала", "ПриписанК_Складу"]]
    return {"products": prod_out, "branches": branch_out, "prices": prices, "supplier_orders": supplier_orders,
            "journal": journal, "stock_now": stock_now, "truth_daily": truth_daily, "truth_events": truth_events}
