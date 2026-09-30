"""Выгрузка готовых данных для страниц проектов на сайте-портфолио.

Берёт результаты ноутбуков из схемы `results` локальной базы (и немного справочной информации из исходных
схем) и сохраняет по одному компактному JSON на проект. База открывается только на чтение.

Порядок обновления:
    python generate.py --scale full
    выполнить ноутбуки: кластеризация -> скоринг -> остатки
    python export_site_data.py
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "synthetic_projects.duckdb"
DEFAULT_OUT = Path(r"C:\Users\User\Desktop\clode folder\projects_code\Личный сайт\site\src\data")


def r(x, d=4):
    """Округление для компактного JSON (numpy-типы -> обычные числа)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), d)


def example_store(seg):
    """Пример для схемы сборки сегмента: первая эконом-точка, сегмент которой не менялся при слиянии."""
    cand = seg[(seg.price_level == "эконом") & (seg.segment == seg.segment_raw)].sort_values("TradePointId")
    x = (cand if len(cand) else seg.sort_values("TradePointId")).iloc[0]
    return {"code": x.TradePointCode, "size_cluster": x.size_money_cluster, "size_level": x.size_level,
            "price_cluster": x.price_cluster, "price_level": x.price_level, "assort_cluster": x.assortment_cluster,
            "profile": x.assortment_profile, "segment": x.segment}


def export_clustering(con):
    seg = con.sql("""
        SELECT s.*, a.archetype
        FROM results.store_segments s JOIN pharmacy.store_archetypes a USING (TradePointId)
    """).df()
    # кластеризация считается по первой половине периода чеков (вторая - отложенная проверка в скоринге)
    full = con.sql("SELECT MIN(Date) AS f, MAX(Date) AS t FROM pharmacy.cheques").df().iloc[0]
    train_days = ((pd.Timestamp(full.t) - pd.Timestamp(full.f)).days + 1) // 2
    period = pd.Series({"f": pd.Timestamp(full.f), "t": pd.Timestamp(full.f) + pd.Timedelta(days=train_days - 1)})
    funnel = con.execute("""
        SELECT COUNT(*) AS all_lines,
               COUNT(*) FILTER (WHERE c.SaleType NOT IN ('Опт', 'Маркетплейс')
                                  AND c.ShelfLifeMonths >= 6
                                  AND p.product_class <> 'Нетоварные позиции') AS kept_lines
        FROM pharmacy.cheques c JOIN pharmacy.products p USING (apCode)
        WHERE c.Date BETWEEN ? AND ?
    """, [period.f, period.t]).df().iloc[0]
    n_stores_all = con.sql("SELECT COUNT(*) FROM pharmacy.stores").fetchone()[0]
    n_products = con.sql("SELECT COUNT(*) FROM pharmacy.products WHERE product_class <> 'Нетоварные позиции'").fetchone()[0]
    n_drugs = con.sql("SELECT COUNT(*) FROM pharmacy.products WHERE source LIKE 'ГРЛС%'").fetchone()[0]

    idx = con.sql("SELECT * FROM results.clustering_assort_index").df()
    clusters = sorted(idx.cluster.unique())
    cats = sorted(idx.category.unique())
    pivot = idx.pivot(index="cluster", columns="category", values="index").loc[clusters, cats]
    info = idx.drop_duplicates("cluster").set_index("cluster")

    price = seg.groupby("price_cluster").agg(n=("TradePointId", "size"), low=("share_low", "mean"),
                                             medium=("share_medium", "mean"), high=("share_high", "mean"),
                                             level=("price_level", "first"))
    summary = seg.groupby("segment").agg(
        n=("TradePointId", "size"), square=("trade_square", "median"), revenue=("revenue", "median"),
        cheque=("mean_cheque", "median"), high=("share_high", "mean"), low=("share_low", "mean"),
    ).sort_values("revenue", ascending=False)
    ct = pd.crosstab(seg.segment, seg.archetype).loc[summary.index]
    quality = con.sql("SELECT * FROM results.clustering_quality").df().set_index("metric").ari
    kdiag = con.sql("SELECT * FROM results.clustering_k_selection ORDER BY clustering, k").df()
    k_selection = {
        name: {"k": g.k.astype(int).tolist(), "cost": [r(x, 3) for x in g.merge_cost],
               "silhouette": [r(x, 3) for x in g.silhouette], "chosen": int(g.loc[g.chosen, "k"].iloc[0]),
               "k_min": int(g.k_min.iloc[0]), "k_max": int(g.k_max.iloc[0])}
        for name, g in kdiag.groupby("clustering")
    }

    return {
        "meta": {
            "period_from": str(period.f.date()), "period_to": str(period.t.date()),
            "stores_all": int(n_stores_all), "stores_used": int(len(seg)),
            "lines_all": int(funnel.all_lines), "lines_kept": int(funnel.kept_lines),
            "products": int(n_products), "drugs_from_registry": int(n_drugs),
            "k_size": int(seg.size_money_cluster.nunique()), "k_square": int(seg.size_square_cluster.nunique()),
            "k_price": int(seg.price_cluster.nunique()), "k_assort": int(len(clusters)),
            "segments": int(len(summary)),
        },
        "quality": {"size": r(quality["ID_size_rev_cheque"], 2), "price": r(quality["ID_prices"], 2),
                    "assort": r(quality["ID_groups"], 2), "segment": r(quality["Сегмент"], 2)},
        # точка: выручка, чеки, площадь, кластер по деньгам, ценовой кластер, доля low, доля high, сегмент
        "stores": [[r(x.revenue, 0), int(x.cheque_count), r(x.trade_square, 1), x.size_money_cluster, x.price_cluster,
                    r(x.share_low, 3), r(x.share_high, 3), x.segment] for x in seg.itertuples()],
        "price_clusters": [{"cluster": c, "n": int(x.n), "low": r(x.low, 3), "medium": r(x.medium, 3),
                            "high": r(x.high, 3), "level": x.level} for c, x in price.iterrows()],
        "assort": {"clusters": clusters, "categories": cats,
                   "profiles": [info.loc[c, "profile"] for c in clusters],
                   "n": [int(info.loc[c, "n_stores"]) for c in clusters],
                   "index": [[r(v, 2) for v in row] for row in pivot.to_numpy()]},
        "k_selection": k_selection,
        "size_clusters": [{"cluster": c, "n": int(len(g)), "level": g.size_level.iloc[0], "revenue": r(g.revenue.median(), 0)}
                          for c, g in seg.sort_values("size_money_cluster").groupby("size_money_cluster")],
        "example": example_store(seg),
        "segments_raw": int(seg.segment_raw.nunique()),
        "merges": [{"from": x.from_segment, "n": int(x.n_stores), "to": x.to_segment}
                   for x in con.sql("SELECT * FROM results.segment_merges").df().itertuples()],
        "segments": [{"segment": s, "n": int(x.n), "square": r(x.square, 0), "revenue": r(x.revenue, 0),
                      "cheque": r(x.cheque, 0), "high": r(x.high, 3), "low": r(x.low, 3)}
                     for s, x in summary.iterrows()],
        "archetypes": {"segments": list(ct.index), "archetypes": list(ct.columns),
                       "counts": ct.to_numpy().astype(int).tolist()},
    }


# Примеры для цепочки «категория → подкатегория → юнит → товар» на странице матрицы: (МНН или подкатегория, форма
# юнита или вид товара). Юниты берутся из справочника results.product_units, который пишет ноутбук скоринга.
HIERARCHY_EXAMPLES = [("Ибупрофен", "таблетки"), ("Ибупрофен", "сироп, суспензия, раствор внутрь"), ("Амлодипин", "таблетки"),
                      ("Подгузники и пеленки", "подгузники"), ("Витаминно-минеральные комплексы", "саше")]
SUBCATEGORY_SHORT = {"Противовоспалительные и противоревматические": "Противовоспалительные"}


def hierarchy_examples(con):
    p = con.sql("""
        SELECT p.apCode, p.category, p.subcategory, p.mnn, p.form_name, p.name, p.trade_name, p.dosage, p.pack_qty,
               p.is_own_brand, u.unit_form, u.Unit, COALESCE(s.revenue, 0) AS revenue
        FROM pharmacy.products p
        JOIN results.product_units u USING (apCode)
        LEFT JOIN (SELECT apCode, SUM(Sum_Fact) AS revenue FROM pharmacy.cheques GROUP BY 1) s USING (apCode)
    """).df()
    size = p.Unit.value_counts()
    rows = []
    for key, form in HIERARCHY_EXAMPLES:
        drug = p.mnn == key
        g = p[(drug & (p.unit_form == form)) | (~drug & (p.subcategory == key) & (p.form_name == form))]
        if g.empty:
            continue
        # пример - самый продаваемый товар обычного бренда (собственная марка сети выглядела бы как заглушка)
        brands = g[~g.is_own_brand]
        top = (brands if len(brands) else g).sort_values("revenue", ascending=False).iloc[0]
        if pd.notna(top.mnn):
            unit = top.Unit
            product = (f"{top.trade_name} {top.dosage}" + (f" №{int(top.pack_qty)}" if top.pack_qty > 1 else "")).replace("  ", " ")
        else:
            unit = top.Unit.replace(f"{key}, {form}", form[0].upper() + form[1:]) if form.lower() in key.lower() else top.Unit
            product = top["name"].split(" (")[0] + (f" №{int(top.pack_qty)}" if top.pack_qty > 1 else "")
        rows.append({"category": top.category, "subcategory": SUBCATEGORY_SHORT.get(top.subcategory, top.subcategory),
                     "unit": unit, "n": int(size[top.Unit]), "product": product})
    return rows


def export_scoring(con):
    v = con.sql("SELECT * FROM results.scoring_variants").df()
    units = con.sql("SELECT * FROM results.scoring_units").df()
    width = con.sql("SELECT * FROM results.scoring_width").df().sort_values("n_stores", ascending=False)
    biggest = width.iloc[0].segment
    u = units[units.segment == biggest].sort_values("cum_share").reset_index(drop=True)

    exp = con.sql("SELECT * FROM results.scoring_experiment").df()
    prm = con.sql("SELECT * FROM results.scoring_params").df().set_index("param").value
    fr = con.sql("SELECT * FROM results.scoring_frontier ORDER BY method, lam").df()
    wk = con.sql("SELECT * FROM results.scoring_weekly ORDER BY method, week").df()

    return {
        "meta": {"segments": int(units.segment.nunique()), "units": int(units.Unit.nunique()),
                 "matrix_fact": int(round(v.loc[v.variant == "факт", "positions"].iloc[0])),
                 "pareto_segment": biggest, "pareto_units": int(len(u))},
        "params": {k: r(prm[k], 4) for k in prm.index},
        "variants": [{"variant": x.variant, "stock": r(x.stock_rub, 0), "revenue": r(x.revenue, 0),
                      "profit": r(x.profit, 0), "positions": int(round(x.positions)),
                      "d_stock": r(x.diff_stock_rub, 0), "d_revenue": r(x.diff_revenue, 0),
                      "d_profit": r(x.diff_profit, 0), "d_positions": int(round(x.diff_positions)),
                      "selected": bool(x.selected)} for x in v.itertuples()],
        "hierarchy": hierarchy_examples(con),
        # кривая способа по сетке строгости: [Δ остатки, Δ прибыль на отложенных неделях, Δ прибыль на обучении, λ]
        "frontier": {m: [[r(x.d_stock, 0), r(x.d_profit_test, 0), r(x.d_profit_train, 0), r(x.lam, 3)]
                          for x in g.itertuples()] for m, g in fr.groupby("method")},
        # две точки пересечения кривой с нулевыми осями, посчитанные точно
        "experiment": [{"method": x.method, "groups": int(x.groups),
                        "d_stock_zero_profit": r(x.d_stock_zero_profit, 0), "positions_zero_profit": int(round(x.positions_zero_profit)),
                        "lam_zero_profit": r(x.lam_zero_profit, 3),
                        "d_profit_zero_stock": r(x.d_profit_zero_stock, 0), "lam_zero_stock": r(x.lam_zero_stock, 3)}
                       for x in exp.itertuples()],
        # по неделям: матрица каждого способа выбрана по неделям обучения; [неделя, прибыль, остатки по нормативу]
        "weekly": {m: {"lam": None if g.lam.isna().all() else r(g.lam.iloc[0], 3),
                       "rows": [[int(x.week), r(x.profit, 0), r(x.stock_rub, 0)] for x in g.itertuples()]}
                   for m, g in wk.groupby("method")},
        "pareto": [[r((i + 1) / len(u), 4), r(x.cum_share, 4)] for i, x in enumerate(u.itertuples())],
        "width": [{"segment": x.segment, "n_stores": int(x.n_stores), "fact": int(round(x.fact)), "new": int(x.new)}
                  for x in width.itertuples()],
    }


def export_stock(con):
    s = con.sql("SELECT * FROM results.stock_summary").df().set_index("metric").value
    av = con.sql("SELECT * FROM results.stock_availability").df()
    order = av.groupby("branch").share.mean().sort_values().index
    dates = sorted(av.date.unique())
    grid = av.pivot(index="branch", columns="date", values="share").loc[order, dates]
    lines = con.sql("SELECT * FROM results.stock_lines ORDER BY date").df()
    sku = con.sql("SELECT * FROM results.stock_sku_deficit ORDER BY deficit_share DESC").df()
    lost = con.sql("SELECT * FROM results.stock_lost_by_branch ORDER BY lost_est DESC").df()
    n_moves = con.sql("SELECT COUNT(*) FROM electronics.schet_41").fetchone()[0]

    return {
        "meta": {"period_from": s["period_from"], "period_to": s["period_to"], "rows": int(float(s["rows"])),
                 "match_share": r(float(s["match_share"]), 4), "max_abs_diff": int(float(s["max_abs_diff"])),
                 "availability": r(float(s["availability"]), 4), "sales": r(float(s["sales_rub"]), 0),
                 "lost_est": r(float(s["lost_est_rub"]), 0), "lost_true": r(float(s["lost_true_rub"]), 0),
                 "branches": int(float(s["branches"])), "skus": int(float(s["skus"])), "moves": int(n_moves),
                 "lost_rank_corr": r(lost[["lost_est", "lost_true"]].corr("spearman").iloc[0, 1], 3)},
        "availability": {"branches": list(order), "dates": [str(pd.Timestamp(d).date()) for d in dates],
                         "share": [[r(v, 3) for v in row] for row in grid.to_numpy()]},
        "lines": {name: {"dates": [str(pd.Timestamp(d).date()) for d in g.date], "stock": g.stock.astype(int).tolist(),
                         "sales": g.sales.astype(int).tolist()} for name, g in lines.groupby("line")},
        "sku": [{"sku": x.sku.replace("Смартфон Apple ", ""), "status": x.status, "deficit": r(x.deficit_share, 3)}
                for x in sku.head(12).itertuples()],
        "lost": [{"branch": x.branch, "est": r(x.lost_est, 0), "true": r(x.lost_true, 0), "sales": r(x.sales, 0)}
                 for x in lost.itertuples()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.db, read_only=True)
    payloads = {"store-clustering.json": export_clustering(con), "assortment-matrix.json": export_scoring(con),
                "stock-history.json": export_stock(con)}
    con.close()
    for name, payload in payloads.items():
        path = out / name
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"-> {path} ({path.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
