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
    period = con.sql("SELECT MIN(Date) AS f, MAX(Date) AS t FROM pharmacy.cheques").df().iloc[0]
    funnel = con.sql("""
        SELECT COUNT(*) AS all_lines,
               COUNT(*) FILTER (WHERE c.SaleType NOT IN ('Опт', 'Маркетплейс')
                                  AND c.ShelfLifeMonths >= 6
                                  AND p.product_class <> 'Нетоварные позиции') AS kept_lines
        FROM pharmacy.cheques c JOIN pharmacy.products p USING (apCode)
    """).df().iloc[0]
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


def export_scoring(con):
    v = con.sql("SELECT * FROM results.scoring_variants").df()
    units = con.sql("SELECT * FROM results.scoring_units").df()
    width = con.sql("SELECT * FROM results.scoring_width").df().sort_values("n_stores", ascending=False)
    biggest = width.iloc[0].segment
    u = units[units.segment == biggest].sort_values("cum_share").reset_index(drop=True)
    classes = units.groupby("segment").Status.value_counts().unstack().fillna(0)

    return {
        "meta": {"segments": int(units.segment.nunique()), "units": int(units.Unit.nunique()),
                 "matrix_fact": int(v.loc[v.variant == "факт", "positions"].iloc[0]),
                 "pareto_segment": biggest, "pareto_units": int(len(u))},
        "variants": [{"variant": x.variant, "stock": r(x.stock_rub, 0), "revenue": r(x.revenue, 0),
                      "profit": r(x.profit, 0), "positions": int(x.positions),
                      "d_stock": r(x.diff_stock_rub, 0), "d_revenue": r(x.diff_revenue, 0),
                      "d_profit": r(x.diff_profit, 0), "d_positions": int(x.diff_positions),
                      "selected": bool(x.selected)} for x in v.itertuples()],
        "pareto": [[r((i + 1) / len(u), 4), r(x.cum_share, 4), x.Status] for i, x in enumerate(u.itertuples())],
        "classes": [{"segment": s, "A": int(x.get("A", 0)), "B": int(x.get("B", 0)), "C": int(x.get("C", 0))}
                    for s, x in classes.loc[width.segment].iterrows()],
        "width": [{"segment": x.segment, "n_stores": int(x.n_stores), "fact": int(x.fact), "new": int(x.new)}
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
