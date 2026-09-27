"""Анализ топ-500 Tesera и топ-500 BGG -> цифры для страницы исследования.

Запуск: python analysis.py [--out путь.json]   (после collect.py и make_wide.py)
Результат: results/tesera-bgg.json (данные для страницы на сайте) + сводка в консоль.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from csv_format import read_csv

MIN_VOTES_TESERA = 20
MIN_VOTES_BGG = 100
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/tesera-bgg.json")
OUT = Path(ap.parse_args().out)

w = read_csv("data/games_wide_top500.csv")
w["diff"] = w.tesera_avg_rating - w.bgg_avg_rating
w["in_t"] = w.rank_in_tesera_top.notna()
w["in_b"] = w.rank_in_bgg_top.notna()

f = w[(w.tesera_num_votes >= MIN_VOTES_TESERA) & (w.bgg_num_votes >= MIN_VOTES_BGG)].copy()
base = f[~f.is_addition]


def r2(x):
    return None if pd.isna(x) else round(float(x), 2) + 0.0  # +0.0 убирает «-0.0»


def sample_stats(s):
    t = stats.ttest_1samp(s["diff"], 0)
    return {"n": int(len(s)), "mean": r2(s["diff"].mean()), "median": r2(s["diff"].median()),
            "share_tesera_higher": r2((s["diff"] > 0).mean()), "p": float(f"{t.pvalue:.2g}")}


# 1. Эффект отбора
samples = {"tesera_top": sample_stats(f[f.in_t]), "bgg_top": sample_stats(f[f.in_b]), "union": sample_stats(f)}
rho = stats.spearmanr(f.tesera_avg_rating, f.bgg_avg_rating)
group = np.select([f.in_t & f.in_b, f.in_t], ["both", "tesera"], "bgg")
scatter = [[r2(b), r2(t), title, g, int(y) if pd.notna(y) else None]
           for b, t, title, g, y in zip(f.bgg_avg_rating, f.tesera_avg_rating, f.title, group, f.tesera_year_published)]

# 2. Год выпуска (только базовые игры)
bins = [0, 1999, 2009, 2014, 2019, 2022, 2100]
labels = ["до 2000", "2000-2009", "2010-2014", "2015-2019", "2020-2022", "2023+"]
per = base.assign(period=pd.cut(base.tesera_year_published, bins, labels=labels))
by_period = [{"period": p, "mean": r2(g["diff"].mean()), "n": int(len(g))}
             for p, g in per.groupby("period", observed=True)]
year_rho = {k: stats.spearmanr(s.tesera_year_published, s["diff"], nan_policy="omit")
            for k, s in [("tesera_top", f[f.in_t]), ("bgg_top", f[f.in_b])]}

# 3. Жанры (базовые игры, жанров с n >= 10)
ge = base.assign(genre=base.tesera_genre.fillna("").str.split(", ")).explode("genre")
ge = ge[ge.genre != ""]
by_genre = (ge.groupby("genre")["diff"].agg(["mean", "count"]).query("count >= 10").sort_values("mean"))
by_genre = [{"genre": g, "mean": r2(r["mean"]), "n": int(r["count"])} for g, r in by_genre.iterrows()]

# 4. Дополнения vs базовые игры
additions = {}
for k, s in [("tesera_top", f[f.in_t]), ("bgg_top", f[f.in_b])]:
    b, a = s[~s.is_addition]["diff"], s[s.is_addition]["diff"]
    additions[k] = {"base": r2(b.mean()), "additions": r2(a.mean()), "n_base": int(len(b)), "n_add": int(len(a)),
                    "p": float(f"{stats.mannwhitneyu(b, a).pvalue:.2g}")}

# 5. Суб-рейтинги Tesera: связь с итоговой оценкой
SUB = {"tesera_gameplay": "Геймплей", "tesera_depth": "Глубина",
       "tesera_originality": "Оригинальность", "tesera_implementation": "Реализация"}
sr = f.dropna(subset=list(SUB))
subratings = {"n": int(len(sr)), "items": [
    {"name": name, "mean": r2(sr[col].mean()), "rho": r2(stats.spearmanr(sr[col], sr.tesera_avg_rating).correlation)}
    for col, name in SUB.items()]}

# 6. Издатель: Hobby World
pub = w.assign(p=w.tesera_publisher.fillna("").str.split(", ")).explode("p")
pub = pub[pub.p != ""]
top_publishers = [{"publisher": p, "games": int(n)} for p, n in pub.p.value_counts().head(6).items()]
w["buy_per_sell"] = w.tesera_buying_now / w.tesera_selling_now.clip(lower=1)
hw = w[w.tesera_publisher.fillna("").str.contains("Hobby World") & (w.tesera_owners >= 300)]
hw_demand = [{"title": r.title, "addition": bool(r.is_addition), "owners": int(r.tesera_owners),
              "buying": int(r.tesera_buying_now), "selling": int(r.tesera_selling_now), "ratio": r2(r.buy_per_sell)}
             for r in hw.sort_values("buy_per_sell", ascending=False).head(8).itertuples()]

snapshot = read_csv("data/tesera_top500_long.csv").snapshot_date.iloc[0]
result = {
    "meta": {"snapshot": snapshot, "games_total": int(len(w)), "games_compared": int(len(f)),
             "overlap": int((w.in_t & w.in_b).sum()), "min_votes_tesera": MIN_VOTES_TESERA,
             "min_votes_bgg": MIN_VOTES_BGG, "additions_total": int(w.is_addition.sum())},
    "selection": {"samples": samples, "spearman": r2(rho.correlation), "scatter": scatter},
    "year": {"by_period": by_period, "rho": {k: r2(v.correlation) for k, v in year_rho.items()},
             "p": {k: float(f"{v.pvalue:.2g}") for k, v in year_rho.items()}},
    "genre": by_genre,
    "additions": additions,
    "subratings": subratings,
    "publisher": {"top": top_publishers, "hw_demand": hw_demand,
                  "median_ratio_all": r2(w.buy_per_sell.median())},
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "selection"}, ensure_ascii=False, indent=1))
print("selection:", json.dumps(result["selection"]["samples"], ensure_ascii=False), "rho", result["selection"]["spearman"])
print("->", OUT)
