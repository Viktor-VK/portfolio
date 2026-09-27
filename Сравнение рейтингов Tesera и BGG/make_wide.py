"""Широкая таблица для просмотра: одна строка на игру из обеих выборок (топ Tesera + топ BGG).

Запуск:
  python make_wide.py                 # по топ-500
  python make_wide.py --n 50          # по топ-50
"""
import argparse

import pandas as pd

from csv_format import read_csv, write_csv

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
args = ap.parse_args()

frames = []
for path, sample in [(f"data/tesera_top{args.n}_long.csv", "tesera_top"), (f"data/bgg_top{args.n}_long.csv", "bgg_top")]:
    d = read_csv(path)
    d["sample"] = sample
    frames.append(d)
d = pd.concat(frames)

keys = ["alias", "title", "bgg_id", "is_addition"]
ranks = (d.drop_duplicates(["alias", "sample"])
         .pivot(index="alias", columns="sample", values="list_rank").add_prefix("rank_in_"))

num = d[d.value_num.notna()
        & d.block.isin(["rating", "subrating", "audience", "specs"])
        & ~d.metric.isin(["component", "box_length", "box_width", "box_height"])]
num = num.drop_duplicates(["alias", "platform", "block", "metric"])  # owners: API раньше HTML
wide = num.pivot_table(index="alias", columns=["platform", "metric"], values="value_num", aggfunc="first")
wide.columns = [f"{p}_{m}" for p, m in wide.columns]

text = (d[d.metric.isin(["genre", "theme", "tag", "publisher"])]
        .drop_duplicates(["alias", "metric", "value_text"])
        .groupby(["alias", "metric"]).value_text.agg(", ".join).unstack().add_prefix("tesera_"))

out = d.drop_duplicates("alias")[keys].set_index("alias").join(ranks).join(wide).join(text).reset_index()

# 0 в рейтингах и числе оценок = «нет данных», а не ноль
for c in [c for c in out.columns if "rating" in c or c.endswith("num_votes")]:
    out[c] = out[c].mask(out[c] == 0)
out["diff_tesera_minus_bgg"] = (out.tesera_avg_rating - out.bgg_avg_rating).round(2)

first = ["rank_in_tesera_top", "rank_in_bgg_top", "title", "is_addition", "tesera_year_published",
         "tesera_avg_rating", "bgg_avg_rating", "diff_tesera_minus_bgg", "tesera_num_votes", "bgg_num_votes"]
out = out[first + [c for c in out.columns if c not in first]]
out = out.sort_values(["rank_in_tesera_top", "rank_in_bgg_top"])
dest = f"data/games_wide_top{args.n}.csv"
write_csv(out, dest)
print(out.shape, "->", dest)
