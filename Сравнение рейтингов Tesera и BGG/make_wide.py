"""Широкая таблица для просмотра: одна строка на игру из обеих выборок (топ Tesera + топ BGG).

Запуск:
  python make_wide.py                 # по топ-500
  python make_wide.py --n 50          # по топ-50
"""
import argparse
from pathlib import Path

import numpy as np
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

# основная игра для дополнений: связь «исходная игра» из блока «Связанные игры»
# (в новых сборах - block = relations в узкой таблице, для среза 25.09 - досбор data/relations_long.csv).
# Флаг is_addition из API Tesera ненадёжен (им помечены и многие базовые игры с дополнениями),
# поэтому base_game / addition определяются по связи: дополнение = у игры есть «исходная игра».
BASE = "исходная игра"
rel = d.loc[(d.block == "relations") & (d.dim_l1 == BASE), ["alias", "value_text", "seq"]].rename(
    columns={"value_text": "related_alias"})
rel_titles = {}
if Path("data/relations_long.csv").exists():
    extra = read_csv("data/relations_long.csv")
    extra = extra[extra.kind == BASE]
    rel = pd.concat([rel, extra[["alias", "related_alias", "seq"]]])
    rel_titles = dict(zip(extra.related_alias, extra.related_title))
rel = rel.drop_duplicates(["alias", "related_alias"])
# если «исходных игр» несколько (сборники, дополнения к дополнениям), берём известную базовую игру:
# она есть в срезе и у неё самой нет «исходной игры»; иначе - первую по порядку на странице
has_base = set(rel.alias)
rel["known_base"] = rel.related_alias.isin(set(out.alias) - has_base)
base_alias = (rel.sort_values(["known_base", "seq"], ascending=[False, True])
              .drop_duplicates("alias").set_index("alias").related_alias)
titles = {**rel_titles, **dict(zip(out.alias, out.title))}

is_add = out.alias.isin(has_base)
out["base_game_alias"] = np.where(is_add, out.alias.map(base_alias), out.alias)
out["base_game"] = out.base_game_alias.map(titles)
out["addition"] = out.title.where(is_add)

first = ["rank_in_tesera_top", "rank_in_bgg_top", "title", "base_game", "addition", "is_addition", "tesera_year_published",
         "tesera_avg_rating", "bgg_avg_rating", "diff_tesera_minus_bgg", "tesera_num_votes", "bgg_num_votes"]
out = out[first + [c for c in out.columns if c not in first]]
out = out.sort_values(["rank_in_tesera_top", "rank_in_bgg_top"])
dest = f"data/games_wide_top{args.n}.csv"
write_csv(out, dest)
print(out.shape, "->", dest)
