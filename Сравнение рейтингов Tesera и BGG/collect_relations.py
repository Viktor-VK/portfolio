"""Досбор связей между играми (блок «Связанные игры» на странице Tesera) для игр текущего среза.

Нужен, чтобы у дополнения знать основную игру (тип связи «исходная игра»), не пересобирая
основные таблицы. В новых сборах collect.py пишет эти же связи в узкую таблицу (block = relations).

Запуск: python collect_relations.py
Результат: data/relations_long.csv - одна строка = одна связь.
"""
import datetime as dt

import pandas as pd
import requests
from bs4 import BeautifulSoup

from collect import API, SITE, get, parse_relations
from csv_format import read_csv, write_csv

games = read_csv("data/games_wide_top500.csv")[["alias", "title"]]
session = requests.Session()
snapshot = dt.date.today().isoformat()

rows = []
for n, (alias, title) in enumerate(games.itertuples(index=False), start=1):
    url = f"{SITE}/game/{alias}/"
    try:
        soup = BeautifulSoup(get(session, url).text, "lxml")
        for seq, (kind, rel_alias) in enumerate(parse_relations(soup)):
            rows.append({"snapshot_date": snapshot, "alias": alias, "title": title, "kind": kind, "seq": seq,
                         "related_alias": rel_alias, "source_url": url})
    except requests.RequestException as e:
        rows.append({"snapshot_date": snapshot, "alias": alias, "title": title, "kind": "collect_error", "seq": 0,
                     "related_alias": str(e), "source_url": url})
    if n % 50 == 0:
        print(f"{n}/{len(games)}", flush=True)

rel = pd.DataFrame(rows)

# полные названия связанных игр: из среза, а для остальных - из API Tesera
known = dict(zip(games.alias, games.title))
for a in rel.loc[rel.kind == "исходная игра", "related_alias"].unique():
    if a not in known:
        try:
            known[a] = get(session, f"{API}/games/{a}").json()["game"]["title"]
        except requests.RequestException:
            pass
rel["related_title"] = rel.related_alias.map(known)

write_csv(rel[["snapshot_date", "alias", "title", "kind", "seq", "related_alias", "related_title", "source_url"]],
          "data/relations_long.csv")
print(f"Итого {len(rel)} связей -> data/relations_long.csv")
print(rel.kind.value_counts().to_string())
