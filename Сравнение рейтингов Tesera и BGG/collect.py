"""Сбор данных Tesera (+ BGG-поля, которые хранит Tesera) в «узкую» таблицу.

Источники:
  - api.tesera.ru/games?sort=-ratingn10   — список «Топ Тесеры» (тот же, что на сайте)
  - api.tesera.ru/games/<alias>           — карточка игры (счётчики владельцев, барахолки, контента)
  - tesera.ru/game/<alias>/               — HTML: суб-рейтинги, жанры, темы, теги, авторы, награды...

Одна строка = одно значение одного измерения одной игры. Ничего не агрегируем и не схлопываем.

Запуск:
  python collect.py --limit 500 --out data/tesera_top500_long.csv                        # топ-500 Tesera
  python collect.py --limit 500 --sort=-ratinggeekbgg --out data/bgg_top500_long.csv    # топ-500 BGG
  python collect.py --limit 2 --out data/sample.csv                                     # пробный прогон
"""
import argparse
import datetime as dt
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from csv_format import write_csv

API = "https://api.tesera.ru"
SITE = "https://tesera.ru"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; portfolio project)"}
DELAY = 1.0  # пауза между запросами, сек

COLUMNS = [
    "snapshot_date", "list_sort", "list_rank",
    "tesera_id", "alias", "title", "bgg_id", "is_addition",
    "platform",   # о какой площадке значение: tesera | bgg
    "source",     # откуда взято: tesera_api_list | tesera_api_game | tesera_html
    "block",      # rating | subrating | audience | market | content | classification | specs | credits | awards | meta
    "metric",     # конкретная метрика внутри блока
    "dim_l1",     # 1-й уровень измерения (жанр, роль, название награды...)
    "dim_l2",     # 2-й уровень (поджанр, результат награды...)
    "seq",        # порядок значения внутри многозначного поля
    "value_num", "value_text", "unit",
    "source_url",
]

# поле API -> (platform, block, metric, unit)
LIST_FIELDS = {
    "ratingUser":          ("tesera", "rating", "avg_rating", "pts_10"),
    "n10Rating":           ("tesera", "rating", "bayes_n10_rating", "pts_10"),
    "n20Rating":           ("tesera", "rating", "bayes_n20_rating", "pts_10"),
    "numVotes":            ("tesera", "rating", "num_votes", "votes"),
    "bggRating":           ("bgg", "rating", "avg_rating", "pts_10"),
    "bggGeekRating":       ("bgg", "rating", "geek_rating", "pts_10"),
    "bggNumVotes":         ("bgg", "rating", "num_votes", "votes"),
    "year":                ("tesera", "specs", "year_published", "year"),
    "playersMin":          ("tesera", "specs", "players_min", "players"),
    "playersMax":          ("tesera", "specs", "players_max", "players"),
    "playersMinRecommend": ("tesera", "specs", "players_recommended_min", "players"),
    "playersMaxRecommend": ("tesera", "specs", "players_recommended_max", "players"),
    "playersAgeMin":       ("tesera", "specs", "age_min", "years"),
    "timeToLearn":         ("tesera", "specs", "time_to_learn", "min"),
    "playtimeMin":         ("tesera", "specs", "playtime_min", "min"),
    "playtimeMax":         ("tesera", "specs", "playtime_max", "min"),
    "commentsTotal":       ("tesera", "content", "comments", "count"),
    "creationDateUtc":     ("tesera", "meta", "created_utc", "datetime"),
    "modificationDateUtc": ("tesera", "meta", "modified_utc", "datetime"),
}
GAME_FIELDS = {
    "ownersTotal":        ("audience", "owners", "users"),
    "sellTotal":          ("market", "sell_offers_active", "offers"),
    "buyTotal":           ("market", "buy_requests_active", "offers"),
    "sellTotalAll":       ("market", "sell_offers_all_time", "offers"),
    "buyTotalAll":        ("market", "buy_requests_all_time", "offers"),
    "reportsTotal":       ("content", "play_reports", "count"),
    "photosTotal":        ("content", "photos", "count"),
    "filesTotal":         ("content", "files", "count"),
    "linksTotal":         ("content", "links", "count"),
    "videoExternalTotal": ("content", "videos_external", "count"),
    "videoInternalTotal": ("content", "videos_internal", "count"),
}
SUBRATINGS = {
    "game_rating_gameplay": "gameplay",
    "game_rating_depth": "depth",
    "game_rating_orig": "originality",
    "game_rating_realiz": "implementation",
}
AUDIENCE_TABS = {  # заголовок вкладки в HTML -> метрика
    "владеют": "owners", "играли": "played", "хотят сыграть": "want_to_play",
    "фанаты": "fans", "продают": "selling_now", "покупают": "buying_now",
}
SPEC_ICONS = {"навыки игры": "skill"}
CREDIT_ROLES = {  # подпись в таблице «Информация» -> роль
    "автор": "author", "художник": "artist", "издатель": "publisher",
    "магазин": "shop", "язык": "language",
}


def num(s):
    if s is None:
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(s).replace("\xa0", " "))
    return float(m.group().replace(",", ".")) if m else None


def get(session, url, retries=4, **params):
    for attempt in range(retries):
        time.sleep(DELAY * (1 + 2 * attempt))
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise


def parse_html(html):
    """Вернёт список (platform, block, metric, dim_l1, dim_l2, seq, value_num, value_text, unit)."""
    soup = BeautifulSoup(html, "lxml")
    out = []

    # жанр -> поджанры (хлебные крошки)
    crumbs = soup.select_one("div.breadcrumbs")
    if crumbs:
        genre = None
        for i, a in enumerate(crumbs.select("a.k, a.o")):
            name = a.get_text(strip=True)
            if "k" in a.get("class", []):
                genre = name
                out.append(("tesera", "classification", "genre", name, None, i, None, name, None))
            else:
                out.append(("tesera", "classification", "subgenre", genre, name, i, None, name, None))
    # темы/сеттинги
    for i, a in enumerate(soup.select("div.breadcrumbs-right a.filter_submit")):
        out.append(("tesera", "classification", "theme", None, None, i, None, a.get_text(strip=True), None))
    # теги (механики)
    for i, a in enumerate(soup.select("ul.tags li a")):
        out.append(("tesera", "classification", "tag", None, None, i, None, a.get_text(strip=True), None))
    # навыки игры (иконки под заголовком)
    for li in soup.select("ul.classnav li"):
        img = li.find("img")
        label = img.get("title") if img else None
        if label in SPEC_ICONS:
            for i, s in enumerate(x.strip() for x in li.get_text().split(",")):
                out.append(("tesera", "classification", SPEC_ICONS[label], None, None, i, None, s, None))

    # суб-рейтинги
    for div_id, metric in SUBRATINGS.items():
        div = soup.find(id=div_id)
        if div:
            td = div.find_parent("tr").find("td", class_="rating")
            out.append(("tesera", "subrating", metric, None, None, 0, num(td.get_text()), None, "pts_10"))

    # вкладки аудитории: владеют / играли / хотят сыграть / фанаты / продают / покупают
    for title in soup.select("div.smt div.title"):
        a, cnt = title.find("a"), title.find("span", class_="l")
        if a and cnt:
            label = a.get_text(strip=True)
            if label in AUDIENCE_TABS:
                out.append(("tesera", "audience", AUDIENCE_TABS[label], None, None, 0,
                            num(cnt.get_text()), None, "users"))

    # «Информация» и «Поставка»: таблицы class=specs с подписью в первой ячейке
    for tr in soup.select("table.specs tr"):
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue
        label = tds[0].get_text(" ", strip=True).split(":")[0].strip().lower()
        if label in CREDIT_ROLES and len(tds) > 1:
            links = tds[1].find_all("a")
            values = [a.get_text(strip=True) for a in links] or \
                     [x.strip() for x in tds[1].get_text("\n").split("\n") if x.strip()]
            for i, v in enumerate(values):
                out.append(("tesera", "credits", CREDIT_ROLES[label], None, None, i, None, v, None))
        elif label == "вес" and len(tds) > 1:
            out.append(("tesera", "specs", "box_weight", None, None, 0, num(tds[1].get_text()), None, "g"))
        elif label == "размер" and len(tds) > 1:
            dims = re.findall(r"\d+(?:[.,]\d+)?", tds[1].get_text())
            for i, (name, d) in enumerate(zip(["box_length", "box_width", "box_height"], dims)):
                out.append(("tesera", "specs", name, None, None, i, float(d.replace(",", ".")), None, "cm"))
        elif label == "комплектация":
            p = tds[0].find("p")
            items = [x.strip() for x in p.get_text("\n").split("\n") if x.strip()] if p else []
            for i, v in enumerate(items):
                out.append(("tesera", "specs", "component", None, None, i, num(v), v, None))

    # награды: год, название, результат
    for i, li in enumerate(soup.select("ul.awards li")):
        a = li.find("a")
        if not a:
            continue
        year = li.find("span")
        result = li.get_text(" ", strip=True).split("—")[-1].strip()
        out.append(("tesera", "awards", "award", a.get("title") or a.get_text(strip=True), result, i,
                    num(year.get_text()) if year else None, None, "year"))

    # связанные игры: dim_l1 = тип связи («исходная игра», «в одной серии»...), value_text = alias связанной игры
    for i, (kind, alias) in enumerate(parse_relations(soup)):
        out.append(("tesera", "relations", "related_game", kind, None, i, None, alias, None))
    return out


def parse_relations(soup):
    """Блок «Связанные игры»: список (тип связи, alias связанной игры)."""
    rel = []
    for box in soup.select("div.gameslinked"):
        kind, link = box.select_one("div.kind"), box.select_one("div.text h3 a")
        if kind and link:
            rel.append((kind.get_text(strip=True), link["href"].strip("/").split("/")[-1]))
    return rel


SORT_FIELD = {"-ratingn10": "n10Rating", "-ratinggeekbgg": "bggGeekRating"}
PAGE = 50


def fetch_list(session, sort, limit):
    """Топ-N игр. В API offset — номер страницы, а не позиция; limit > 100 режется.
    При равных рейтингах порядок между страницами плавает (бывают дубли и пропуски),
    поэтому берём страницы с запасом, убираем дубли и пересортировываем сами."""
    by_id = {}
    page = 0
    while len(by_id) < limit + PAGE:
        batch = get(session, f"{API}/games", offset=page, limit=PAGE, sort=sort).json()
        if not batch:
            break
        for g in batch:
            by_id.setdefault(g["id"], g)
        page += 1
    field = SORT_FIELD[sort]
    return sorted(by_id.values(), key=lambda g: g.get(field) or 0, reverse=True)[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--sort", default="-ratingn10", help="-ratingn10 = топ Tesera, -ratinggeekbgg = топ BGG")
    ap.add_argument("--out", default="data/tesera_top50_long.csv")
    args = ap.parse_args()

    session = requests.Session()
    snapshot = dt.date.today().isoformat()
    games = fetch_list(session, args.sort, args.limit)

    rows = []
    for rank, g in enumerate(games, start=1):
        alias = g["alias"]
        key = dict(snapshot_date=snapshot, list_sort=args.sort, list_rank=rank, tesera_id=g["teseraId"],
                   alias=alias, title=g["title"], bgg_id=g.get("bggId"), is_addition=g.get("isAddition"))

        def add(platform, source, block, metric, dim_l1=None, dim_l2=None, seq=0,
                value_num=None, value_text=None, unit=None, url=None):
            rows.append({**key, "platform": platform, "source": source, "block": block, "metric": metric,
                         "dim_l1": dim_l1, "dim_l2": dim_l2, "seq": seq, "value_num": value_num,
                         "value_text": value_text, "unit": unit, "source_url": url})

        list_url = f"{API}/games?sort={args.sort}"
        for field, (platform, block, metric, unit) in LIST_FIELDS.items():
            v = g.get(field)
            if v is None:
                continue
            is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
            add(platform, "tesera_api_list", block, metric,
                value_num=v if is_num else None, value_text=None if is_num else v, unit=unit, url=list_url)

        game_url = f"{API}/games/{alias}"
        try:
            card = get(session, game_url).json()
            if card["game"].get("title2"):
                add("tesera", "tesera_api_game", "meta", "title_original", value_text=card["game"]["title2"], url=game_url)
            for field, (block, metric, unit) in GAME_FIELDS.items():
                if card.get(field) is not None:
                    add("tesera", "tesera_api_game", block, metric, value_num=card[field], unit=unit, url=game_url)
        except requests.RequestException as e:
            add("tesera", "tesera_api_game", "meta", "collect_error", value_text=str(e), url=game_url)

        page_url = f"{SITE}/game/{alias}/"
        try:
            html = get(session, page_url).text
            for platform, block, metric, d1, d2, seq, vnum, vtext, unit in parse_html(html):
                add(platform, "tesera_html", block, metric, d1, d2, seq, vnum, vtext, unit, page_url)
        except requests.RequestException as e:
            add("tesera", "tesera_html", "meta", "collect_error", value_text=str(e), url=page_url)

        print(f"{rank:>3}. {g['title']}: {sum(r['alias'] == alias for r in rows)} строк")

    write_csv(pd.DataFrame(rows, columns=COLUMNS), args.out)
    print(f"Итого {len(rows)} строк -> {args.out}")


if __name__ == "__main__":
    main()
