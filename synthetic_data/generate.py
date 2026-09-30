"""Генерация синтетических данных для проектов портфолио в одну базу DuckDB.

Запуск (из папки synthetic_data):
    python generate.py                  # масштаб dev - маленькие таблицы для разработки
    python generate.py --scale full     # полный объём для финальных прогонов
    python generate.py --only pharmacy  # только одна схема

Схемы базы:
    pharmacy     - аптечная сеть: товары, точки, чеки, остатки, текущая матрица (кластеризация и скоринг)
    supply       - цепь поставок сети электроники: поставщик -> склады -> магазины
    meta         - описание таблиц и параметры генерации
"""
import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import supply_chain
import pharmacy_products
import pharmacy_sales

DB_PATH = Path(__file__).parent / "synthetic_projects.duckdb"
SEED = 2026

SCALES = {
    # dev: таблицы порядка тысячи строк, всё собирается за секунды
    "dev": dict(n_drugs=700, n_non_drugs=297, n_stores=40, days=3, cheque_frac=0.04, stock_max_rows=3000,
                sc_stores=12, sc_products=6, sc_days=92),
    # full: объём для финальных прогонов и графиков
    "full": dict(n_drugs=2500, n_non_drugs=700, n_stores=260, days=56, cheque_frac=0.2, stock_max_rows=None,
                 sc_stores=30, sc_products=None, sc_days=92),
}

PHARMACY_PERIOD_START = pd.Timestamp("2026-03-02")

TABLE_DOCS = {
    "pharmacy.products": "Справочник товаров: лекарства из реестра ЖНВЛП (ГРЛС, 28.09.2026) + синтетические "
                         "нелекарственные категории. Категории, розничная цена, СТМ, минимальный ассортимент - синтетика.",
    "pharmacy.stores": "Точки продаж аптечной сети (синтетика): бренд, БЕ, тип, площадь, выкладка, локация, даты работы.",
    "pharmacy.cheques": "Строки чеков (синтетика, случайная выборка чеков за период): суммы до скидки, со скидкой, "
                        "по закупке, тип продажи, остаточный срок годности.",
    "pharmacy.stock": "Снимок остатков по точкам на конец периода (синтетика).",
    "pharmacy.matrix_fact": "Действующая ассортиментная матрица сети по текущей категории точки A-E (синтетика).",
    "supply.products": "Номенклатура: смартфоны Apple (названия реальные), модель, ассортиментный статус.",
    "supply.branches": "Склады и магазины; привязка магазина к складу (синтетика).",
    "supply.prices": "Цена продажи и себестоимость номенклатуры (условные).",
    "supply.supplier_orders": "Заказы поставщику: еженедельный закуп по складам, заказано и подтверждено, дата поступления.",
    "supply.journal": "Журнал движений по складам и магазинам: поступления, отгрузки в магазины, продажи.",
    "supply.stock_now": "Текущий остаток складов и магазинов на последний день периода и товар в пути.",
    "supply.truth_daily": "Истинный остаток на начало дня и спрос покупателей из симуляции - только для проверки.",
    "supply.truth_events": "Проблемы, заложенные в симуляцию на каждом этапе цепочки - только для проверки разбора.",
    "pharmacy.store_archetypes": "Скрытый архетип, заложенный в точку при генерации. Только для проверки качества "
                                 "кластеризации, в расчётах не используется.",
}


def to_duckdb(con, schema, tables: dict):
    """Записывает датафреймы в схему. numpy-строки (np.str_) DuckDB без pyarrow не принимает - приводим к str."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns[df.dtypes == object]:
            df[c] = df[c].map(lambda v: str(v) if isinstance(v, np.str_) else v)
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")


def generate_pharmacy(con, cfg):
    rng = np.random.default_rng(SEED)
    t = time.time()
    products = pharmacy_products.build_products(cfg["n_drugs"], cfg["n_non_drugs"], seed=SEED)
    behaviour = pharmacy_sales.product_behaviour(products, rng)
    period_end = PHARMACY_PERIOD_START + pd.Timedelta(days=cfg["days"] - 1)
    stores, truth = pharmacy_sales.generate_stores(cfg["n_stores"], period_end, rng)
    cheques = pharmacy_sales.generate_cheques(stores, truth, products, behaviour, PHARMACY_PERIOD_START,
                                              cfg["days"], cfg["cheque_frac"], rng)
    stock = pharmacy_sales.generate_stock(stores, cheques, behaviour, period_end, rng, cfg["stock_max_rows"])
    matrix = pharmacy_sales.generate_fact_matrix(stores, products, behaviour, rng)

    tables = {"products": products, "stores": stores, "cheques": cheques, "stock": stock,
              "matrix_fact": matrix, "store_archetypes": truth}
    to_duckdb(con, "pharmacy", tables)
    print(f"pharmacy: {', '.join(f'{k} {len(v):,}' for k, v in tables.items())} ({time.time() - t:.0f} c)")
    return {f"pharmacy.{k}": len(v) for k, v in tables.items()}



def generate_supply(con, cfg):
    t = time.time()
    tables = supply_chain.generate(cfg["sc_stores"], cfg["sc_products"], cfg["sc_days"], seed=SEED)
    to_duckdb(con, "supply", tables)
    print(f"supply: {', '.join(f'{k} {len(v):,}' for k, v in tables.items())} ({time.time() - t:.0f} c)")
    return {f"supply.{k}": len(v) for k, v in tables.items()}


def write_meta(con, counts, scale):
    con.execute("CREATE SCHEMA IF NOT EXISTS meta")
    con.execute("""CREATE TABLE IF NOT EXISTS meta.tables (
        table_name VARCHAR PRIMARY KEY, description VARCHAR, n_rows BIGINT, scale VARCHAR, generated_at TIMESTAMP)""")
    for name, n in counts.items():
        con.execute("INSERT OR REPLACE INTO meta.tables VALUES (?, ?, ?, ?, now())",
                    [name, TABLE_DOCS.get(name, ""), n, scale])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="dev")
    ap.add_argument("--only", choices=["pharmacy", "supply"])
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()
    cfg = SCALES[args.scale]

    con = duckdb.connect(args.db)
    counts = {}
    if args.only in (None, "pharmacy"):
        counts.update(generate_pharmacy(con, cfg))
    if args.only in (None, "supply"):
        counts.update(generate_supply(con, cfg))
    write_meta(con, counts, args.scale)
    con.execute("CHECKPOINT")
    con.close()
    print(f"-> {args.db} ({Path(args.db).stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
