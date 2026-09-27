"""Единый формат CSV проекта — под Power BI с русской локалью.

Разделитель столбцов «;», десятичный разделитель «,», UTF-8 с BOM.
Целые числа пишутся без хвоста «,0».
"""
import pandas as pd

READ_KW = dict(sep=";", decimal=",", encoding="utf-8-sig", low_memory=False)

INT_COLUMNS = ["list_rank", "tesera_id", "bgg_id", "seq"]


def _fmt(x):
    return f"{x:.10g}".replace(".", ",")


def write_csv(df: pd.DataFrame, path: str) -> None:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f" and (c in INT_COLUMNS or (df[c].dropna() % 1 == 0).all()):
            df[c] = df[c].astype("Int64")
    df.to_csv(path, index=False, sep=";", encoding="utf-8-sig", float_format=_fmt)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, **READ_KW)
