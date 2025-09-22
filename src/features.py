from __future__ import annotations
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas de ano e mês a partir de ym_dt."""
    df = df.copy()
    df["ym_dt"] = pd.to_datetime(df["ym_dt"])
    df["year"] = df["ym_dt"].dt.year
    df["month"] = df["ym_dt"].dt.month
    return df


def add_lags_and_ma(df: pd.DataFrame, lags=(1, 12), mas=(3, 6, 12)) -> pd.DataFrame:
    """Cria colunas de lags e médias móveis por família."""
    out = []
    for fam, g in df.groupby("family", as_index=False):
        g = g.sort_values("ym_dt")
        for L in lags:
            g[f"lag_{L}"] = g["sales"].shift(L)
        for w in mas:
            g[f"ma_{w}"] = g["sales"].rolling(window=w, min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def train_val_test_split(df: pd.DataFrame, val_months=3, test_months=3):
    """Divide os dados em treino, validação e teste respeitando ordem temporal."""
    df = df.sort_values(["family", "ym_dt"])
    unique_dates = df["ym_dt"].drop_duplicates().sort_values()

    test_start = unique_dates.iloc[-test_months]
    val_start = unique_dates.iloc[-(test_months + val_months)]

    train = df[df["ym_dt"] < val_start]
    val = df[(df["ym_dt"] >= val_start) & (df["ym_dt"] < test_start)]
    test = df[df["ym_dt"] >= test_start]

    return train, val, test
