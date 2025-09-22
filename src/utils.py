from __future__ import annotations
import numpy as np
import pandas as pd


def ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Converte uma coluna em datetime."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


def add_month_key(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Adiciona colunas de ano, mês, dias e chave mensal (period)."""
    df = df.copy()
    df["ym"] = df[date_col].dt.to_period("M")
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["days_in_month"] = df[date_col].dt.days_in_month
    return df


# ======== Métricas ========

def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true, y_pred, seasonality: int = 12) -> float:
    """Mean Absolute Scaled Error (escala relativa ao naive sazonal)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    if n <= seasonality:
        return np.nan
    d = np.abs(np.diff(y_true, seasonality)).mean()
    if d == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / d)
