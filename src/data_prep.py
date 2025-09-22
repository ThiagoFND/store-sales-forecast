from __future__ import annotations
import pandas as pd
from .utils import ensure_datetime, add_month_key


def load_raw(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega CSVs de treino e teste já convertendo datas."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = ensure_datetime(train, "date")
    test = ensure_datetime(test, "date")
    return train, test


def monthly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega vendas mensalmente por família.
    Espera colunas: date, family, sales, onpromotion.
    """
    df = add_month_key(df, "date")

    grp = (
        df.groupby(["family", "ym"], as_index=False)
          .agg(
              sales=("sales", "sum"),
              onpromotion=("onpromotion", "sum"),
              days_in_month=("days_in_month", "max")
          )
    )

    grp["onpromotion_per_day"] = grp["onpromotion"] / grp["days_in_month"].clip(lower=1)
    grp["ym_dt"] = grp["ym"].dt.to_timestamp()

    return grp.sort_values(["family", "ym_dt"]).reset_index(drop=True)
