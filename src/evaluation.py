from __future__ import annotations
import pandas as pd
from .utils import rmse, mape, mase
from .modeling import fit_per_family, predict_per_family


def evaluate_split(train: pd.DataFrame, val: pd.DataFrame, model_name: str):
    """Treina e avalia um modelo por família, retornando métricas por família e médias gerais."""
    fam_models = fit_per_family(train, model_name)
    pv = predict_per_family(fam_models, val)

    if pv.empty:
        return None, None

    metrics = (
        pv.groupby("family", group_keys=False)
          .apply(
              lambda g: pd.Series({
                  "RMSE": rmse(g["sales"], g["pred"]),
                  "MAPE": mape(g["sales"], g["pred"]),
                  "MASE": mase(g["sales"], g["pred"], seasonality=12),
              }),
              include_groups=False
          )
          .reset_index()
    )

    metrics["model"] = model_name
    overall = metrics[["RMSE", "MAPE", "MASE"]].mean().to_dict()
    return metrics, overall


def baseline_naive(df: pd.DataFrame, horizon_df: pd.DataFrame, lag=1):
    """Baseline usando apenas a coluna lag já criada."""
    m = horizon_df.copy()
    m["pred"] = m[f"lag_{lag}"]
    return m[["family", "ym", "ym_dt", "sales", "pred"]]


def evaluate_baseline(val_df: pd.DataFrame, lag: int):
    """Avalia baseline de lag (1 ou 12 meses)."""
    pv = baseline_naive(val_df, val_df, lag=lag)
    metrics = (
        pv.groupby("family", group_keys=False)
          .apply(
              lambda g: pd.Series({
                  "RMSE": rmse(g["sales"], g["pred"]),
                  "MAPE": mape(g["sales"], g["pred"]),
                  "MASE": mase(g["sales"], g["pred"], seasonality=12),
              }),
              include_groups=False
          )
          .reset_index()
    )

    metrics["model"] = f"baseline_lag{lag}"
    overall = metrics[["RMSE", "MAPE", "MASE"]].mean().to_dict()
    return metrics, overall
