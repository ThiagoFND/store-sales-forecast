from __future__ import annotations
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

FEATURES = [
    "onpromotion", "onpromotion_per_day", "days_in_month",
    "lag_1", "lag_12", "ma_3", "ma_6", "ma_12",
    "year", "month"
]


def build_preprocessor(df: pd.DataFrame):
    """Define pipeline de pré-processamento (numérico + categórico)."""
    num_feats = [f for f in FEATURES if f not in ("year", "month")]
    cat_feats = ["year", "month"]

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
    ])


def make_models():
    """Modelos disponíveis."""
    return {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "rf": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }


def fit_per_family(train: pd.DataFrame, model_name: str) -> dict[str, Pipeline]:
    """Treina 1 modelo por família."""
    pre = build_preprocessor(train)
    base = make_models()[model_name]
    family_models = {}

    for fam, g in train.groupby("family"):
        X, y = g[FEATURES], g["sales"]
        pipe = Pipeline([("pre", pre), ("model", base)])
        pipe.fit(X, y)
        family_models[fam] = pipe

    return family_models


def predict_per_family(models: dict[str, Pipeline], df: pd.DataFrame) -> pd.DataFrame:
    """Faz previsões por família usando os modelos treinados."""
    preds = []
    for fam, g in df.groupby("family"):
        if fam not in models:
            continue
        X = g[FEATURES]
        yhat = models[fam].predict(X)
        out = g[["family", "ym", "ym_dt", "sales"]].copy()
        out["pred"] = yhat
        preds.append(out)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
