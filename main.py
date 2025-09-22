from __future__ import annotations
import os
import pandas as pd

from src.data_prep import load_raw, monthly_aggregate
from src.features import add_time_features, add_lags_and_ma, train_val_test_split
from src.evaluation import evaluate_split, evaluate_baseline, baseline_naive
from src.modeling import fit_per_family, predict_per_family
from src.utils import rmse, mape, mase

# =======================
# Parâmetros principais
# =======================
TRAIN_PATH = os.path.join("data", "raw", "train.csv")
TEST_PATH = os.path.join("data", "raw", "test.csv")
VAL_MONTHS = 3
TEST_MONTHS = 3


def main():
    # -----------------------
    # 1) Carregar dados crus
    # -----------------------
    train_raw, test_raw = load_raw(TRAIN_PATH, TEST_PATH)

    # -----------------------
    # 2) Agregar mensalmente
    # -----------------------
    train_m = monthly_aggregate(train_raw)

    # -----------------------
    # 3) Features temporais
    # -----------------------
    train_m = add_time_features(train_m)

    # -------------------------------
    # 4) Features de lags e médias móveis
    # -------------------------------
    train_f = add_lags_and_ma(train_m, lags=(1, 12), mas=(3, 6, 12))

    # Remove apenas linhas sem lag_1 (necessário para baseline e modelos)
    train_f = train_f.dropna(subset=["lag_1"]).reset_index(drop=True)

    # -----------------------
    # 5) Split temporal
    # -----------------------
    train, val, test = train_val_test_split(
        train_f, val_months=VAL_MONTHS, test_months=TEST_MONTHS
    )

    # Salvar os conjuntos processados (requisito de entrega)
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train_processed.csv", index=False)
    val.to_csv("data/processed/val_processed.csv", index=False)
    test.to_csv("data/processed/test_processed.csv", index=False)

    # -----------------------
    # 6) Baselines
    # -----------------------
    base1_m, base1_o = evaluate_baseline(val, lag=1)
    base12_m, base12_o = evaluate_baseline(val, lag=12)

    # -----------------------
    # 7) Modelos supervisionados
    # -----------------------
    metrics_all = []
    os.makedirs("reports", exist_ok=True)

    for model_name in ["ridge", "tree", "rf"]:
        m, o = evaluate_split(train, val, model_name)
        if m is not None:
            metrics_all.append(("val", model_name, o))
            m.to_csv(os.path.join("reports", f"metrics_val_{model_name}.csv"), index=False)

    # -----------------------
    # 8) Escolher melhor modelo
    # -----------------------
    choices = [(model, o["RMSE"]) for _, model, o in metrics_all]
    choices.extend([
        ("baseline_lag1", base1_o["RMSE"]),
        ("baseline_lag12", base12_o["RMSE"]),
    ])
    choices = sorted(choices, key=lambda x: x[1])
    best_model_name = choices[0][0]

    print("➡ Melhor modelo (val RMSE):", best_model_name)

    # -----------------------
    # 9) Re-treinar no train+val e avaliar no teste
    # -----------------------
    tv = pd.concat([train, val], ignore_index=True)

    if best_model_name.startswith("baseline"):
        lag = 1 if "lag1" in best_model_name else 12
        test_pred = baseline_naive(tv, test, lag=lag)
    else:
        fam_models = fit_per_family(tv, best_model_name)
        test_pred = predict_per_family(fam_models, test)

    # -----------------------
    # 10) Salvar previsões
    # -----------------------
    os.makedirs("outputs", exist_ok=True)
    test_pred.to_csv(os.path.join("outputs", "predictions.csv"), index=False)

    # -----------------------
    # 11) Avaliar no conjunto de teste
    # -----------------------
    test_metrics = (
        test_pred.groupby("family", group_keys=False)[["sales", "pred"]]  # 👈 seleciona só colunas necessárias
        .apply(lambda g: pd.Series({
            "RMSE": rmse(g["sales"], g["pred"]),
            "MAPE": mape(g["sales"], g["pred"]),
            "MASE": mase(g["sales"], g["pred"], seasonality=12),
        }))
        .reset_index()
    )

    test_metrics["model"] = best_model_name
    test_metrics.to_csv(os.path.join("reports", "metrics_test.csv"), index=False)

    print("✅ Concluído. Arquivos salvos em outputs/, reports/ e data/processed/.")


if __name__ == "__main__":
    main()
