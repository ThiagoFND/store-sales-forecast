# Predição Mensal de Vendas por Família — Fase 1

Este projeto implementa a Fase 1 descrita no PDF: agrega vendas **mensalmente por `family`**,
cria *features* temporais, treina e compara modelos (Regressão Linear, Decision Tree, Random Forest),
e gera um relatório com as métricas.

## 📦 Estrutura
```
project/
├─ data/
│  ├─ raw/            # coloque aqui: train.csv e test.csv do Kaggle
│  └─ processed/      # gerado pelo pipeline
├─ notebooks/
│  └─ 00-eda.ipynb    # rascunho opcional
├─ reports/
│  └─ metrics.csv     # métricas por modelo
├─ outputs/
│  ├─ predictions.csv # previsões por (family, YYYY-MM)
│  └─ model.pkl       # melhor modelo global (opcional)
├─ src/
│  ├─ data_prep.py
│  ├─ features.py
│  ├─ modeling.py
│  ├─ evaluation.py
│  └─ utils.py
└─ main.py
```

## ▶️ Como rodar
1. Crie um ambiente e instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Coloque `train.csv` e `test.csv` em `data/raw/` (do dataset Kaggle *Store Sales – Time Series Forecasting*).
3. Execute o pipeline padrão:
   ```bash
   python main.py
   ```
   Isso irá:
   - Agregar dados para o nível **mensal × família**;
   - Criar *lags* e *médias móveis*;
   - Separar treino/validação/teste **por tempo**;
   - Treinar e comparar **Linear (Ridge)**, **DecisionTree** e **RandomForest**;
   - Calcular **baselines** (Naive e Sazonal-Naive);
   - Salvar **métricas** em `reports/metrics.csv` e **previsões** em `outputs/predictions.csv`.

## ⚙️ Parâmetros rápidos
Você pode alterar meses de validação/teste e *lags* no topo do `main.py`.

## 📝 Relatório
Use os arquivos `reports/metrics.csv` e `outputs/predictions.csv` para montar o PDF.
