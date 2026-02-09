# Datathon – Machine Learning Engineering (PEDE Passos Mágicos)

API e pipeline de treinamento para **prever risco de defasagem educacional** (aluno atrasado em relação ao nível ideal).

## Visão geral
- **Target (classificação binária):** `1` se `DEFASAGEM_2021 < 0` (aluno atrasado), senão `0`.
- **Features (anti-leakage):** principalmente colunas `*_2020` + feature derivada `ANOS_PM_POR_IDADE_2020`.
- **Modelo:** `RandomForestClassifier` (com pipeline sklearn + ColumnTransformer).
- **Métricas acompanhadas:** AUC, Recall e F1 (Recall é crítico para reduzir falsos negativos).

## Estrutura do projeto
```
project-root/
  app/
    main.py
    routes.py
    model/
      model.joblib
      preprocessor.joblib
      metadata.json
  src/
    preprocessing.py
    feature_engineering.py
    train.py
    evaluate.py
    utils.py
  tests/
    test_preprocessing.py
    test_api.py
  Dockerfile
  requirements.txt
  README.md
```

## Como treinar localmente
Pré-requisitos: Python 3.11+

```bash
pip install -r requirements.txt
python -m src.train --data "/caminho/PEDE_PASSOS_DATASET_FIAP.xlsx"
```

Os artefatos serão salvos em `app/model/` e a referência de treino (para drift) em `data/train_reference.csv`.

## Como rodar a API localmente
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict`
- `GET /metrics` (Prometheus)
- `GET /drift` (PSI simples: treino vs produção)

### Exemplo de chamada (/predict)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IDADE_ALUNO_2020": 13,
    "ANOS_PM_2020": 3,
    "INDE_2020": 6.7,
    "IEG_2020": 7.1,
    "IDA_2020": 6.2,
    "IAN_2020": 6.8,
    "IPS_2020": 7.0,
    "IPP_2020": 6.5,
    "IPV_2020": 6.9,
    "IAA_2020": 7.2,
    "FASE_TURMA_2020": "2A",
    "PEDRA_2020": "Ametista",
    "INSTITUICAO_ENSINO_ALUNO_2020": "Escola Estadual"
  }'
```

## Docker
```bash
docker build -t pede-mlops .
docker run -p 8000:8000 pede-mlops
```

## Testes e cobertura
```bash
pytest -q --cov=src --cov=app --cov-report=term-missing --cov-fail-under=80
```

## Monitoramento e Drift
- **Logs estruturados (JSON)** via `python-json-logger`.
- **Prometheus**: endpoint `/metrics`.
- **Drift**: endpoint `/drift` calcula PSI por feature numérica comparando `data/train_reference.csv` (treino)
  com amostras de produção registradas em `data/predictions.sqlite`.

## Deploy
- Local via Docker (acima).
- Em cloud: Render / Cloud Run / ECS (basta publicar imagem Docker e apontar a porta 8000).
