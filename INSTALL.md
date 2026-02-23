# PEDE Passos Mágicos — MLOps (Defasagem Risk)

Este repositório entrega um projeto end-to-end (treino + API + Docker + testes + monitoramento + drift) para **prever risco de defasagem educacional**.

## Como os 2 datasets são usados juntos (sem leakage)
- O projeto lê automaticamente **todos os arquivos `.xlsx` dentro de `./data/`**.
- Os datasets podem ter **schemas diferentes** (ex.: FIAP vs Base 2024). O módulo `src/preprocessing.py` **padroniza** ambos para um schema comum (`IDADE`, `INDE`, `IEG`, `IDA`, `IPV`, `IAN`, `PONTO_VIRADA`, etc.).
- O target é construído como:
  - `risk = 1` se **defasagem < 0** (aluno atrás do nível ideal)
  - `risk = 0` caso contrário  
  Suporta `DEFASAGEM_2021` (FIAP) e `Defas` (Base 2024).

> Observação: Ao treinar com múltiplos arquivos, o modelo aprende padrões mais robustos e o drift pode ser avaliado comparando **referência de treino** vs **amostras de produção logadas**.

## Estrutura
```
app/
  main.py
  routes.py
  model/              # artefatos gerados após o treino
src/
  data_loader.py      # lê ./data/*.xlsx
  preprocessing.py    # padroniza schema + target
  feature_engineering.py
  train.py            # treino + validação + artefatos
  utils.py            # PSI/drift, paths, helpers
tests/
  test_api.py
  test_features.py
Dockerfile
requirements.txt
```

## Pré-requisitos
- Python 3.11+
- Coloque seus datasets em `./data/` (sem precisar informar o caminho completo)

## Treinar
```bash
python -m src.train
```

Artefatos gerados:
- `app/model/model.joblib`
- `app/model/metadata.json`
- `data/train_reference.csv` (para drift)

## Subir a API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict`
- `GET /explain?student_id=...` (histórico + última explicação)
- `GET /metrics` (Prometheus)
- `GET /drift` (PSI simples)

## Exemplo de /predict
Você pode enviar as chaves em qualquer ordem e até omitir algumas. O serviço reordena/complete automaticamente para a ordem do treino.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IDADE": 13,
    "INDE": 6.7,
    "IEG": 7.1,
    "IDA": 6.2,
    "PONTO_VIRADA": 0,
    "FASE_TURMA": "3-A",
    "PEDRA": "Ametista",
    "INSTITUICAO": "Escola Estadual"
  }'
```

Resposta (inclui explicabilidade):
- `top_risk_factors` vem via **SHAP** quando disponível.
- Se SHAP falhar/ não estiver disponível no ambiente, cai em fallback (feature importances globais).

## Exemplo de /explain
O `/predict` devolve `student_id` (extraído de `student_id`/`id`/`NOME`/`Nome` quando presente; caso contrário, gera um `anon:...`).

```bash
curl "http://localhost:8000/explain?student_id=123&limit=10"
```

## Rodar com Docker
```bash
docker build -t pede-mlops .
docker run -p 8000:8000 pede-mlops
```

## Testes (coverage >= 80%)
```bash
pytest -q --cov=src --cov=app --cov-report=term-missing --cov-fail-under=80
```
Dentro da pasta `/tests/coverage_report` existe o arquivo `index.html`. Nele você pode verificar individualmente a cobertura de testes de cada arquivo/classe
