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

> **Dica:** há um `docker-compose.yml` incluído que levantará a API junto com
> Prometheus e Grafana. basta executar `docker-compose up --build` e abrir:
> - API:  http://localhost:8000
> - Prometheus: http://localhost:9090
> - Grafana:    http://localhost:3000 (usuário `admin`/senha `admin`)

## Monitoramento (Prometheus + Grafana)

A aplicação já exporta métricas de instrumentação no endpoint `/metrics`.
As métricas básicas geradas são:

- `api_requests_total{endpoint,status}` – contador de chamadas por rota
- `api_request_latency_seconds_bucket{endpoint,...}` – histograma de latência

Você pode apontar o Prometheus para esse caminho usando o `prometheus.yml`
fornecido (o job `pede-api` já está configurado) ou adicionando manualmente um
datasource no Grafana.

### Iniciando o stack via Docker Compose

```bash
# sobe a API + Prometheus + Grafana
docker-compose up --build
```

Após o container do Grafana estar pronto, crie um *datasource*:

1. abra http://localhost:3000 e entre (admin/admin)
2. vá em **Configuration → Data Sources → Add data source**
3. escolha **Prometheus** e use `http://prometheus:9090` como URL
4. salve e teste

### Painéis de exemplo

- **Requests por endpoint/status:**
  ```promql
  sum by(endpoint,status)(api_requests_total)
  ```
- **Latência p95 por endpoint:**
  ```promql
  histogram_quantile(0.95, sum(rate(api_request_latency_seconds_bucket[5m])) by (le,endpoint))
  ```

Você pode importar um dashboard pronto (por exemplo, o JSON `grafana_dashboard.json`
fornecido no repositório) ou montar seus próprios gráficos usando as consultas acima.

> **Importando automaticamente** (após o datasource estar configurado):
> ```bash
> curl -s -X POST http://localhost:3000/api/dashboards/db \
>      -H "Content-Type: application/json" \
>      -u admin:admin \
>      --data @grafana_dashboard.json
> ```

### Gerando tráfego para popular o Prometheus

Para simular uso real e gerar dados de monitoramento:

```bash
# 1. levante o stack
docker-compose up --build

# 2. em outro terminal, execute o gerador de tráfego
pip install -r requirements.txt
python scripts/generate_traffic.py --num-requests 200 --delay 0.1

# 3. acesse o Prometheus para verificar as métricas
#    http://localhost:9090
#    Consulte: api_requests_total ou api_request_latency_seconds

# 4. configure o datasource no Grafana
#    Abra: http://localhost:3000 (admin/admin)
#    Configuration → Data Sources → Add Prometheus
#    URL: http://prometheus:9090

# 5. importe o dashboard (opcional)
curl -s -X POST http://localhost:3000/api/dashboards/db \
     -H "Content-Type: application/json" \
     -u admin:admin \
     -d @grafana_dashboard.json

## Testes (coverage >= 80%)
```bash
pytest -q --cov=src --cov=app --cov-report=term-missing --cov-fail-under=80
```
