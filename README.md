# Passos Mágicos MLOps: Previsão de Defasagem Escolar

**API em Produção (Cloud):** [https://fivemlet-fase5-final.onrender.com/](https://www.google.com/search?q=https://fivemlet-fase5-final.onrender.com/docs)

Este repositório entrega um projeto de Machine Learning End-to-End (Pipeline de Dados, Treinamento, API, Docker, Testes Unitários e Monitoramento) desenvolvido para a **Associação Passos Mágicos**. O objetivo é prever o risco de defasagem educacional de estudantes e permitir intervenções pedagógicas preventivas.

## Arquitetura e Decisões de Negócio

* **Schema Enforcement (Anti-Leakage e Padronização):** O módulo `src/preprocessing.py` lê múltiplos arquivos `.xlsx` (FIAP 2021 vs Base 2024) com schemas distintos e os padroniza em tempo real. Entradas "sujas" (como notas com vírgulas ou texto em colunas numéricas) são tratadas antes de atingirem o modelo.
* **Otimização de Threshold (Foco em Recall):** Ajustamos o ponto de corte do RandomForestClassifier para **0.35**. Para o contexto de assistência social, priorizamos **Falsos Positivos** sobre **Falsos Negativos**. Com essa calibração, o modelo atinge >97% de Recall, garantindo que a esmagadora maioria das crianças em risco seja detectada.
* **Model Registry (Hugging Face):** Os binários pesados (`.joblib`) não poluem o repositório Git. A API realiza o download automático da versão mais recente do modelo hospedado no Hugging Face durante o startup do container.
* **Observabilidade e Explicabilidade:** Integração nativa com métricas do Prometheus e explicabilidade global/local usando SHAP Values.

## Estrutura do Projeto

```text
app/
  main.py             # Inicialização do FastAPI e Prometheus ASGI
  routes.py           # Endpoints, regras de negócio e pull do Hugging Face
  model/              # Artefatos do modelo (baixados automaticamente)
src/
  data_loader.py      # Ingestão de ./data/*.xlsx
  preprocessing.py    # Schema Enforcement e Target Engineering
  feature_engineering.py
  train.py            # Pipeline de treino, validação e metadados
  utils.py            # Cálculo de PSI (Drift), logs e helpers
tests/                # Suíte de testes (Pytest + Mocks)
Dockerfile            # Containerização Multi-stage (Non-root)
requirements.txt

```

## Como Executar Localmente

**Pré-requisitos:** Python 3.11+ e arquivos `.xlsx` na pasta `./data/`.

**1. Treinar o modelo:**

```bash
python -m src.train

```

*Gera: `app/model/model.joblib`, `app/model/metadata.json` e `data/train_reference.csv` (para cálculo de drift).*

**2. Subir a API:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

```

## Deploy com Docker

A aplicação está conteinerizada seguindo boas práticas de segurança (execução com usuário não-root).

```bash
docker build -t passos-magicos-api .
docker run -p 8000:8000 passos-magicos-api

```

## Deploy na Nuvem (Render & Hugging Face)

A infraestrutura foi desenhada para CI/CD Serverless. O deploy atual está hospedado no Render (camada gratuita).

* **Nota de Cold Start:** A primeira requisição à API em produção pode levar até 50 segundos para responder caso o container esteja "adormecido". As requisições subsequentes ocorrem em tempo real.
* O download do modelo a partir do Hugging Face Hub é feito de forma transparente pela função `load_artifacts` caso a pasta `app/model` esteja vazia no servidor.

## Endpoints e Exemplos de Uso

A API (disponível visualmente em `/docs`) é flexível. Você pode enviar as chaves em qualquer ordem e até omitir algumas; o serviço reordena e imputa os dados automaticamente de acordo com o pipeline treinado.

### Predição de Risco (POST /predict)

```bash
curl -X POST https://URL_DO_SEU_RENDER_AQUI/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "RA-9999",
    "IDADE": 13,
    "INDE": 6.7,
    "IEG": 7.1,
    "IDA": 6.2,
    "FASE_TURMA": "3-A",
    "PEDRA": "Ametista",
    "INSTITUICAO": "Escola Pública"
  }'

```

**Resposta (inclui explicabilidade SHAP):**
O endpoint retorna o `risk_score`, a classe de risco e os `top_risk_factors` calculados dinamicamente via SHAP (ou Feature Importances globais como fallback).

### Explicabilidade Histórica (GET /explain)

Retorna o histórico de inferências salvas no banco SQLite local (`predictions.sqlite`).

```bash
curl "https://URL_DO_SEU_RENDER_AQUI/explain?student_id=RA-9999&limit=5"

```

### Data Drift (GET /drift)

Compara a distribuição dos dados de treino (Ground Truth) com as inferências em produção usando o **Population Stability Index (PSI)**.

```bash
curl "https://URL_DO_SEU_RENDER_AQUI/drift"

```

### Métricas (GET /metrics)

Expõe contadores de requisições e histogramas de latência formatados para scraping pelo Prometheus.

## Qualidade de Código (Testes Unitários)

O projeto possui cobertura de código superior a 80%, validando o Schema Enforcement, padronização de targets e respostas da API. As integrações externas (Hugging Face) são isoladas através de bibliotecas de Mocking.

```bash
pytest -q --cov=src --cov=app --cov-report=term-missing --cov-fail-under=80 tests/

```