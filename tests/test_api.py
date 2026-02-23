import json
import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.data_loader import load_all_training_data
from src.utils import DATA_DIR
from app.main import app
from app.routes import load_artifacts

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_endpoint_predict_sucesso():
    """Garante que a predição funciona e aplica o threshold otimizado."""
    # Payload simulando os dados de um aluno
    payload = {
        "student_id": "RA-9999",
        "IDADE": 15,
        "FASE_TURMA": "5G",
        "INDE": 5.5,
        "IDA": 4.0,
        "IEG": 6.0,
        "IAA": 7.0,
        "IPS": 5.0,
        "IPV": 5.0,
        "IAN": 5.0
    }

    response = client.post("/predict", json=payload)

    # A API deve responder 200 OK
    assert response.status_code == 200

    dados = response.json()
    assert "risk_score" in dados
    assert "risk_class" in dados
    assert "top_risk_factors" in dados

    # Valida se o threshold de negócio (0.35) está a ser enviado na resposta
    assert dados["threshold"] == 0.35


def test_endpoint_metrics():
    """Garante que o Prometheus está a exportar as métricas corretamente."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text

def test_load_all_training_data():
    """Garante que a função consegue encontrar e carregar dados no ambiente."""
    try:
        df = load_all_training_data(DATA_DIR)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    except FileNotFoundError:
        # Se os dados não estiverem na máquina de CI, não deve falhar
        pass


def test_load_artifacts_from_huggingface(tmp_path):
    """
    Garante que a API tenta baixar do Hugging Face se os artefatos locais não existirem,
    sem fazer requisições reais à internet (Mock).
    """
    # 1. Limpa o cache da API para forçar a função a rodar de novo
    load_artifacts.cache_clear()

    # 2. Usamos o patch para simular (mockar) dependências externas
    # tmp_path é uma pasta temporária vazia criada pelo próprio pytest
    with patch("app.routes.ARTIFACT_DIR", tmp_path), \
            patch("app.routes.hf_hub_download") as mock_hf_download, \
            patch("app.routes.joblib.load") as mock_joblib_load, \
            patch("app.routes.load_json") as mock_load_json:
        # Dizemos o que os mocks devem retornar para o código não quebrar
        mock_joblib_load.return_value = "modelo_fake"
        mock_load_json.return_value = {"threshold": 0.35, "model_version": "hf_test"}

        # 3. Executamos a função
        model, meta = load_artifacts()

        # 4. Verificamos se o hf_hub_download foi chamado exatamente 2 vezes
        # (uma para o model.joblib e outra para o metadata.json)
        assert mock_hf_download.call_count == 2

        # Garante que os valores retornados são os que injetamos
        assert model == "modelo_fake"
        assert meta["threshold"] == 0.35

    # Limpa o cache novamente para não interferir em outros testes
    load_artifacts.cache_clear()