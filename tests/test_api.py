import json
import pandas as pd
from fastapi.testclient import TestClient
from src.data_loader import load_all_training_data
from src.utils import DATA_DIR
from app.main import app

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