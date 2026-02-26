import json
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ativo"


def test_metrics_endpoint():
    # uma varredura simples deve retornar métricas em texto plano com os contadores definidos
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")
    assert "api_requests_total" in r.text


def test_predict_increases_metrics():
    # enviar um payload mínimo válido e depois ler as métricas
    payload = {
        "IDADE": 15,
        "INDE": 6.0,
        "IEG": 7.0,
        "IDA": 6.0,
        "PONTO_VIRADA": 0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    m = client.get("/metrics")
    assert "api_requests_total" in m.text
    assert 'endpoint="/predict"' in m.text


def test_predict_without_artifacts(tmp_path, monkeypatch):
    # monkeypatch load_artifacts to simulate missing artifacts
    import app.routes as routes
    monkeypatch.setattr(routes, "load_artifacts", lambda: (_ for _ in ()).throw(RuntimeError("Model artifacts not found")))
    payload = {"IDADE": 10, "INDE": 5, "IEG": 5, "IDA": 5, "PONTO_VIRADA": 0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 500
    assert "Model artifacts not found" in r.text


def test_util_funcs():
    # test internal helpers via import
    from app import routes
    # student id extraction
    assert routes._extract_student_id({"student_id": "abc"}) == "abc"
    assert routes._extract_student_id({}) .startswith("anon:")
    assert routes._json_safe_number(float("nan")) is None
    assert routes._json_safe_number(5) == 5
