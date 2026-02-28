import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.data_loader import load_all_training_data
from src.utils import DATA_DIR
from app.main import app
from app.routes import load_artifacts
# noinspection PyProtectedMember
from app.routes import _top_factors_shap

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

def test_top_factors_shap_coverage():
    """
    Garante que todas as novas branches de tratamento de dimensoes 
    e matrizes esparsas do SHAP sejam testadas para manter a cobertura.
    """
    # 1. Setup do pipeline mock
    mock_pipeline = MagicMock()
    mock_pre = MagicMock()

    # Simula uma matriz esparsa para cobrir o bloco hasattr(Xt, 'toarray')
    mock_sparse = MagicMock()
    mock_sparse.toarray.return_value = np.array([[1.0, 2.0]])
    mock_pre.transform.return_value = mock_sparse
    mock_pre.get_feature_names_out.return_value = ["f1", "f2"]
    mock_pipeline.named_steps = {"preprocessor": mock_pre}

    df = pd.DataFrame({"f1": [1], "f2": [2]})

    with patch("app.routes.load_shap_explainer") as mock_load:
        mock_explainer = MagicMock()
        mock_load.return_value = mock_explainer

        # Cenário 1: SHAP retorna Lista (formato antigo)
        mock_explainer.shap_values.return_value = [np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])]
        res1 = _top_factors_shap(mock_pipeline, df)
        assert res1 is not None
        assert len(res1) == 2

        # Cenário 2: SHAP retorna Array 3D (formato novo RandomForest)
        mock_explainer.shap_values.return_value = np.array([[[0.1, 0.8], [0.2, 0.9]]])
        res2 = _top_factors_shap(mock_pipeline, df)
        assert res2 is not None

        # Cenário 3: SHAP retorna Array 2D (fallback)
        mock_explainer.shap_values.return_value = np.array([[0.5, 0.6]])
        res3 = _top_factors_shap(mock_pipeline, df)
        assert res3 is not None

def test_explain_no_db():
    """Garante que a API avisa quando o banco de dados ainda não foi criado."""
    # Alteramos a forma de fazer o patch: mockamos o DB_PATH inteiro
    with patch("app.routes.DB_PATH") as mock_db_path:
        mock_db_path.exists.return_value = False

        response = client.get("/explain?student_id=RA-123")
        assert response.status_code == 200
        assert response.json() == {"message": "No prediction history yet. Call /predict first."}

def test_explain_no_history():
    """Garante que a API lida corretamente com um aluno sem previsões anteriores."""
    with patch("app.routes.DB_PATH") as mock_db_path, \
            patch("app.routes._db") as mock_db:
        mock_db_path.exists.return_value = True

        # Simula o banco de dados retornando uma lista vazia
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_db.return_value = mock_conn

        response = client.get("/explain?student_id=RA-123")
        assert response.status_code == 200
        assert "No predictions found" in response.json()["message"]

def test_explain_with_history():
    """Garante que a API formata e retorna o histórico corretamente."""
    with patch("app.routes.DB_PATH") as mock_db_path, \
            patch("app.routes._db") as mock_db:
        mock_db_path.exists.return_value = True

        # Simula uma linha real retornada pelo SQLite
        mock_row = (
            1670000000,
            0.85,
            1,
            "v1.0",
            '[{"feature": "IDADE", "impact": 0.5}]',
            '{"IDADE": 15}'
        )
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [mock_row]
        mock_db.return_value = mock_conn

        response = client.get("/explain?student_id=RA-123")
        assert response.status_code == 200
        data = response.json()

        assert data["student_id"] == "RA-123"
        assert data["count"] == 1
        assert data["latest"]["risk_score"] == 0.85
        assert data["latest"]["risk_class"] == 1
        assert data["latest"]["risk_level"] == "alto"
        assert len(data["latest"]["top_risk_factors"]) == 1

    # Limpa o cache novamente para não interferir em outros testes
    load_artifacts.cache_clear()