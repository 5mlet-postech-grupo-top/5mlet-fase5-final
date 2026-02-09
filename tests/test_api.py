import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.main import create_app
from src.utils import ARTIFACT_DIR, save_json


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifacts():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Minimal training set
    df = pd.DataFrame({
        "IDADE_ALUNO_2020": [10, 11, 12, 13],
        "ANOS_NA_PM_2020": [1, 2, 2, 3],
        "INDE_2020": [5.0, 6.0, 6.5, 7.2],
        "IEG_2020": [5.0, 6.0, 6.0, 7.0],
        "FASE_TURMA_2020": ["1A", "1B", "2A", "2B"],
        "PEDRA_2020": ["Quartzo", "Ágata", "Ametista", "Ametista"],
        "INSTITUICAO_ENSINO_ALUNO_2020": ["X", "Y", "X", "Z"],
    })
    y = [1, 1, 0, 0]

    categorical = ["FASE_TURMA_2020", "PEDRA_2020", "INSTITUICAO_ENSINO_ALUNO_2020"]
    numeric = ["IDADE_ALUNO_2020", "ANOS_NA_PM_2020", "INDE_2020", "IEG_2020"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    model = LogisticRegression(max_iter=200)
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(df, y)

    joblib.dump(pipe, ARTIFACT_DIR / "model.joblib")
    joblib.dump(pipe.named_steps["preprocessor"], ARTIFACT_DIR / "preprocessor.joblib")

    save_json(ARTIFACT_DIR / "metadata.json", {
        "model_version": "test",
        "threshold": 0.5,
        "drift_bins": {"INDE_2020": [0, 10], "IEG_2020": [0, 10]},
    })


def test_predict_endpoint_returns_score():
    app = create_app()
    client = TestClient(app)

    payload = {
        "IDADE_ALUNO_2020": 13,
        "ANOS_NA_PM_2020": 3,
        "INDE_2020": 6.7,
        "IEG_2020": 7.1,
        "FASE_TURMA_2020": "2A",
        "PEDRA_2020": "Ametista",
        "INSTITUICAO_ENSINO_ALUNO_2020": "X"
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["model_version"] == "test"
