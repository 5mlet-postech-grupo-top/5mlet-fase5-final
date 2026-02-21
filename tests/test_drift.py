import json
import sqlite3
import time
from app.main import app
import pandas as pd
from fastapi.testclient import TestClient

from src.utils import DATA_DIR

client = TestClient(app)

def test_endpoint_drift():
    """Garante que a rota de monitoramento de Data Drift funciona e não quebra."""
    response = client.get("/drift")
    assert response.status_code == 200
    dados = response.json()
    assert "message" in dados or "psi" in dados