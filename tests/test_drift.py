import json
import sqlite3
import time

import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app
from src.utils import DATA_DIR


def test_drift_no_production_returns_message(tmp_path):
    # ensure no DB
    db = DATA_DIR / "predictions.sqlite"
    if db.exists():
        db.unlink()

    # create reference file
    ref = pd.DataFrame({"INDE_2020": [5, 6, 7], "IEG_2020": [5, 6, 7]})
    DATA_DIR.mkdir(exist_ok=True)
    ref.to_csv(DATA_DIR / "train_reference.csv", index=False)

    app = create_app()
    client = TestClient(app)

    r = client.get("/drift")
    assert r.status_code == 200
    assert "message" in r.json()
