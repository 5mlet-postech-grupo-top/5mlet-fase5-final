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
    assert r.json().get("message") == "Nenhum dado de produção registrado ainda."


def test_drift_with_data(tmp_path):
    ref = pd.DataFrame({"INDE_2020": [1, 2, 3], "IEG_2020": [1, 2, 3]})
    DATA_DIR.mkdir(exist_ok=True)
    ref.to_csv(DATA_DIR / "train_reference.csv", index=False)

    db = DATA_DIR / "predictions.sqlite"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE predictions (ts INTEGER, student_id TEXT, payload TEXT, risk_score REAL, risk_class INTEGER, model_version TEXT, top_factors TEXT)"
    )
    sample = {"INDE_2020": 5, "IEG_2020": 5}
    conn.execute(
        "INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
        (int(time.time()), "s1", json.dumps(sample), 0.1, 0, "v1", json.dumps([])),
    )
    conn.commit()
    conn.close()

    app = create_app()
    client = TestClient(app)
    r = client.get("/drift")
    assert r.status_code == 200
    data = r.json()
    assert "psi" in data
    assert "n_production_samples" in data


def test_explain_endpoint(tmp_path):
    db = DATA_DIR / "predictions.sqlite"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE predictions (ts INTEGER, student_id TEXT, payload TEXT, risk_score REAL, risk_class INTEGER, model_version TEXT, top_factors TEXT)"
    )
    row = (123, "stu1", json.dumps({}), 0.5, 1, "v", json.dumps([{"feature":"x","importance":0.1}]))
    conn.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)", row)
    conn.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)", (124, "stu1", json.dumps({}), 0.6, 0, "v", ""))
    conn.commit()
    conn.close()

    app = create_app()
    client = TestClient(app)
    r = client.get("/explain", params={"student_id": "stu1", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["student_id"] == "stu1"
    assert body["count"] == 2
    assert "history" in body

    r2 = client.get("/explain", params={"student_id": "unknown"})
    assert r2.status_code == 200
    assert "No predictions" in r2.json().get("message", "")
