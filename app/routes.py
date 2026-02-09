from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.feature_engineering import add_derived_features
from src.utils import ARTIFACT_DIR, DATA_DIR, compute_psi, load_json, logger


router = APIRouter()

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

DB_PATH = DATA_DIR / "predictions.sqlite"


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
            ts INTEGER NOT NULL,
            payload TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_class INTEGER NOT NULL,
            model_version TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def load_model():
    model_path = ARTIFACT_DIR / "model.joblib"
    meta_path = ARTIFACT_DIR / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise RuntimeError("Model artifacts not found. Run src/train.py first.")
    model = joblib.load(model_path)
    meta = load_json(meta_path)
    return model, meta


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Allow any feature fields


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@router.post("/predict")
async def predict(req: Request, body: PredictRequest):
    t0 = time.time()
    endpoint = "/predict"
    try:
        model, meta = load_model()
        payload = dict(body)

        X = pd.DataFrame([payload])
        X = add_derived_features(X)

        proba = float(model.predict_proba(X)[:, 1][0])
        threshold = float(meta.get("threshold", 0.5))
        pred = int(proba >= threshold)

        out = {
            "risk_score": proba,
            "risk_class": pred,
            "threshold": threshold,
            "model_version": meta.get("model_version"),
            "interpretation": "Alto risco de defasagem" if pred == 1 else "Baixo risco de defasagem",
        }

        # persist sample for drift/monitoring
        conn = _db()
        conn.execute(
            "INSERT INTO predictions(ts, payload, risk_score, risk_class, model_version) VALUES (?, ?, ?, ?, ?)",
            (int(time.time()), json.dumps(payload, ensure_ascii=False), proba, pred, out["model_version"]),
        )
        conn.commit()
        conn.close()

        REQUESTS.labels(endpoint=endpoint, status="200").inc()
        return out
    except Exception as e:
        REQUESTS.labels(endpoint=endpoint, status="500").inc()
        logger.exception("predict_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        LATENCY.labels(endpoint=endpoint).observe(time.time() - t0)


@router.get("/drift")
def drift(limit: int = 1000):
    """Compute simple PSI drift vs training (numeric only).
    PSI guideline: <0.1 no drift, 0.1-0.25 moderate, >0.25 significant.
    """
    model, meta = load_model()
    bins_map: Dict[str, Any] = meta.get("drift_bins", {})

    if not DB_PATH.exists():
        return {"message": "No production data logged yet."}

    conn = _db()
    rows = conn.execute(
        "SELECT payload FROM predictions ORDER BY ts DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    conn.close()

    if not rows:
        return {"message": "No production data logged yet."}

    payloads = [json.loads(r[0]) for r in rows]
    prod = pd.DataFrame(payloads)

    # load training reference from metadata if available; else drift only within production
    # Here we approximate reference distribution using bins + production; PSI needs expected.
    # We store only bins, so we compute expected distribution using saved quantile bins
    # and approximate expected counts assuming uniform across bins isn't ideal.
    # To keep it correct, we also store training feature samples in data/train_reference.parquet when training (optional).
    ref_path = DATA_DIR / "train_reference.csv"
    if not ref_path.exists():
        return {
            "message": "Training reference not found (data/train_reference.parquet). Re-run training with --save-reference.",
            "hint": "You can still use this endpoint after enabling reference saving.",
        }

    ref = pd.read_csv(ref_path)

    results = {}
    for col, bins in bins_map.items():
        if col not in prod.columns or col not in ref.columns:
            continue
        b = np.array(bins, dtype=float)
        exp = pd.to_numeric(ref[col], errors="coerce").to_numpy()
        act = pd.to_numeric(prod[col], errors="coerce").to_numpy()
        psi = compute_psi(exp, act, b)
        results[col] = psi

    # summarize
    if results:
        worst = sorted(results.items(), key=lambda x: (np.nan_to_num(x[1], nan=-1)), reverse=True)[:10]
    else:
        worst = []

    return {
        "n_production_samples": int(len(prod)),
        "psi": results,
        "top_drift": [{"feature": k, "psi": float(v)} for k, v in worst],
        "guideline": {"no_drift": "<0.10", "moderate": "0.10-0.25", "significant": ">0.25"},
    }
