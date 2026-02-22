from __future__ import annotations

import json
import sqlite3
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from huggingface_hub import hf_hub_download

from src.feature_engineering import add_derived_features
from src.preprocessing import enforce_types
from src.utils import ARTIFACT_DIR, DATA_DIR, compute_psi, load_json, logger

router = APIRouter()

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

DB_PATH = DATA_DIR / "predictions.sqlite"


def _db():
    """SQLite connection + best-effort migrations."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # Base table (newest schema)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
            ts INTEGER NOT NULL,
            student_id TEXT,
            payload TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_class INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            top_factors TEXT
        )"""
    )

    # Backward compatible migrations (if table was created with older schema)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "student_id" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN student_id TEXT")
    if "top_factors" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN top_factors TEXT")

    conn.commit()
    return conn


@lru_cache(maxsize=1)
def load_artifacts():
    model_path = ARTIFACT_DIR / "model.joblib"
    meta_path = ARTIFACT_DIR / "metadata.json"
    # Se os arquivos não existirem, fazemos o pull do Model Registry
    if not model_path.exists() or not meta_path.exists():
        logger.info("Artefatos não encontrados localmente. Baixando do Hugging Face...")
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        # Substitua pelo nome exato do seu repositório no Hugging Face
        REPO_ID = "tiagoparibeiro/passos-magicos-model"

        try:
            hf_hub_download(repo_id=REPO_ID, filename="model.joblib", local_dir=ARTIFACT_DIR)
            hf_hub_download(repo_id=REPO_ID, filename="metadata.json", local_dir=ARTIFACT_DIR)
        except Exception as e:
            raise RuntimeError(f"Falha ao baixar modelo do Hugging Face: {e}")

    model = joblib.load(model_path)
    meta = load_json(meta_path)
    return model, meta


@lru_cache(maxsize=1)
def load_shap_explainer():
    """Returns SHAP TreeExplainer or None if unavailable."""
    try:
        import shap  # noqa: F401
    except Exception:
        return None

    model, _ = load_artifacts()
    try:
        tree_model = model.named_steps["model"]
        import shap

        return shap.TreeExplainer(tree_model)
    except Exception:
        return None


class PredictRequest(BaseModel):
    """
        Schema flexível para receber dados de qualquer ano do Datathon.
        Mantemos campos core para gerar um bom Swagger/OpenAPI docs.
        """
    model_config = ConfigDict(extra="allow")

    student_id: Optional[str] = Field(None, description="Identificador único do aluno")
    IDADE: Optional[int] = Field(None, description="Idade do aluno")
    FASE_TURMA: Optional[str] = Field(None, description="Ex: 5G")
    INDE: Optional[float] = Field(None, description="Indicador de Desenvolvimento Educacional")

def _extract_student_id(payload: Dict[str, Any]) -> str:
    """Best-effort student identifier for history/explain."""
    for k in ("student_id", "STUDENT_ID", "id", "ID", "NOME", "Nome"):
        v = payload.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    # fallback deterministic-ish hash of the payload
    return f"anon:{abs(hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))) % 10**10}"


def _json_safe_number(x: Any):
    """Convert NaN/Inf to None for JSON compliance."""
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _ensure_expected_columns(X: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    feature_order = meta.get("feature_order") or []
    if not feature_order:
        feat_cfg = meta.get("features", {})
        feature_order = list(
            dict.fromkeys(
                (feat_cfg.get("numeric", []) + feat_cfg.get("categorical", []) + feat_cfg.get("derived", []))
            )
        )

    for col in feature_order:
        if col not in X.columns:
            X[col] = np.nan

    return X.reindex(columns=feature_order, fill_value=np.nan)


def _top_factors_shap(model_pipeline, X: pd.DataFrame, top_k: int = 5) -> Optional[List[Dict[str, Any]]]:
    explainer = load_shap_explainer()
    if explainer is None:
        return None

    try:
        pre = model_pipeline.named_steps["preprocessor"]
        Xt = pre.transform(X)

        try:
            feature_names = list(pre.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(Xt.shape[1])]

        sv = explainer.shap_values(Xt)
        if isinstance(sv, list) and len(sv) > 1:
            contrib = sv[1][0]  # class 1
        else:
            contrib = np.array(sv)[0]

        pairs = sorted(zip(feature_names, contrib), key=lambda x: abs(float(x[1])), reverse=True)[:top_k]
        return [{"feature": f, "impact": float(v)} for f, v in pairs]
    except Exception as e:
        logger.exception("shap_failed", extra={"error": str(e)})
        return None


def _top_factors_fallback(model_pipeline, top_k: int = 5) -> List[Dict[str, Any]]:
    """Always-available fallback: top global feature importances from RF."""
    try:
        pre = model_pipeline.named_steps["preprocessor"]
        tree = model_pipeline.named_steps["model"]

        try:
            feature_names = list(pre.get_feature_names_out())
        except Exception:
            n = getattr(tree, "feature_importances_", np.array([])).shape[0]
            feature_names = [f"f{i}" for i in range(n)]

        importances = getattr(tree, "feature_importances_", None)
        if importances is None:
            return []

        idx = np.argsort(np.abs(importances))[::-1][:top_k]
        return [{"feature": feature_names[i], "importance": float(importances[i])} for i in idx]
    except Exception:
        return []


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
def predict(body: PredictRequest):
    t0 = time.time()
    endpoint = "/predict"
    try:
        model, meta = load_artifacts()
        payload = body.model_dump()
        student_id = _extract_student_id(payload)

        X = pd.DataFrame([payload])
        X = add_derived_features(X)
        X = enforce_types(X)
        X = _ensure_expected_columns(X, meta)

        proba = float(model.predict_proba(X)[:, 1][0])
        threshold = float(meta.get("threshold", 0.35))
        pred = int(proba >= threshold)

        out: Dict[str, Any] = {
            "risk_score": proba,
            "risk_class": pred,
            "risk_level": "alto" if pred == 1 else "baixo",
            "threshold": threshold,
            "model_version": meta.get("model_version"),
            "interpretation": "Alto risco de defasagem" if pred == 1 else "Baixo risco de defasagem",
            "student_id": student_id,
        }

        top = _top_factors_shap(model, X, top_k=5)
        if top is None:
            top = _top_factors_fallback(model, top_k=5)
        out["top_risk_factors"] = top

        conn = _db()
        conn.execute(
            """INSERT INTO predictions(ts, student_id, payload, risk_score, risk_class, model_version, top_factors)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                int(time.time()),
                student_id,
                json.dumps(payload, ensure_ascii=False),
                proba,
                pred,
                out["model_version"],
                json.dumps(top, ensure_ascii=False),
            ),
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
    _, meta = load_artifacts()
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
    prod = add_derived_features(prod)

    ref_path = DATA_DIR / "train_reference.csv"
    if not ref_path.exists():
        return {"message": "Training reference not found (data/train_reference.csv). Re-run training."}

    ref = pd.read_csv(ref_path)

    results: Dict[str, Any] = {}
    for col, bins in bins_map.items():
        if col not in prod.columns or col not in ref.columns:
            continue
        b = np.array(bins, dtype=float)
        exp = pd.to_numeric(ref[col], errors="coerce").to_numpy()
        act = pd.to_numeric(prod[col], errors="coerce").to_numpy()
        results[col] = _json_safe_number(compute_psi(exp, act, b))

    worst = sorted(
        [(k, v) for k, v in results.items() if v is not None],
        key=lambda x: x[1],
        reverse=True,
    )[:10] if results else []

    return {
        "n_production_samples": int(len(prod)),
        "psi": results,
        "top_drift": [{"feature": k, "psi": v} for k, v in worst],
        "guideline": {"no_drift": "<0.10", "moderate": "0.10-0.25", "significant": ">0.25"},
    }


@router.get("/explain")
def explain(student_id: str, limit: int = 10):
    """Return prediction history + latest explanation for a given student_id."""
    if not DB_PATH.exists():
        return {"message": "No prediction history yet. Call /predict first."}

    conn = _db()
    rows = conn.execute(
        """SELECT ts, risk_score, risk_class, model_version, top_factors, payload
           FROM predictions
           WHERE student_id = ?
           ORDER BY ts DESC
           LIMIT ?""",
        (student_id, int(limit)),
    ).fetchall()
    conn.close()

    if not rows:
        return {"message": "No predictions found for this student_id.", "student_id": student_id}

    def _row_to_item(r):
        ts, score, cls, ver, top_factors, payload = r
        try:
            top = json.loads(top_factors) if top_factors else []
        except Exception:
            top = []
        try:
            pay = json.loads(payload) if payload else {}
        except Exception:
            pay = {}
        return {
            "ts": int(ts),
            "risk_score": _json_safe_number(score),
            "risk_class": int(cls),
            "risk_level": "alto" if int(cls) == 1 else "baixo",
            "model_version": ver,
            "top_risk_factors": top,
            "payload": pay,
        }

    items = [_row_to_item(r) for r in rows]
    return {
        "student_id": student_id,
        "count": len(items),
        "latest": items[0],
        "history": items,
    }
