from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from .preprocessing import split_X_y
from .feature_engineering import add_derived_features
from .utils import ARTIFACT_DIR


def evaluate(data_path: str) -> Dict:
    df = pd.read_excel(data_path)
    X, y = split_X_y(df)
    X = add_derived_features(X)

    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    proba = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, proba))
    pred = (proba >= 0.5).astype(int)

    report = classification_report(y, pred, output_dict=True)
    out = {"auc": auc, "report": report}
    (ARTIFACT_DIR / "evaluation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
