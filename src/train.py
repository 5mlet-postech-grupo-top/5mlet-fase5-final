from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .preprocessing import split_X_y
from .feature_engineering import add_derived_features
from .utils import ARTIFACT_DIR, DATA_DIR, DEFAULT_MODEL_VERSION, logger, make_bins, save_json


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, numeric, categorical


def train(df: pd.DataFrame, model_version: str, save_reference: bool = True) -> Dict:
    X, y = split_X_y(df)
    X = add_derived_features(X)

    # Ensure object columns are consistently strings (avoid mixed types in imputers)
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = X[c].astype(str)

    pre, numeric, categorical = build_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=int(os.getenv('RF_TREES','400')),
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    clf = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_val, proba)),
        "recall": float(recall_score(y_val, pred)),
        "f1": float(f1_score(y_val, pred)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "positive_rate": float(y.mean()),
    }

    drift_bins = {}
    for col in numeric:
        try:
            drift_bins[col] = make_bins(pd.to_numeric(X_train[col], errors="coerce")).tolist()
        except Exception:
            continue

    if save_reference:
        DATA_DIR.mkdir(exist_ok=True)
        ref_cols = list(dict.fromkeys(list(X_train.columns)))  # stable order
        X_train[ref_cols].to_csv(DATA_DIR / "train_reference.csv", index=False)

    metadata = {
        "model_version": model_version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "DEFASAGEM_2021 < 0",
        "features": {
            "numeric": numeric,
            "categorical": categorical,
            "derived": [c for c in X.columns if c not in numeric + categorical],
        },
        "metrics": metrics,
        "threshold": 0.5,
        "drift_bins": drift_bins,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ARTIFACT_DIR / "model.joblib")
    joblib.dump(clf.named_steps["preprocessor"], ARTIFACT_DIR / "preprocessor.joblib")
    save_json(ARTIFACT_DIR / "metadata.json", metadata)

    logger.info("training_complete", extra={"metrics": metrics, "model_version": model_version})
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to PEDE_PASSOS xlsx")
    parser.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--no-save-reference", action="store_true", help="Do not save training reference parquet")
    args = parser.parse_args()

    df = pd.read_excel(args.data)
    train(df, args.model_version, save_reference=not args.no_save_reference)


if __name__ == "__main__":
    main()
