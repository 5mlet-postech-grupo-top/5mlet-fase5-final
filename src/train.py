from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
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
from .data_loader import load_all_training_data
from .utils import ARTIFACT_DIR, DATA_DIR, DEFAULT_MODEL_VERSION, logger, make_bins, save_json
from .preprocessing import split_X_y, enforce_types

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
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
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def train(df: pd.DataFrame, model_version: str, save_reference: bool = True) -> Dict:
    X, y = split_X_y(df)
    X = add_derived_features(X)

    X = enforce_types(X)

    # 1. Definição explícita (Schema Enforcement)
    categorical = ["FASE_TURMA", "PEDRA", "INSTITUICAO"]
    # Garante que só consideramos as colunas categóricas que realmente existem no df
    categorical = [c for c in categorical if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    # 2. Tratamento rigoroso de variáveis numéricas
    for col in numeric:
        if X[col].dtype == "object" or X[col].dtype.name == "string":
            X[col] = X[col].astype(str).str.replace(',', '.', regex=False)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 3. Tratamento rigoroso de variáveis categóricas
    for col in categorical:
        X[col] = X[col].astype(str)

    # 4. Passamos as listas explicitamente para o preprocessor
    pre = build_preprocessor(numeric, categorical)

    model = RandomForestClassifier(
        n_estimators=int(os.getenv("RF_TREES", "400")),
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

    THRESHOLD = 0.35
    proba = clf.predict_proba(X_val)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

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

    feature_order = list(X_train.columns)

    if save_reference:
        DATA_DIR.mkdir(exist_ok=True)
        X_train[feature_order].to_csv(DATA_DIR / "train_reference.csv", index=False)

    metadata = {
        "model_version": model_version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "defasagem < 0 (behind ideal level)",
        "feature_order": feature_order,
        "features": {
            "numeric": numeric,
            "categorical": categorical,
            "derived": ["ANOS_PM_POR_IDADE"],
        },
        "metrics": metrics,
        "threshold": THRESHOLD,
        "drift_bins": drift_bins,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ARTIFACT_DIR / "model.joblib")
    save_json(ARTIFACT_DIR / "metadata.json", metadata)

    logger.info("training_complete", extra={"metrics": metrics, "model_version": model_version})
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--no-save-reference", action="store_true")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Default: ./data")
    args = parser.parse_args()

    df = load_all_training_data(Path(args.data_dir))
    train(df, args.model_version, save_reference=not args.no_save_reference)


if __name__ == "__main__":
    main()
