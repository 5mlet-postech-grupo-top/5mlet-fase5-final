from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


TEXT_COLUMNS = {
    "NOME",
    "DESTAQUE_IEG_2020",
    "DESTAQUE_IDA_2020",
    "DESTAQUE_IPV_2020",
    "DESTAQUE_IEG_2022",
    "DESTAQUE_IDA_2022",
    "DESTAQUE_IPV_2022",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # keep original column names, but strip spaces
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """Target: 1 if DEFASAGEM_2021 < 0 (aluno atrasado), else 0.
    This matches the dataset behavior where negative means behind ideal level.
    """
    if "DEFASAGEM_2021" not in df.columns:
        raise ValueError("Column DEFASAGEM_2021 not found.")
    y = df["DEFASAGEM_2021"]
    y = pd.to_numeric(y, errors="coerce")
    return (y < 0).astype("int64")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Use only 'past' features to avoid leakage: mostly *_2020 plus a few safe columns."""
    df = df.copy()

    # Standardize ANOS feature name (dataset uses ANOS_PM_2020)
    if "ANOS_PM_2020" in df.columns and "ANOS_NA_PM_2020" not in df.columns:
        df["ANOS_NA_PM_2020"] = df["ANOS_PM_2020"]

    # Include 2020 numeric indicators + core context columns
    cols = [c for c in df.columns if c.endswith("_2020")]
    # Remove text / label-like columns
    cols = [c for c in cols if c not in TEXT_COLUMNS and not c.endswith("CONCEITO_2020")]
    # Also include 2020 institution/turma/pedra (already endswith _2020)
    X = df[cols].copy()
    # Drop the raw name if present
    for c in list(X.columns):
        if c in TEXT_COLUMNS:
            X.drop(columns=[c], inplace=True, errors="ignore")
    return X


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = normalize_columns(df)
    y = build_target(df)
    X = select_features(df)
    # keep only rows with target present
    mask = df["DEFASAGEM_2021"].notna()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
