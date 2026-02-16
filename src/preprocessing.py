from __future__ import annotations

from typing import Tuple
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Target: 1 if student is behind ideal level (defasagem < 0), else 0.
    Supports:
      - FIAP dataset: DEFASAGEM_2021
      - 2024 dataset: Defas
    """
    if "DEFASAGEM_2021" in df.columns:
        y = _to_numeric(df["DEFASAGEM_2021"])
        return (y < 0).astype("int64")
    if "Defas" in df.columns:
        y = _to_numeric(df["Defas"])
        return (y < 0).astype("int64")
    raise ValueError("Target column not found. Expected DEFASAGEM_2021 or Defas.")


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize both datasets into a shared feature schema.

    Output columns:
      - IDADE, ANOS_NA_PM, PONTO_VIRADA
      - INDE, IAA, IEG, IPS, IDA, IPP (optional), IPV, IAN
      - FASE_TURMA, PEDRA, INSTITUICAO
    """
    df = normalize_columns(df)

    is_fiap = "INDE_2020" in df.columns and "IEG_2020" in df.columns
    is_2024 = "INDE 22" in df.columns and "IEG" in df.columns

    out = pd.DataFrame()

    if is_fiap:
        mapping = {
            "IDADE_ALUNO_2020": "IDADE",
            "ANOS_NA_PM_2020": "ANOS_NA_PM",
            "ANOS_PM_2020": "ANOS_NA_PM",
            "PONTO_VIRADA_2020": "PONTO_VIRADA",
            "INDE_2020": "INDE",
            "IAA_2020": "IAA",
            "IEG_2020": "IEG",
            "IPS_2020": "IPS",
            "IDA_2020": "IDA",
            "IPP_2020": "IPP",
            "IPV_2020": "IPV",
            "IAN_2020": "IAN",
            "FASE_TURMA_2020": "FASE_TURMA",
            "PEDRA_2020": "PEDRA",
            "INSTITUICAO_ENSINO_ALUNO_2020": "INSTITUICAO",
        }
        for src, dst in mapping.items():
            if src in df.columns:
                out[dst] = df[src]
        return out

    if is_2024:
        mapping = {
            "Idade 22": "IDADE",
            "Atingiu PV": "PONTO_VIRADA",
            "INDE 22": "INDE",
            "IAA": "IAA",
            "IEG": "IEG",
            "IPS": "IPS",
            "IDA": "IDA",
            "IPV": "IPV",
            "IAN": "IAN",
            "Fase": "FASE",
            "Turma": "TURMA",
            "Instituição de ensino": "INSTITUICAO",
            "Pedra 22": "PEDRA",
        }
        for src, dst in mapping.items():
            if src in df.columns:
                out[dst] = df[src]

        if "FASE" in out.columns and "TURMA" in out.columns:
            out["FASE_TURMA"] = out["FASE"].astype(str).str.strip() + "-" + out["TURMA"].astype(str).str.strip()
        elif "FASE" in out.columns:
            out["FASE_TURMA"] = out["FASE"].astype(str)
        else:
            out["FASE_TURMA"] = None

        # ANOS_NA_PM is not reliably present in 2024 schema -> keep missing
        out["ANOS_NA_PM"] = pd.NA

        return out

    # Fallback: already standardized (or unknown)
    return df.copy()


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    df_std = standardize_schema(df)

    # PV -> 0/1 numeric where possible
    if "PONTO_VIRADA" in df_std.columns:
        pv = df_std["PONTO_VIRADA"]
        if pv.dtype == "bool":
            df_std["PONTO_VIRADA"] = pv.astype("int64")
        else:
            s = pv.astype(str).str.strip().str.lower()
            df_std["PONTO_VIRADA"] = s.map({
                "true": 1, "false": 0,
                "sim": 1, "não": 0, "nao": 0,
                "1": 1, "0": 0
            }).fillna(pd.to_numeric(pv, errors="coerce"))

    keep = [
        "IDADE", "ANOS_NA_PM", "PONTO_VIRADA",
        "INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN",
        "FASE_TURMA", "PEDRA", "INSTITUICAO",
    ]
    cols = [c for c in keep if c in df_std.columns]
    return df_std[cols].copy()


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = normalize_columns(df)
    y = build_target(df)
    X = select_features(df)

    # only keep rows where target is present
    if "DEFASAGEM_2021" in df.columns:
        mask = df["DEFASAGEM_2021"].notna()
    elif "Defas" in df.columns:
        mask = df["Defas"].notna()
    else:
        mask = pd.Series([True] * len(df))

    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
