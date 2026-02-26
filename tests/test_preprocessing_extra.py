import pandas as pd
import numpy as np
import pytest

from src.preprocessing import (
    normalize_columns,
    build_target,
    standardize_schema,
    select_features,
    split_X_y,
)


def test_normalize_columns():
    df = pd.DataFrame({" A ": [1]})
    out = normalize_columns(df)
    assert "A" in out.columns


def test_build_target_variants():
    df = pd.DataFrame({"DEFASAGEM_2021": [-1, 0, 1]})
    y = build_target(df)
    assert list(y) == [1, 0, 0]

    df2 = pd.DataFrame({"Defas": [-0.1, 0.2]})
    y2 = build_target(df2)
    assert list(y2) == [1, 0]

    with pytest.raises(ValueError):
        build_target(pd.DataFrame({"foo": [1]}))


def test_standardize_schema_fiap():
    df = pd.DataFrame({
        "IDADE_ALUNO_2020": [10],
        "ANOS_NA_PM_2020": [2],
        "INDE_2020": [5],
        "IEG_2020": [6],
        "FASE_TURMA_2020": ["1A"],
    })
    out = standardize_schema(df)
    assert out.columns.tolist() == ["IDADE", "ANOS_NA_PM", "INDE", "IEG", "FASE_TURMA"]


def test_standardize_schema_2024():
    df = pd.DataFrame({
        "INDE 22": [5],
        "IEG": [6],
        "Fase": ["1"],
        "Turma": ["A"],
    })
    out = standardize_schema(df)
    assert "FASE_TURMA" in out.columns
    assert out.loc[0, "FASE_TURMA"] == "1-A"
    assert out.loc[0, "ANOS_NA_PM"] is pd.NA


def test_select_features_pv_mapping():
    df = pd.DataFrame({
        "PONTO_VIRADA": ["sim", "não", 1, 0],
        "IDADE": [1, 2, 3, 4],
    })
    out = select_features(df)
    assert set(out["PONTO_VIRADA"].tolist()) <= {0, 1}


def test_split_X_y_masking():
    df = pd.DataFrame({
        "IDADE": [1, 2],
        "ANOS_NA_PM": [1, 1],
        "DEFASAGEM_2021": [None, -1],
    })
    X, y = split_X_y(df)
    assert len(X) == 1
    assert len(y) == 1
