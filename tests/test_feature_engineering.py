import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import add_derived_features


def test_add_derived_standard_schema():
    df = pd.DataFrame({
        "IDADE": [10, 0],
        "ANOS_NA_PM": [2, 1],
        "PONTO_VIRADA": [0, 1],
    })
    out = add_derived_features(df)
    assert "ANOS_PM_POR_IDADE" in out.columns
    assert np.isfinite(out.loc[0, "ANOS_PM_POR_IDADE"])
    assert np.isnan(out.loc[1, "ANOS_PM_POR_IDADE"])


def test_add_derived_fiap_schema():
    df = pd.DataFrame({
        "IDADE_ALUNO_2020": [10],
        "ANOS_PM_2020": [5],
        "PONTO_VIRADA_2020": [1],
    })
    out = add_derived_features(df)
    assert "IDADE" in out.columns
    assert "ANOS_NA_PM" in out.columns
    assert "PONTO_VIRADA" in out.columns
    assert out.loc[0, "ANOS_PM_POR_IDADE"] == pytest.approx(0.5)


def test_add_derived_missing():
    df = pd.DataFrame({})
    out = add_derived_features(df)
    assert "ANOS_PM_POR_IDADE" in out.columns
    assert out.empty
