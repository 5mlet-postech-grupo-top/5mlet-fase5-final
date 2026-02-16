import pandas as pd
from src.feature_engineering import add_derived_features


def test_add_derived_features_creates_ratio():
    df = pd.DataFrame([{"IDADE": 10, "ANOS_NA_PM": 2}])
    out = add_derived_features(df)
    assert "ANOS_PM_POR_IDADE" in out.columns
    assert abs(out.loc[0, "ANOS_PM_POR_IDADE"] - 0.2) < 1e-9
