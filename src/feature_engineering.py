from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Simple derived feature: years in PM / age
    age_col = "IDADE_ALUNO_2020"
    years_col = "ANOS_NA_PM_2020"
    if age_col in X.columns and years_col in X.columns:
        age = pd.to_numeric(X[age_col], errors="coerce")
        years = pd.to_numeric(X[years_col], errors="coerce")
        X["ANOS_PM_POR_IDADE_2020"] = years / age.replace({0: np.nan})

    return X
