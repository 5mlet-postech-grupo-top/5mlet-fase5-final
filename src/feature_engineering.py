from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns expected by the model.

    Supported input schemas:
      - standardized schema: IDADE, ANOS_NA_PM, PONTO_VIRADA, ...
      - FIAP-like schema: IDADE_ALUNO_2020, ANOS_NA_PM_2020, PONTO_VIRADA_2020, ...
    """
    X = X.copy()

    # back-compat mapping to standardized names
    if "IDADE" not in X.columns and "IDADE_ALUNO_2020" in X.columns:
        X["IDADE"] = X["IDADE_ALUNO_2020"]

    if "ANOS_NA_PM" not in X.columns:
        if "ANOS_NA_PM_2020" in X.columns:
            X["ANOS_NA_PM"] = X["ANOS_NA_PM_2020"]
        elif "ANOS_PM_2020" in X.columns:
            X["ANOS_NA_PM"] = X["ANOS_PM_2020"]

    if "PONTO_VIRADA" not in X.columns and "PONTO_VIRADA_2020" in X.columns:
        X["PONTO_VIRADA"] = X["PONTO_VIRADA_2020"]

    # Derived feature: years in PM / age
    if "IDADE" in X.columns and "ANOS_NA_PM" in X.columns:
        age = pd.to_numeric(X["IDADE"], errors="coerce")
        yrs = pd.to_numeric(X["ANOS_NA_PM"], errors="coerce")
        X["ANOS_PM_POR_IDADE"] = yrs / age.replace({0: np.nan})
    else:
        X["ANOS_PM_POR_IDADE"] = np.nan

    return X
