import os
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils import compute_psi, make_bins, ARTIFACT_DIR, DATA_DIR
from src import train as train_mod
from src.evaluate import evaluate


def test_psi_basic_behavior():
    exp = np.array([0, 0, 1, 1, 1], dtype=float)
    act = np.array([0, 1, 1, 1, 1], dtype=float)
    bins = np.array([-0.1, 0.5, 1.1], dtype=float)
    psi = compute_psi(exp, act, bins)
    assert psi >= 0.0


def test_train_and_evaluate_end_to_end(tmp_path):
    os.environ["RF_TREES"] = "5"

    df = pd.DataFrame({
        "INSTITUICAO_ENSINO_ALUNO_2020": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "IDADE_ALUNO_2020": [10, 11, 12, 13, 14, 15, 16, 17],
        "ANOS_PM_2020": [1, 2, 2, 3, 3, 4, 4, 5],
        "FASE_TURMA_2020": ["1A", "1B", "2A", "2B", "2A", "3A", "3B", "4A"],
        "PONTO_VIRADA_2020": [0, 1, 0, 1, 0, 1, 0, 1],
        "INDE_2020": [5.0, 5.5, 6.0, 6.2, 6.5, 6.8, 7.0, 7.2],
        "IEG_2020": [5.1, 5.2, 6.0, 6.1, 6.4, 6.7, 7.1, 7.3],
        "IPS_2020": [6.0, 6.1, 6.1, 6.2, 6.4, 6.6, 6.8, 6.9],
        "IDA_2020": [5.0, 5.1, 6.0, 6.1, 6.3, 6.6, 6.9, 7.0],
        "IPP_2020": [5.5, 5.6, 6.1, 6.2, 6.4, 6.5, 6.7, 6.8],
        "IPV_2020": [5.2, 5.3, 6.2, 6.2, 6.6, 6.7, 7.0, 7.1],
        "IAN_2020": [5.3, 5.4, 6.3, 6.3, 6.7, 6.8, 7.2, 7.3],
        "IAA_2020": [5.4, 5.5, 6.4, 6.4, 6.8, 6.9, 7.3, 7.4],
        "DEFASAGEM_2021": [-1, -1, 0, 0, -1, 0, -2, 1],
    })

    meta = train_mod.train(df, model_version="test_train", save_reference=True)
    assert (ARTIFACT_DIR / "model.joblib").exists()
    assert (ARTIFACT_DIR / "metadata.json").exists()
    assert (DATA_DIR / "train_reference.csv").exists()
    assert "auc" in meta["metrics"]
    xlsx = tmp_path / "mini.xlsx"
    df.to_excel(xlsx, index=False)
    out = evaluate(str(xlsx))
    assert "auc" in out


def test_build_preprocessor_simple():
    df_small = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    pre, num, cat = train_mod.build_preprocessor(df_small)
    assert "a" in num
    assert "b" in cat
    arr = pre.fit_transform(df_small)
    assert arr.shape[0] == 2
