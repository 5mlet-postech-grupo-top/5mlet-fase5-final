from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pythonjsonlogger import jsonlogger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "app" / "model"
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_VERSION = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_utc")


def get_logger(name: str = "pede-ml") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = get_logger()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: np.ndarray) -> float:
    """Population Stability Index (PSI). Lower is better.
    bins must be monotonically increasing; includes edges.
    """
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    # avoid zeros
    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    psi = np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc))
    return float(psi)


def make_bins(series: pd.Series, n_bins: int = 10) -> np.ndarray:
    x = series.dropna().astype(float)
    if x.empty:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(x, qs))
    if bins.size < 2:
        # constant feature
        v = float(x.iloc[0])
        bins = np.array([v - 1e-6, v + 1e-6])
    return bins.astype(float)
