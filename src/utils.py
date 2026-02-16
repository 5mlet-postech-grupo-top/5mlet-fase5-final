from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Project paths (relative to repo root)
DATA_DIR = Path("data")
ARTIFACT_DIR = Path("app") / "model"

DEFAULT_MODEL_VERSION = "local"

logger = logging.getLogger("pede-mlops")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_bins(s, n_bins: int = 10) -> np.ndarray:
    """Create numeric bins using quantiles; returns monotonically increasing edges."""
    arr = np.asarray(s, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(arr, qs)
    # ensure strictly increasing
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([edges[0], edges[0] + 1.0], dtype=float)
    return edges


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: np.ndarray) -> float:
    """Population Stability Index between expected and actual distributions."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    # avoid div-by-zero
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, 1)
    act_pct = np.clip(act_pct, eps, 1)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)
