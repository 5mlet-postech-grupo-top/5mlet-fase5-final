import os
from pathlib import Path
import numpy as np
import pandas as pd

import pytest

from src.utils import save_json, load_json, make_bins, compute_psi


def test_save_and_load_json(tmp_path):
    data = {"foo": 1}
    path = tmp_path / "sub" / "test.json"
    save_json(path, data)
    assert path.exists()
    loaded = load_json(path)
    assert loaded == data


def test_make_bins_edges():
    arr = np.array([1, 2, 3, 4])
    edges = make_bins(arr, n_bins=3)
    assert edges.shape[0] >= 2

    edges2 = make_bins(np.array([]))
    assert np.allclose(edges2, [0.0, 1.0])

    edges3 = make_bins(np.array([5, 5, 5]))
    assert edges3[0] != edges3[-1]


def test_compute_psi_empty():
    assert np.isnan(compute_psi(np.array([]), np.array([]), np.array([0,1])))


def test_compute_psi_normal():
    exp = np.array([0.1, 0.2, 0.3])
    act = np.array([0.1, 0.25, 0.4])
    bins = np.array([0, 0.2, 0.4, 1])
    psi = compute_psi(exp, act, bins)
    assert psi >= 0
