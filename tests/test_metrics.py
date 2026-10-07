import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import ks_statistic, classification_metrics
from psi import population_stability_index


def test_ks_is_between_zero_and_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    ks = ks_statistic(y, s)
    assert 0 <= ks <= 1


def test_gini_matches_auc_definition():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, s)
    assert abs(m["gini"] - (2 * m["roc_auc"] - 1)) < 1e-12


def test_psi_identical_distributions_is_small():
    x = np.linspace(0.01, 0.99, 1000)
    psi = population_stability_index(x, x.copy(), bins=10)
    assert psi < 1e-10
