\
import numpy as np


def population_stability_index(expected, actual, bins=10, eps=1e-6):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    cuts = np.unique(cuts)

    if len(cuts) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual, bins=cuts)

    exp_pct = np.maximum(exp_counts / max(exp_counts.sum(), 1), eps)
    act_pct = np.maximum(act_counts / max(act_counts.sum(), 1), eps)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)
