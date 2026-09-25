"""Small statistics helpers for the paired APF-vs-ARC comparisons."""
import math

import numpy as np
from scipy import stats


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(a_ok, b_ok):
    """Exact (binomial) McNemar test on paired binary outcomes.
    Returns (n_a_only, n_b_only, two-sided p)."""
    a_ok, b_ok = np.asarray(a_ok, bool), np.asarray(b_ok, bool)
    a_only = int(np.sum(a_ok & ~b_ok))
    b_only = int(np.sum(~a_ok & b_ok))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return a_only, b_only, p


def wilcoxon_paired(x, y):
    """Two-sided Wilcoxon signed-rank test on paired samples; returns (n, statistic, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, y):
        return len(x), float("nan"), float("nan")
    r = stats.wilcoxon(x, y)
    return len(x), float(r.statistic), float(r.pvalue)


def bootstrap_mean_ci(x, n_boot=10000, seed=0):
    """Percentile bootstrap 95% CI of the mean."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))
