"""
WS-F statistics, used by scripts/make_numbers.py:
  iqm, iqm_ci         interquartile mean over seeds with a 95% (stratified) bootstrap interval
                      (Agarwal et al., 2021: trimmed mean cutting 25% from each end; resampling runs
                      with replacement independently within each task)
  newcombe_paired     95% interval for a difference of paired proportions, Newcombe (1998) method 10
  holm                Holm (1979) step-down adjusted p-values within one family of tests
  wilson              Wilson score interval (as experiments/stats_utils.py)

    python scripts/stats_v2.py   runs the checks against published worked examples
"""
import math

import numpy as np
from scipy import stats


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def iqm(x):
    return float(stats.trim_mean(np.asarray(x, float), 0.25))


def iqm_ci(tasks, n_boot=20000, seed=0, level=0.95):
    """tasks: list of per-task score arrays (one score per seed). One task gives a plain bootstrap over
    seeds; several give the stratified bootstrap (resampling within each task), and the statistic is
    the IQM of all resampled scores pooled, as in rliable."""
    rng = np.random.default_rng(seed)
    tasks = [np.asarray(t, float) for t in tasks]
    point = iqm(np.concatenate(tasks))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = iqm(np.concatenate([t[rng.integers(0, len(t), len(t))] for t in tasks]))
    a = (1 - level) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def newcombe_paired(a_ok, b_ok, z=1.959964):
    """Difference P(A succeeds) - P(B succeeds) over paired environments; returns (diff, lo, hi).
    Method 10 of Newcombe (1998), Stat. Med. 17:2635-2650 (Wilson intervals, phi correction)."""
    a_ok, b_ok = np.asarray(a_ok, bool), np.asarray(b_ok, bool)
    n = len(a_ok)
    a = int(np.sum(a_ok & b_ok)); b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok)); d = n - a - b - c
    return newcombe_counts(a, b, c, d, z)


def newcombe_counts(a, b, c, d, z=1.959964):
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    num = a * d - b * c
    if num > 0:                                   # method 10: continuity-corrected phi when positive
        num = max(num - n / 2, 0.0)
    phi = num / math.sqrt(den) if den > 0 else 0.0
    theta = (b - c) / n
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return theta, max(-1.0, theta - dl), min(1.0, theta + du)


def holm(pvals):
    """Holm step-down adjusted p-values, same order as the input."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k])
        adj[k] = min(1.0, run)
    return adj.tolist()


def _checks():
    # Newcombe (1998), Stat. Med. 17:2635, Table III, method 10 (e, f, g, h = a, b, c, d here)
    for cells, ref in (((36, 12, 2, 0), (0.0569, 0.3404)), ((20, 12, 2, 16), (0.0562, 0.3292)),
                       ((18, 12, 2, 18), (0.0562, 0.3290))):
        t, lo, hi = newcombe_counts(*cells)
        assert abs(lo - ref[0]) < 5e-5 and abs(hi - ref[1]) < 5e-5, (cells, lo, hi)
    # Holm: p = (0.01, 0.04, 0.03, 0.005) -> (0.03, 0.06, 0.06, 0.02)
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    # IQM of 10 seeds = mean of the middle 6 (scipy trims int(0.25 n) from each end)
    assert abs(iqm(range(10)) - 4.5) < 1e-12 and abs(iqm([0, 0, 0, 0, 0, 100, 100, 100, 100, 100]) - 50) < 1e-12
    # Wilson: 50/50 -> 0.929-1.0 (paper: 93-100%)
    lo, hi = wilson(50, 50)
    assert abs(lo - 0.9287) < 5e-4 and abs(hi - 1.0) < 1e-12
    print("stats_v2 checks passed")


if __name__ == "__main__":
    _checks()
