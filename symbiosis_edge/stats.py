"""Statistical rigor toolkit: t-based CIs, paired tests, effect sizes, Pareto.

Everything here is dependency-free (numpy only) and deterministic given a seed.

Contents
--------
* **Confidence intervals** -- Student-t intervals (:func:`mean_ci_t`), which are
  correct for the small seed counts typical of benchmark runs (a normal
  z-interval is ~40% too narrow at n=5), and percentile-bootstrap intervals
  (:func:`bootstrap_ci`) that make no distributional assumption.
* **Paired significance tests** -- :func:`paired_permutation_test`, an exact
  sign-flip permutation test on per-seed paired differences (Monte-Carlo
  approximated for large n). Distribution-free and valid at any sample size.
* **Effect sizes** -- :func:`paired_cohens_d` (d_z on paired differences) and
  :func:`cliffs_delta` (ordinal, robust).
* **Multiple comparisons** -- :func:`holm_bonferroni` step-down correction.
* **Method comparison** -- :func:`compare_methods` runs the full battery
  (p-value, Holm-adjusted p, effect size, CI on the difference) for a target
  method against every baseline, per dataset and metric.
* **Pareto analysis** -- :func:`pareto_frontier` extracts the non-dominated
  set of the cost-quality trade-off (minimise cost, maximise quality).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "t_critical",
    "mean_ci_t",
    "bootstrap_ci",
    "paired_permutation_test",
    "paired_cohens_d",
    "cliffs_delta",
    "holm_bonferroni",
    "compare_methods",
    "pareto_frontier",
]

# --------------------------------------------------------------------------- #
# Student-t confidence intervals
# --------------------------------------------------------------------------- #

#: Two-sided critical values t_{df, 1-alpha/2} for common confidence levels.
#: Exact to 4 decimals for df 1..30; beyond 30 we fall back to the normal
#: quantile, where the relative error is < 2%.
_T_TABLE: Dict[float, Tuple[float, ...]] = {
    0.90: (6.3138, 2.9200, 2.3534, 2.1318, 2.0150, 1.9432, 1.8946, 1.8595,
           1.8331, 1.8125, 1.7959, 1.7823, 1.7709, 1.7613, 1.7531, 1.7459,
           1.7396, 1.7341, 1.7291, 1.7247, 1.7207, 1.7171, 1.7139, 1.7109,
           1.7081, 1.7056, 1.7033, 1.7011, 1.6991, 1.6973),
    0.95: (12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060,
           2.2622, 2.2281, 2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199,
           2.1098, 2.1009, 2.0930, 2.0860, 2.0796, 2.0739, 2.0687, 2.0639,
           2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423),
    0.99: (63.6567, 9.9248, 5.8409, 4.6041, 4.0321, 3.7074, 3.4995, 3.3554,
           3.2498, 3.1693, 3.1058, 3.0545, 3.0123, 2.9768, 2.9467, 2.9208,
           2.8982, 2.8784, 2.8609, 2.8453, 2.8314, 2.8188, 2.8073, 2.7969,
           2.7874, 2.7787, 2.7707, 2.7633, 2.7564, 2.7500),
}

_Z_FALLBACK: Dict[float, float] = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def t_critical(df: int, ci: float = 0.95) -> float:
    """Two-sided Student-t critical value ``t_{df, 1-(1-ci)/2}``.

    Supports ``ci`` in {0.90, 0.95, 0.99}. For ``df > 30`` the normal quantile
    is used (error < 2%).
    """
    level = round(float(ci), 2)
    if level not in _T_TABLE:
        raise ValueError(f"ci must be one of {sorted(_T_TABLE)}, got {ci}")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    table = _T_TABLE[level]
    if df <= len(table):
        return table[df - 1]
    return _Z_FALLBACK[level]


def mean_ci_t(values: Sequence[float], ci: float = 0.95) -> Tuple[float, float]:
    """Mean and Student-t CI half-width of ``values`` (NaNs dropped).

    Returns ``(mean, half_width)``; half-width is 0 for fewer than two finite
    values. Unlike a z-interval, this is calibrated for small samples --
    the difference matters: at n=5 seeds, t_4 = 2.776 vs z = 1.960.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    se = float(v.std(ddof=1) / np.sqrt(v.size))
    return float(v.mean()), t_critical(v.size - 1, ci) * se


def bootstrap_ci(
    values: Sequence[float],
    *,
    ci: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile-bootstrap CI for the mean: ``(mean, lo, hi)``.

    Distribution-free companion to :func:`mean_ci_t`; useful when the per-seed
    metric distribution is skewed (e.g. AGUC).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = v[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return float(v.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Paired permutation test and effect sizes
# --------------------------------------------------------------------------- #

def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> float:
    """Sign-flip permutation test on paired samples; returns the p-value.

    Tests H0: the paired differences ``a - b`` are symmetric about zero, using
    the mean difference as the statistic. For ``n <= 14`` pairs, all ``2^n``
    sign assignments are enumerated (an *exact* test); otherwise
    ``n_permutations`` random sign flips are drawn and the p-value includes
    the observed statistic (add-one correction), so it is never 0.

    ``alternative`` is ``"two-sided"``, ``"greater"`` (mean(a) > mean(b)) or
    ``"less"``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    d = a - b
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return float("nan")
    observed = d.mean()

    if n <= 14:  # exact enumeration: at most 16384 sign patterns
        signs = np.array(
            [[1 if (m >> i) & 1 else -1 for i in range(n)] for m in range(2 ** n)],
            dtype=float,
        )
        null = (signs * d).mean(axis=1)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
        null = (signs * d).mean(axis=1)
        exact = False

    tol = 1e-12
    if alternative == "two-sided":
        hits = np.sum(np.abs(null) >= abs(observed) - tol)
    elif alternative == "greater":
        hits = np.sum(null >= observed - tol)
    elif alternative == "less":
        hits = np.sum(null <= observed + tol)
    else:
        raise ValueError(f"unknown alternative {alternative!r}")

    if exact:
        return float(hits / null.size)
    return float((hits + 1) / (null.size + 1))


def paired_cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d_z: mean of paired differences over their SD.

    Conventional magnitude labels: 0.2 small, 0.5 medium, 0.8 large. Returns
    ``inf`` (signed) when all differences are identical and non-zero, ``nan``
    when there are fewer than two finite pairs.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd < 1e-15:
        m = d.mean()
        if abs(m) < 1e-15:
            return 0.0
        return float(np.sign(m) * np.inf)
    return float(d.mean() / sd)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta: P(a > b) - P(a < b) over all cross pairs, in [-1, 1].

    An ordinal, outlier-robust effect size. |delta| < 0.147 negligible,
    < 0.33 small, < 0.474 medium, else large (Romano et al., 2006).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (controls family-wise error rate).

    NaN inputs propagate as NaN and do not count toward the family size.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite_idx = np.where(np.isfinite(p))[0]
    m = finite_idx.size
    if m == 0:
        return out.tolist()
    order = finite_idx[np.argsort(p[finite_idx])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, adj)
        out[idx] = running_max
    return out.tolist()


# --------------------------------------------------------------------------- #
# Full method comparison battery
# --------------------------------------------------------------------------- #

#: Metrics where larger is better; anything else (e.g. total_cost) is
#: "smaller is better" purely for the human-readable ``better`` column.
_HIGHER_IS_BETTER = {
    "accuracy", "balanced_accuracy", "macro_f1", "mcc", "cohen_kappa", "aguc",
}


def compare_methods(
    per_seed: pd.DataFrame,
    *,
    target: str = "Symbiosis-Edge",
    metrics: Sequence[str] = ("accuracy", "macro_f1", "total_cost"),
    ci: float = 0.95,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare ``target`` against every other method, per dataset and metric.

    ``per_seed`` must contain columns ``dataset, method, seed`` plus each
    metric (one row per (dataset, method, seed) -- see
    :func:`symbiosis_edge.metrics.per_seed_summary`). Pairing is by seed.

    Returns one row per (dataset, metric, baseline) with:

    * ``mean_target`` / ``mean_baseline`` -- across-seed means,
    * ``diff_mean`` / ``diff_ci`` -- mean paired difference and its Student-t
      CI half-width,
    * ``p_value`` -- exact/MC paired permutation test (two-sided),
    * ``p_holm`` -- Holm-adjusted within each (dataset, metric) family,
    * ``cohens_dz`` / ``cliffs_delta`` -- effect sizes,
    * ``better`` -- whether the target improves on the baseline in the
      metric's natural direction.
    """
    required = {"dataset", "method", "seed"}
    missing = required - set(per_seed.columns)
    if missing:
        raise ValueError(f"per_seed is missing columns: {sorted(missing)}")
    if target not in set(per_seed["method"]):
        raise ValueError(f"target method {target!r} not present in per_seed")

    rows: List[Dict[str, object]] = []
    for dataset, dd in per_seed.groupby("dataset"):
        pivot = {
            m: g.sort_values("seed") for m, g in dd.groupby("method")
        }
        tgt = pivot[target]
        baselines = [m for m in pivot if m != target]
        for metric in metrics:
            family: List[Dict[str, object]] = []
            for base in baselines:
                bl = pivot[base]
                merged = pd.merge(
                    tgt[["seed", metric]], bl[["seed", metric]],
                    on="seed", suffixes=("_t", "_b"),
                )
                va = merged[f"{metric}_t"].to_numpy(dtype=float)
                vb = merged[f"{metric}_b"].to_numpy(dtype=float)
                diff_mean, diff_hw = mean_ci_t(va - vb, ci)
                p = paired_permutation_test(
                    va, vb, n_permutations=n_permutations, seed=seed,
                )
                higher = metric in _HIGHER_IS_BETTER
                improves = diff_mean > 0 if higher else diff_mean < 0
                family.append({
                    "dataset": str(dataset),
                    "metric": metric,
                    "target": target,
                    "baseline": base,
                    "n_pairs": int(merged.shape[0]),
                    "mean_target": float(np.nanmean(va)) if va.size else float("nan"),
                    "mean_baseline": float(np.nanmean(vb)) if vb.size else float("nan"),
                    "diff_mean": diff_mean,
                    "diff_ci": diff_hw,
                    "p_value": p,
                    "cohens_dz": paired_cohens_d(va, vb),
                    "cliffs_delta": cliffs_delta(va, vb),
                    "better": bool(improves),
                })
            adj = holm_bonferroni([r["p_value"] for r in family])
            for r, pa in zip(family, adj):
                r["p_holm"] = pa
            rows.extend(family)

    out = pd.DataFrame(rows)
    col_order = [
        "dataset", "metric", "target", "baseline", "n_pairs",
        "mean_target", "mean_baseline", "diff_mean", "diff_ci",
        "p_value", "p_holm", "cohens_dz", "cliffs_delta", "better",
    ]
    return out[col_order].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Pareto frontier of the cost-quality trade-off
# --------------------------------------------------------------------------- #

def pareto_frontier(
    cost: Sequence[float],
    quality: Sequence[float],
    *,
    eps: float = 0.0,
) -> np.ndarray:
    """Boolean mask of Pareto-optimal points (minimise cost, maximise quality).

    A point is dominated if another point has cost <= its cost *and*
    quality >= its quality, with at least one strict inequality (by more than
    ``eps``). Duplicate points are all kept on the frontier.
    """
    c = np.asarray(cost, dtype=float)
    q = np.asarray(quality, dtype=float)
    if c.shape != q.shape:
        raise ValueError("cost and quality must have equal length")
    n = c.size
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not (np.isfinite(c[i]) and np.isfinite(q[i])):
            mask[i] = False
            continue
        no_worse = (c <= c[i] + eps) & (q >= q[i] - eps)
        strictly_better = (c < c[i] - eps) | (q > q[i] + eps)
        if np.any(no_worse & strictly_better):
            mask[i] = False
    return mask
