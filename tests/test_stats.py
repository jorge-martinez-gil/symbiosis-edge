import math

import numpy as np
import pytest

from symbiosis_edge import SimParams, simulate_datasets
from symbiosis_edge.metrics import per_seed_summary
from symbiosis_edge.params import DatasetConfig
from symbiosis_edge.stats import (
    bootstrap_ci,
    cliffs_delta,
    compare_methods,
    holm_bonferroni,
    mean_ci_t,
    paired_cohens_d,
    paired_permutation_test,
    pareto_frontier,
    t_critical,
)

# --------------------------------------------------------------------------- #
# t critical values / CIs
# --------------------------------------------------------------------------- #


def test_t_critical_known_values():
    assert t_critical(4, 0.95) == pytest.approx(2.7764, abs=1e-3)
    assert t_critical(9, 0.95) == pytest.approx(2.2622, abs=1e-3)
    assert t_critical(4, 0.99) == pytest.approx(4.6041, abs=1e-3)
    # large df falls back to the normal quantile
    assert t_critical(1000, 0.95) == pytest.approx(1.96, abs=1e-2)


def test_t_critical_rejects_bad_inputs():
    with pytest.raises(ValueError):
        t_critical(0, 0.95)
    with pytest.raises(ValueError):
        t_critical(5, 0.80)


def test_mean_ci_t_wider_than_z_for_small_n():
    vals = [0.7, 0.72, 0.68, 0.74, 0.71]
    mean, hw = mean_ci_t(vals, 0.95)
    assert mean == pytest.approx(np.mean(vals))
    se = np.std(vals, ddof=1) / math.sqrt(len(vals))
    assert hw == pytest.approx(2.7764 * se, rel=1e-3)
    assert hw > 1.96 * se  # strictly wider than the old z interval


def test_mean_ci_t_edge_cases():
    m, hw = mean_ci_t([0.5], 0.95)
    assert m == 0.5 and hw == 0.0
    m, hw = mean_ci_t([], 0.95)
    assert math.isnan(m) and math.isnan(hw)
    m, hw = mean_ci_t([np.nan, 0.4, 0.6], 0.95)
    assert m == pytest.approx(0.5)


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(0.7, 0.05, size=30)
    mean, lo, hi = bootstrap_ci(v, ci=0.95, n_boot=2000, seed=1)
    assert lo < mean < hi
    assert lo < 0.7 < hi  # generous: true mean should be inside
    # deterministic given the seed
    assert bootstrap_ci(v, ci=0.95, n_boot=2000, seed=1) == (mean, lo, hi)


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #


def test_permutation_exact_p_for_identical_samples():
    a = [0.5, 0.6, 0.7, 0.8]
    p = paired_permutation_test(a, a)
    assert p == pytest.approx(1.0)


def test_permutation_detects_consistent_difference():
    a = [0.80, 0.82, 0.81, 0.83, 0.79, 0.84, 0.80, 0.82]
    b = [0.60, 0.63, 0.61, 0.62, 0.59, 0.64, 0.61, 0.60]
    p = paired_permutation_test(a, b)
    # exact test with 8 pairs: smallest achievable two-sided p is 2/256
    assert p == pytest.approx(2 / 256, abs=1e-9)


def test_permutation_exact_matches_enumeration_semantics():
    # 3 pairs, all diffs positive: two-sided p = 2/8
    a = [1.0, 2.0, 3.0]
    b = [0.0, 1.0, 2.0]
    p = paired_permutation_test(a, b)
    assert p == pytest.approx(2 / 8)


def test_permutation_one_sided():
    a = [1.0, 2.0, 3.0]
    b = [0.0, 1.0, 2.0]
    assert paired_permutation_test(a, b, alternative="greater") == pytest.approx(1 / 8)
    assert paired_permutation_test(a, b, alternative="less") == pytest.approx(1.0)


def test_permutation_monte_carlo_never_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.01, size=20)  # n > 14 -> Monte Carlo path
    b = rng.normal(0.0, 0.01, size=20)
    p = paired_permutation_test(a, b, n_permutations=500, seed=3)
    assert 0.0 < p <= 1 / 501 + 1e-9


def test_permutation_no_signal_is_insignificant():
    rng = np.random.default_rng(42)
    a = rng.normal(0.5, 0.1, size=10)
    b = rng.normal(0.5, 0.1, size=10)
    assert paired_permutation_test(a, b) > 0.05


# --------------------------------------------------------------------------- #
# Effect sizes and Holm correction
# --------------------------------------------------------------------------- #


def test_paired_cohens_d():
    a = np.array([1.0, 1.1, 0.9, 1.05])
    b = a - 0.5
    d = paired_cohens_d(a, b)  # constant diff -> infinite d_z
    assert math.isinf(d) and d > 0
    assert paired_cohens_d(a, a) == 0.0


def test_cliffs_delta_bounds_and_sign():
    assert cliffs_delta([2, 3, 4], [0, 1]) == pytest.approx(1.0)
    assert cliffs_delta([0, 1], [2, 3, 4]) == pytest.approx(-1.0)
    assert cliffs_delta([1, 2], [1, 2]) == pytest.approx(0.0)


def test_holm_bonferroni_known_example():
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone: 0.03, 0.06, 0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)  # raised to keep monotonicity


def test_holm_bonferroni_handles_nan_and_caps_at_one():
    adj = holm_bonferroni([0.9, float("nan"), 0.8])
    assert math.isnan(adj[1])
    assert all(p <= 1.0 for p in adj if not math.isnan(p))


# --------------------------------------------------------------------------- #
# Pareto frontier
# --------------------------------------------------------------------------- #


def test_pareto_frontier_simple():
    cost = [0.0, 10.0, 5.0, 10.0]
    qual = [0.6, 0.9, 0.9, 0.7]
    mask = pareto_frontier(cost, qual)
    # (0,0.6) cheapest, (5,0.9) dominates (10,0.9); (10,0.7) dominated
    assert mask.tolist() == [True, False, True, False]


def test_pareto_frontier_keeps_duplicates_and_drops_nan():
    mask = pareto_frontier([1.0, 1.0, float("nan")], [0.5, 0.5, 0.9])
    assert mask.tolist() == [True, True, False]


# --------------------------------------------------------------------------- #
# End-to-end: compare_methods on a real (small) simulation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_per_seed():
    params = SimParams(n=400, drift_t=200)
    raw = simulate_datasets(
        datasets=[DatasetConfig("SYNTHETIC", 200)],
        params_by_dataset={"SYNTHETIC": params},
        seeds=range(5),
    )
    return per_seed_summary(raw, drift_t=200)


def test_per_seed_summary_shape(small_per_seed):
    # 4 methods x 5 seeds
    assert small_per_seed.shape[0] == 20
    assert {"dataset", "method", "seed", "accuracy", "total_cost"} <= set(
        small_per_seed.columns
    )


def test_compare_methods_structure_and_validity(small_per_seed):
    out = compare_methods(small_per_seed, n_permutations=500, seed=0)
    # 3 baselines x 3 metrics
    assert out.shape[0] == 9
    assert set(out["baseline"]) == {"Static", "SAL", "ADWIN-SAL"}
    assert ((out["p_value"] >= 0) & (out["p_value"] <= 1)).all()
    assert (out["p_holm"] >= out["p_value"] - 1e-12).all()
    assert (out["n_pairs"] == 5).all()
    # diff must equal the difference of the reported means
    np.testing.assert_allclose(
        out["diff_mean"], out["mean_target"] - out["mean_baseline"], atol=1e-9
    )


def test_compare_methods_rejects_missing_target(small_per_seed):
    with pytest.raises(ValueError):
        compare_methods(small_per_seed, target="NoSuchMethod")
