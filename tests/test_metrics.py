import math

import numpy as np

from symbiosis_edge import SimParams, simulate_one_run
from symbiosis_edge.metrics import (
    CostModel,
    aguc,
    classification_metrics,
    confusion_matrix,
    mean_ci,
    method_cost,
    post_drift_summary,
)


def test_confusion_matrix_counts():
    yt = np.array([0, 0, 1, 1, 2])
    yp = np.array([0, 1, 1, 1, 2])
    cm = confusion_matrix(yt, yp, k=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 2
    assert cm[2, 2] == 1
    assert cm.sum() == len(yt)


def test_perfect_classification_metrics():
    yt = np.array([0, 1, 2, 3] * 5)
    m = classification_metrics(yt, yt.copy(), k=4)
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "mcc", "cohen_kappa"):
        assert math.isclose(m[key], 1.0, abs_tol=1e-9)


def test_chance_kappa_near_zero():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 4, size=20000)
    yp = rng.integers(0, 4, size=20000)
    m = classification_metrics(yt, yp, k=4)
    assert abs(m["cohen_kappa"]) < 0.05
    assert abs(m["mcc"]) < 0.05
    assert 0.20 < m["accuracy"] < 0.30


def test_macro_f1_known_value():
    # Binary: TP=1(class0 correct? ) compute by hand.
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])
    # class0: prec=1/1=1, rec=1/2=0.5 -> f1=2*1*0.5/1.5=0.6667
    # class1: prec=2/3, rec=2/2=1 -> f1=2*(2/3)/(5/3)=0.8
    m = classification_metrics(yt, yp, k=2)
    assert math.isclose(m["macro_f1"], (0.66666667 + 0.8) / 2, rel_tol=1e-5)


def test_method_cost_rules():
    cost = CostModel(cost_edge_step=0.0, cost_oracle=1.0, cost_human=10.0)
    assert method_cost("Static", n_steps=100, q_oracle=10, q_human=2, cost=cost) == 0.0
    # SAL/ADWIN billed at human rate on their queries.
    assert method_cost("SAL", n_steps=100, q_oracle=10, q_human=0, cost=cost) == 100.0
    assert method_cost("ADWIN-SAL", n_steps=100, q_oracle=7, q_human=0, cost=cost) == 70.0
    # Symbiosis: oracle + human (+ edge per step).
    assert method_cost("Symbiosis-Edge", n_steps=100, q_oracle=10, q_human=2, cost=cost) == 30.0


def test_aguc_formula_and_zero_cost():
    assert math.isclose(aguc(0.9, 0.6, 30.0), 0.01, rel_tol=1e-9)
    assert math.isnan(aguc(0.9, 0.6, 0.0))


def test_mean_ci_shapes_and_single_run():
    mat = np.random.default_rng(0).normal(size=(5, 40))
    mean, lo, hi = mean_ci(mat, ci=0.95)
    assert mean.shape == lo.shape == hi.shape == (40,)
    assert np.all(lo <= mean) and np.all(mean <= hi)
    # single run => degenerate band
    m1, l1, h1 = mean_ci(mat[:1], ci=0.95)
    assert np.allclose(m1, l1) and np.allclose(m1, h1)


def test_post_drift_summary_structure():
    params = SimParams(n=600, drift_t=200)
    df = simulate_one_run(dataset="SYNTHETIC", seed=0, params=params)
    summ = post_drift_summary(df, drift_t=200)
    assert set(summ["method"]) == {"Static", "SAL", "ADWIN-SAL", "Symbiosis-Edge"}
    for col in ["accuracy", "macro_f1", "mcc", "total_cost", "n_queries", "aguc"]:
        assert col in summ.columns
    # Static has zero cost and undefined AGUC.
    static = summ[summ["method"] == "Static"].iloc[0]
    assert static["total_cost"] == 0.0
    assert math.isnan(static["aguc"])
