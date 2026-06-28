import numpy as np

from symbiosis_edge.routing import quantile_threshold, symbiosis_thresholds


def test_empty_window_returns_inf():
    assert quantile_threshold(np.array([]), 0.2) == float("inf")


def test_budget_extremes():
    vals = np.arange(100.0)
    # budget 0 => threshold at the max (escalate ~nothing)
    assert quantile_threshold(vals, 0.0) == 99.0
    # budget 1 => threshold at the min (escalate ~everything)
    assert quantile_threshold(vals, 1.0) == 0.0


def test_realized_escalation_rate_matches_budget():
    rng = np.random.default_rng(0)
    window = rng.normal(size=5000)
    for b in (0.05, 0.1, 0.25, 0.5):
        tau = quantile_threshold(window, b)
        realized = float(np.mean(window > tau))
        assert abs(realized - b) < 0.02


def test_symbiosis_thresholds_are_nested():
    rng = np.random.default_rng(1)
    window = rng.normal(size=2000)
    for b_o, b_h in [(0.12, 0.05), (0.2, 0.1), (0.3, 0.02)]:
        tau_oracle, tau_human = symbiosis_thresholds(window, b_oracle=b_o, b_human=b_h)
        # Human tier is the most selective => highest threshold.
        assert tau_oracle <= tau_human
        # Combined oracle+human escalation rate is close to b_o + b_h.
        realized = float(np.mean(window > tau_oracle))
        assert abs(realized - (b_o + b_h)) < 0.03
