"""Budget-aware supervision routing via sliding-window quantile thresholds.

Given a window of recent uncertainty scores and a supervision *budget* (the
fraction of instances we are willing to escalate), the routing thresholds are
estimated as empirical quantiles of the window. This keeps the realised query
rate close to the budget regardless of the absolute scale of the uncertainty
distribution, which drifts over time.

Single-tier policies (``SAL``, ``ADWIN-SAL``) use one threshold and escalate to
a single annotator. Symbiosis-Edge uses two nested thresholds to split the
supervision budget between a cheaper oracle and a more expensive human expert.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["quantile_threshold", "symbiosis_thresholds"]


def quantile_threshold(values: np.ndarray, budget: float) -> float:
    """Threshold that escalates (in expectation) a ``budget`` fraction of items.

    Returns the ``1 - budget`` empirical quantile of ``values``. An empty window
    returns ``+inf`` (escalate nothing until evidence accumulates).

    Parameters
    ----------
    values:
        Recent uncertainty scores (the sliding window).
    budget:
        Target escalation fraction in ``[0, 1]``.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("inf")
    b = float(np.clip(budget, 0.0, 1.0))
    return float(np.quantile(values, 1.0 - b))


def symbiosis_thresholds(
    window_u: np.ndarray,
    *,
    b_oracle: float,
    b_human: float,
) -> Tuple[float, float]:
    """Two nested thresholds ``(tau_oracle, tau_human)`` for three-way routing.

    Items above ``tau_human`` go to the human expert; items between
    ``tau_oracle`` and ``tau_human`` go to the oracle; the rest stay on the edge.
    The human budget ``b_human`` is the most selective tier, so its threshold is
    the highest. We enforce ``tau_oracle <= tau_human`` for monotonic routing.

    Parameters
    ----------
    window_u:
        Sliding window of recent uncertainty scores.
    b_oracle:
        Fraction of items routed to the oracle tier.
    b_human:
        Fraction of items routed to the human tier.

    Returns
    -------
    (tau_oracle, tau_human)
    """
    tau_human = quantile_threshold(window_u, b_human)
    tau_oracle = quantile_threshold(window_u, b_human + b_oracle)
    if tau_oracle > tau_human:
        tau_oracle = tau_human
    return tau_oracle, tau_human
