"""Uncertainty scoring for the edge model.

The edge model's confidence on an instance is summarised by a scalar
``p_correct`` (the probability it labels the instance correctly). From this we
build a calibrated-looking probability vector and derive an uncertainty score
that combines Shannon entropy with the top-two margin:

.. math::

    u(x) = H(p) + \\alpha \\, (1 - \\mathrm{margin}(p))

Higher ``u`` means the model is less certain and the instance is a stronger
candidate for supervision (oracle or human).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "probs_from_pcorrect",
    "entropy",
    "margin",
    "uncertainty_score",
]


def probs_from_pcorrect(p_correct: float, k: int) -> np.ndarray:
    """Build a length-``k`` probability vector from a scalar correctness prob.

    The correct class receives mass ``p_correct``; the remaining mass is spread
    uniformly across the other ``k - 1`` classes. The result is renormalised to
    guard against floating-point drift.

    Parameters
    ----------
    p_correct:
        Probability assigned to the (assumed) correct class, clipped to
        ``(0, 1)``.
    k:
        Number of classes (``k >= 1``).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    p_correct = float(np.clip(p_correct, 1e-8, 1.0 - 1e-8))
    rest = (1.0 - p_correct) / max(1, (k - 1))
    p = np.full(k, rest, dtype=float)
    p[0] = p_correct
    p = p / p.sum()
    return p


def entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) of a probability vector."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def margin(p: np.ndarray) -> float:
    """Top-two probability margin (``p[(1)] - p[(2)]``).

    For a single-class vector the margin degenerates to the lone probability.
    """
    ps = np.sort(np.asarray(p, dtype=float))[::-1]
    if ps.size < 2:
        return float(ps[0])
    return float(ps[0] - ps[1])


def uncertainty_score(p_correct: float, *, k: int, alpha: float) -> float:
    """Combined entropy + margin uncertainty score.

    Parameters
    ----------
    p_correct:
        Scalar correctness probability of the edge model.
    k:
        Number of classes.
    alpha:
        Weight on the ``(1 - margin)`` term relative to entropy.
    """
    p = probs_from_pcorrect(p_correct, k)
    h = entropy(p)
    m = margin(p)
    return float(h + float(alpha) * (1.0 - m))
