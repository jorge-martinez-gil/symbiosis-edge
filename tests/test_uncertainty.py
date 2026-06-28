import math

import numpy as np
import pytest

from symbiosis_edge.uncertainty import (
    entropy,
    margin,
    probs_from_pcorrect,
    uncertainty_score,
)


def test_probs_sum_to_one_and_place_mass_on_correct_class():
    for k in (2, 3, 4, 7):
        p = probs_from_pcorrect(0.8, k)
        assert p.shape == (k,)
        assert math.isclose(p.sum(), 1.0, abs_tol=1e-9)
        assert np.argmax(p) == 0
        assert math.isclose(p[0], 0.8, abs_tol=1e-9)


def test_probs_clip_extremes():
    p = probs_from_pcorrect(1.5, 4)  # clipped below 1
    assert math.isclose(p.sum(), 1.0, abs_tol=1e-9)
    assert p[0] < 1.0


def test_probs_invalid_k():
    with pytest.raises(ValueError):
        probs_from_pcorrect(0.5, 0)


def test_entropy_bounds():
    k = 4
    uniform = np.full(k, 1.0 / k)
    assert math.isclose(entropy(uniform), math.log(k), abs_tol=1e-9)
    onehot = np.array([1.0, 0.0, 0.0, 0.0])
    assert entropy(onehot) < 1e-6


def test_margin_known_values():
    assert math.isclose(margin(np.array([0.7, 0.2, 0.1])), 0.5, abs_tol=1e-9)
    assert math.isclose(margin(np.array([0.25, 0.25, 0.25, 0.25])), 0.0, abs_tol=1e-9)
    assert math.isclose(margin(np.array([0.9])), 0.9, abs_tol=1e-9)


def test_uncertainty_decreases_with_confidence():
    scores = [uncertainty_score(pc, k=4, alpha=0.6) for pc in (0.3, 0.5, 0.7, 0.9, 0.99)]
    # More confident (higher p_correct) => strictly lower uncertainty.
    assert all(earlier > later for earlier, later in zip(scores, scores[1:]))
