"""A lightweight ADWIN-style change detector.

This is a compact, dependency-free adaptive-windowing detector used by the
``ADWIN-SAL`` baseline. It keeps a bounded window of recent scalar observations
and flags a change whenever some split of the window yields sub-window means
that differ by more than a Hoeffding-style bound. On detection it drops the
older half of the window so the detector re-centres on the new regime.

It deliberately mirrors the spirit of Bifet & Gavalda's ADWIN without the exact
exponential-histogram bookkeeping, keeping the simulation fast and transparent.
The split scan is vectorised with prefix sums: it is mathematically identical to
the naive double loop (same candidate cuts, same "first admissible cut wins"
semantics) but runs in ``O(window)`` instead of ``O(window^2)`` per update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

__all__ = ["SimpleADWIN"]


@dataclass
class SimpleADWIN:
    """Adaptive-windowing change detector over a stream of scalars.

    Parameters
    ----------
    max_window:
        Maximum number of observations retained.
    min_window:
        Minimum number of observations before any test is performed.
    delta:
        Confidence parameter; smaller ``delta`` -> larger bound -> fewer alarms.
    """

    max_window: int = 300
    min_window: int = 30
    delta: float = 0.08
    values: List[float] = field(default_factory=list)

    def update(self, x: float) -> bool:
        """Add observation ``x``; return ``True`` if a change is detected."""
        self.values.append(float(x))
        if len(self.values) > self.max_window:
            self.values.pop(0)

        n = len(self.values)
        if n < self.min_window:
            return False

        half_min = max(5, self.min_window // 2)
        if half_min > n - half_min:
            return False

        arr = np.asarray(self.values, dtype=float)
        csum = np.cumsum(arr)
        total = float(csum[-1])

        # Candidate split points: `cut` = number of elements in the left window.
        cuts = np.arange(half_min, n - half_min + 1)
        left_sum = csum[cuts - 1]
        left_mean = left_sum / cuts
        right_mean = (total - left_sum) / (n - cuts)

        gap = np.abs(left_mean - right_mean)
        eps = np.sqrt(2.0 * np.log(2.0 / self.delta) * (1.0 / cuts + 1.0 / (n - cuts)))

        if np.any(gap > eps):
            self.values = self.values[n // 2:]
            return True
        return False
