from dataclasses import dataclass, field
from typing import List

import numpy as np

from symbiosis_edge.drift import SimpleADWIN


@dataclass
class _NaiveADWIN:
    """Reference O(window^2) implementation used to lock vectorised behaviour."""

    max_window: int = 300
    min_window: int = 30
    delta: float = 0.08
    values: List[float] = field(default_factory=list)

    def update(self, x: float) -> bool:
        self.values.append(float(x))
        if len(self.values) > self.max_window:
            self.values.pop(0)
        n = len(self.values)
        if n < self.min_window:
            return False
        hm = max(5, self.min_window // 2)
        for cut in range(hm, n - hm + 1):
            left = np.asarray(self.values[:cut])
            right = np.asarray(self.values[cut:])
            if left.size < hm or right.size < hm:
                continue
            gap = abs(left.mean() - right.mean())
            eps = np.sqrt(2.0 * np.log(2.0 / self.delta) * (1.0 / left.size + 1.0 / right.size))
            if gap > eps:
                self.values = self.values[len(self.values) // 2:]
                return True
        return False


def test_no_false_alarm_on_stationary_stream():
    rng = np.random.default_rng(0)
    det = SimpleADWIN()
    alarms = sum(det.update(x) for x in rng.normal(0.0, 0.5, size=600))
    assert alarms == 0


def test_detects_clear_mean_shift():
    rng = np.random.default_rng(0)
    det = SimpleADWIN()
    for x in rng.normal(0.0, 0.3, size=200):
        det.update(x)
    detected = any(det.update(x) for x in rng.normal(5.0, 0.3, size=200))
    assert detected


def test_vectorised_matches_naive_reference():
    rng = np.random.default_rng(42)
    stream = np.concatenate([
        rng.normal(0.0, 1.0, size=700),
        rng.normal(2.5, 1.0, size=700),
        rng.normal(0.0, 1.0, size=600),
    ])
    fast, slow = SimpleADWIN(), _NaiveADWIN()
    assert [fast.update(x) for x in stream] == [slow.update(x) for x in stream]
