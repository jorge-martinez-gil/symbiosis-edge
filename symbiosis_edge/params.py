"""Configuration dataclasses for the Symbiosis-Edge simulation.

``SimParams`` is the full set of knobs for one stream. ``Schema`` names the
columns of the long-format result frame. ``DatasetConfig`` binds a dataset name
to its drift time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["SimParams", "Schema", "DatasetConfig"]


@dataclass(frozen=True)
class Schema:
    """Column names of the long-format simulation result frame."""

    dataset: str = "dataset"
    t: str = "t"
    method: str = "method"
    y_true: str = "y_true"
    y_pred: str = "y_pred"
    q_human: str = "q_human"
    q_oracle: str = "q_oracle"


@dataclass(frozen=True)
class DatasetConfig:
    """A named stream and the time step at which drift occurs."""

    name: str
    drift_t: Optional[int]


@dataclass(frozen=True)
class SimParams:
    """All parameters governing one simulated stream.

    The defaults reproduce the ``SYNTHETIC`` preset. Dataset-specific presets
    live in :mod:`symbiosis_edge.datasets`.
    """

    # Stream geometry
    n: int = 2000
    k_classes: int = 4
    drift_t: int = 500
    window_w: int = 200

    # Supervision budgets (fraction of items escalated)
    b_sal: float = 0.12
    b_adwin_base: float = 0.10
    b_adwin_alarm: float = 0.28
    b_oracle: float = 0.12
    b_human: float = 0.05

    # ADWIN-SAL detector
    adwin_delta: float = 0.08
    adwin_max_window: int = 300
    adwin_min_window: int = 30
    adwin_alarm_window: int = 220
    adwin_lr_boost: float = 0.004

    # Uncertainty mix: weight on the (1 - margin) term
    alpha_margin: float = 0.6

    # Edge pseudo-update gate
    u_floor: float = 0.0

    # Learning rates (state increments after a correct annotation)
    lr_edge: float = 0.001
    lr_oracle: float = 0.010
    lr_human: float = 0.016

    # Penalty applied when supervision is wrong
    lr_oracle_wrong: float = 0.012
    lr_human_wrong: float = 0.010

    # Supervision reliability (annotator accuracy)
    oracle_acc: float = 0.95
    human_acc: float = 0.99

    # Environment accuracy targets (pre-/post-drift, per method)
    pre_acc_static: float = 0.92
    post_acc_static: float = 0.60
    pre_acc_sal: float = 0.93
    post_acc_sal: float = 0.55
    pre_acc_adwin: float = 0.93
    post_acc_adwin: float = 0.60
    pre_acc_sym: float = 0.93
    post_acc_sym: float = 0.55

    # Per-step environment noise (post-drift only)
    post_noise: float = 0.02
