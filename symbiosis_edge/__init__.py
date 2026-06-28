"""Symbiosis-Edge: cost-aware supervision routing under concept drift.

This package is the single, tested source of truth for the Symbiosis-Edge
simulation engine. It de-duplicates the logic that previously lived in the
standalone ``scripts/*.py`` files and exposes it as an importable library, a
command-line interface, and a reproducible experiment runner.

Scientific framing
------------------
The bundled experiments are a **parametric stream simulation**, not training of
real models on real datasets. Each method's accuracy is a state variable driven
toward configurable pre-/post-drift targets, and supervision (oracle / human)
nudges that state via configurable learning rates. The dataset names
(``SECOM``, ``APS``, ``SYNTHETIC``) select *parameter presets*; no external data
is loaded. This makes the dynamics transparent and fully reproducible. See
``docs/methodology.md`` for the precise model and its limitations.

Quick start
-----------
>>> from symbiosis_edge import SimParams, simulate_one_run, post_drift_summary
>>> df = simulate_one_run(dataset="SYNTHETIC", seed=0, params=SimParams())
>>> summary = post_drift_summary(df, drift_t=SimParams().drift_t)
>>> sorted(summary["method"])
['ADWIN-SAL', 'SAL', 'Static', 'Symbiosis-Edge']
"""

from __future__ import annotations

from .datasets import DATASET_PRESETS, default_datasets, get_preset
from .drift import SimpleADWIN
from .metrics import (
    CostModel,
    aguc,
    mean_ci,
    method_cost,
    post_drift_summary,
    summarize_runs,
)
from .params import DatasetConfig, Schema, SimParams
from .routing import quantile_threshold, symbiosis_thresholds
from .simulation import (
    METHODS,
    apply_supervision_update,
    simulate_datasets,
    simulate_one_run,
)
from .uncertainty import (
    entropy,
    margin,
    probs_from_pcorrect,
    uncertainty_score,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # params
    "SimParams",
    "DatasetConfig",
    "Schema",
    # uncertainty
    "probs_from_pcorrect",
    "entropy",
    "margin",
    "uncertainty_score",
    # routing
    "quantile_threshold",
    "symbiosis_thresholds",
    # drift
    "SimpleADWIN",
    # simulation
    "METHODS",
    "simulate_one_run",
    "simulate_datasets",
    "apply_supervision_update",
    # metrics
    "CostModel",
    "post_drift_summary",
    "method_cost",
    "aguc",
    "mean_ci",
    "summarize_runs",
    # datasets
    "DATASET_PRESETS",
    "get_preset",
    "default_datasets",
]
