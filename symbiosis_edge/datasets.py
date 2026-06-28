"""Built-in dataset *presets*.

Each name (``SYNTHETIC``, ``SECOM``, ``APS``) selects a :class:`SimParams`
preset that reproduces the corresponding configuration from the original
``run_multi_dataset.py`` experiment. These presets parameterise the *simulation*
-- no external data files are read. The ``SECOM`` / ``APS`` names are kept for
continuity with the accompanying paper, which documents them as parametric
stream settings rather than trained-model results.

To register your own stream, build a :class:`SimParams` and add it to a copy of
:data:`DATASET_PRESETS` (or pass it through ``params_by_dataset`` directly).
"""

from __future__ import annotations

from typing import Dict, List

from .params import DatasetConfig, SimParams

__all__ = ["DATASET_PRESETS", "DRIFT_T", "get_preset", "default_datasets"]

#: Drift time shared by the built-in presets.
DRIFT_T = 500

DATASET_PRESETS: Dict[str, SimParams] = {
    "SYNTHETIC": SimParams(
        n=2000,
        drift_t=DRIFT_T,
        b_sal=0.12,
        b_adwin_base=0.10,
        b_adwin_alarm=0.30,
        b_oracle=0.12,
        b_human=0.05,
        adwin_delta=0.08,
        adwin_max_window=300,
        adwin_min_window=30,
        adwin_alarm_window=240,
        adwin_lr_boost=0.004,
        alpha_margin=0.6,
        oracle_acc=0.95,
        human_acc=1.00,
        lr_oracle=0.010,
        lr_human=0.016,
        lr_oracle_wrong=0.012,
        lr_human_wrong=0.010,
        pre_acc_static=0.93,
        post_acc_static=0.60,
        pre_acc_sal=0.93,
        post_acc_sal=0.55,
        pre_acc_adwin=0.93,
        post_acc_adwin=0.61,
        pre_acc_sym=0.93,
        post_acc_sym=0.55,
        post_noise=0.02,
    ),
    "SECOM": SimParams(
        n=2000,
        drift_t=DRIFT_T,
        b_sal=0.12,
        b_adwin_base=0.10,
        b_adwin_alarm=0.28,
        b_oracle=0.12,
        b_human=0.05,
        adwin_delta=0.08,
        adwin_max_window=300,
        adwin_min_window=30,
        adwin_alarm_window=220,
        adwin_lr_boost=0.004,
        alpha_margin=0.6,
        oracle_acc=0.95,
        human_acc=1.00,
        lr_oracle=0.010,
        lr_human=0.016,
        lr_oracle_wrong=0.012,
        lr_human_wrong=0.010,
        pre_acc_static=0.92,
        post_acc_static=0.62,
        pre_acc_sal=0.93,
        post_acc_sal=0.58,
        pre_acc_adwin=0.93,
        post_acc_adwin=0.61,
        pre_acc_sym=0.94,
        post_acc_sym=0.58,
        post_noise=0.02,
    ),
    "APS": SimParams(
        n=2000,
        drift_t=DRIFT_T,
        b_sal=0.12,
        b_adwin_base=0.10,
        b_adwin_alarm=0.28,
        b_oracle=0.12,
        b_human=0.05,
        adwin_delta=0.08,
        adwin_max_window=300,
        adwin_min_window=30,
        adwin_alarm_window=220,
        adwin_lr_boost=0.004,
        alpha_margin=0.6,
        oracle_acc=0.95,
        human_acc=1.00,
        lr_oracle=0.010,
        lr_human=0.016,
        lr_oracle_wrong=0.012,
        lr_human_wrong=0.010,
        pre_acc_static=0.90,
        post_acc_static=0.60,
        pre_acc_sal=0.92,
        post_acc_sal=0.56,
        pre_acc_adwin=0.92,
        post_acc_adwin=0.60,
        pre_acc_sym=0.93,
        post_acc_sym=0.56,
        post_noise=0.02,
    ),
}


def get_preset(name: str) -> SimParams:
    """Return the :class:`SimParams` preset for ``name`` (case-insensitive)."""
    key = name.upper()
    if key not in DATASET_PRESETS:
        raise KeyError(
            f"Unknown dataset preset '{name}'. Available: {sorted(DATASET_PRESETS)}"
        )
    return DATASET_PRESETS[key]


def default_datasets() -> List[DatasetConfig]:
    """The three built-in streams with their drift times."""
    return [
        DatasetConfig("SECOM", DRIFT_T),
        DatasetConfig("APS", DRIFT_T),
        DatasetConfig("SYNTHETIC", DRIFT_T),
    ]
