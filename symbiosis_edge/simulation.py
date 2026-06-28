"""The Symbiosis-Edge stream simulator.

This is a faithful, de-duplicated port of the ``simulate_one_run`` logic that
previously lived (identically) in each ``scripts/*.py`` file. Numerics are
preserved exactly: the order of random draws is unchanged, so a given
``(dataset, seed, params)`` produces the same stream as before.

Model in one paragraph
----------------------
Each method maintains a scalar accuracy ``state``. At every step the state is
pulled toward a pre-/post-drift environment target (an EMA plus optional noise),
the edge model's uncertainty is computed from that state, and the routing policy
decides whether to escalate to an oracle or human. A correct annotation raises
the state by a method-specific learning rate; a wrong one lowers it. The
realised prediction is then sampled with probability equal to the current state.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .drift import SimpleADWIN
from .params import DatasetConfig, SimParams
from .routing import quantile_threshold, symbiosis_thresholds
from .uncertainty import uncertainty_score

__all__ = [
    "METHODS",
    "apply_supervision_update",
    "simulate_one_run",
    "simulate_datasets",
]

#: Methods compared by the simulator, in a fixed order (drives RNG determinism).
METHODS: Tuple[str, ...] = ("Static", "SAL", "ADWIN-SAL", "Symbiosis-Edge")


def apply_supervision_update(
    current_state: float,
    *,
    rng: np.random.Generator,
    annotator_acc: float,
    lr_correct: float,
    lr_wrong: float,
) -> Tuple[float, bool]:
    """Apply one annotation to a method's accuracy state.

    With probability ``annotator_acc`` the annotation is correct and the state
    is incremented by ``lr_correct``; otherwise it is decremented by
    ``lr_wrong``. The state is clipped to ``[0.05, 0.999]``.

    Returns the new state and whether the annotation was correct.
    """
    is_correct_label = bool(rng.random() < annotator_acc)
    if is_correct_label:
        new_state = current_state + lr_correct
    else:
        new_state = current_state - lr_wrong
    return float(np.clip(new_state, 0.05, 0.999)), is_correct_label


def simulate_one_run(*, dataset: str, seed: int, params: SimParams) -> pd.DataFrame:
    """Simulate one stream for all methods and return a long-format frame.

    Columns: ``dataset, t, method, y_true, y_pred, q_oracle, q_human,
    oracle_correct, human_correct``.
    """
    rng = np.random.default_rng(seed)
    n = params.n
    k = params.k_classes

    y_true = rng.integers(0, k, size=n)

    state = {
        "Static": float(params.pre_acc_static),
        "SAL": float(params.pre_acc_sal),
        "ADWIN-SAL": float(params.pre_acc_adwin),
        "Symbiosis-Edge": float(params.pre_acc_sym),
    }

    u_hist: Dict[str, List[float]] = {m: [] for m in state.keys()}
    rows: List[dict] = []

    adwin = SimpleADWIN(
        max_window=params.adwin_max_window,
        min_window=params.adwin_min_window,
        delta=params.adwin_delta,
    )
    adwin_alarm_until = -1
    adwin_u_ema = None
    adwin_ema_alpha = 0.12

    for t in range(n):
        is_post = t >= params.drift_t
        env_noise = params.post_noise * (1.0 if is_post else 0.0)

        env_target = {
            "Static": params.post_acc_static if is_post else params.pre_acc_static,
            "SAL": params.post_acc_sal if is_post else params.pre_acc_sal,
            "ADWIN-SAL": params.post_acc_adwin if is_post else params.pre_acc_adwin,
            "Symbiosis-Edge": params.post_acc_sym if is_post else params.pre_acc_sym,
        }

        for method in METHODS:
            p = state[method]
            p = 0.995 * p + 0.005 * env_target[method]
            p = float(np.clip(p + rng.normal(0.0, env_noise), 0.05, 0.999))
            state[method] = p

            u = uncertainty_score(p, k=k, alpha=params.alpha_margin)
            u_hist[method].append(u)

            q_oracle = False
            q_human = False
            oracle_correct = np.nan
            human_correct = np.nan

            if method == "SAL":
                window = np.array(u_hist[method][-params.window_w:], dtype=float)
                tau = quantile_threshold(window, params.b_sal)
                if u > tau:
                    q_oracle = True

            elif method == "ADWIN-SAL":
                window = np.array(u_hist[method][-params.window_w:], dtype=float)
                b_now = params.b_adwin_alarm if t <= adwin_alarm_until else params.b_adwin_base
                tau = quantile_threshold(window, b_now)
                if u > tau:
                    q_oracle = True

            elif method == "Symbiosis-Edge":
                window = np.array(u_hist[method][-params.window_w:], dtype=float)
                tau1, tau2 = symbiosis_thresholds(
                    window,
                    b_oracle=params.b_oracle,
                    b_human=params.b_human,
                )
                if u > tau2:
                    q_human = True
                elif u > tau1:
                    q_oracle = True

            pred_correct = rng.random() < state[method]
            if pred_correct:
                y_pred = int(y_true[t])
            else:
                other = [c for c in range(k) if c != int(y_true[t])]
                y_pred = int(rng.choice(other))

            if method == "ADWIN-SAL":
                if adwin_u_ema is None:
                    adwin_u_ema = u
                else:
                    adwin_u_ema = (1.0 - adwin_ema_alpha) * adwin_u_ema + adwin_ema_alpha * u

                changed = adwin.update(adwin_u_ema)
                if changed:
                    adwin_alarm_until = max(adwin_alarm_until, t + params.adwin_alarm_window)

            if method == "SAL":
                if q_oracle:
                    state[method], oc = apply_supervision_update(
                        state[method],
                        rng=rng,
                        annotator_acc=params.oracle_acc,
                        lr_correct=params.lr_oracle,
                        lr_wrong=params.lr_oracle_wrong,
                    )
                    oracle_correct = bool(oc)

            elif method == "ADWIN-SAL":
                if q_oracle:
                    lr_adwin = params.lr_oracle
                    if t <= adwin_alarm_until:
                        lr_adwin = params.lr_oracle + params.adwin_lr_boost

                    state[method], oc = apply_supervision_update(
                        state[method],
                        rng=rng,
                        annotator_acc=params.oracle_acc,
                        lr_correct=lr_adwin,
                        lr_wrong=params.lr_oracle_wrong,
                    )
                    oracle_correct = bool(oc)
                else:
                    if is_post and t <= adwin_alarm_until:
                        state[method] = float(np.clip(
                            state[method] + 0.5 * params.lr_edge,
                            0.05,
                            0.999,
                        ))

            elif method == "Symbiosis-Edge":
                if q_human:
                    state[method], hc = apply_supervision_update(
                        state[method],
                        rng=rng,
                        annotator_acc=params.human_acc,
                        lr_correct=params.lr_human,
                        lr_wrong=params.lr_human_wrong,
                    )
                    human_correct = bool(hc)
                elif q_oracle:
                    state[method], oc = apply_supervision_update(
                        state[method],
                        rng=rng,
                        annotator_acc=params.oracle_acc,
                        lr_correct=params.lr_oracle,
                        lr_wrong=params.lr_oracle_wrong,
                    )
                    oracle_correct = bool(oc)
                else:
                    if is_post and u >= params.u_floor:
                        state[method] = float(np.clip(state[method] + params.lr_edge, 0.05, 0.999))

            rows.append({
                "dataset": dataset,
                "t": t,
                "method": method,
                "y_true": int(y_true[t]),
                "y_pred": int(y_pred),
                "q_oracle": bool(q_oracle),
                "q_human": bool(q_human),
                "oracle_correct": oracle_correct,
                "human_correct": human_correct,
            })

    return pd.DataFrame(rows)


def simulate_datasets(
    *,
    datasets: Sequence[DatasetConfig],
    params_by_dataset: Dict[str, SimParams],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Run every ``(dataset, seed)`` combination and concatenate the results.

    A ``seed`` column is added so multi-seed runs can be aggregated with
    confidence intervals downstream.

    Parameters
    ----------
    datasets:
        Streams to simulate.
    params_by_dataset:
        Mapping from dataset name to its :class:`SimParams`.
    seeds:
        Random seeds; one stream is generated per seed.
    """
    frames: List[pd.DataFrame] = []
    for ds in datasets:
        if ds.name not in params_by_dataset:
            raise KeyError(f"No SimParams provided for dataset '{ds.name}'")
        params = params_by_dataset[ds.name]
        for seed in seeds:
            df = simulate_one_run(dataset=ds.name, seed=int(seed), params=params)
            df["seed"] = int(seed)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
