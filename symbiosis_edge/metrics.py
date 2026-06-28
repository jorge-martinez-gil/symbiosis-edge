"""Evaluation metrics: supervision cost, AGUC, and classification quality.

Two families of metric live here:

1. **Cost / efficiency** -- the supervision cost model and *Accuracy Gain per
   Unit Cost* (AGUC), ported faithfully from the original table-generation code.
2. **Classification quality** -- accuracy, balanced accuracy, macro-F1, MCC and
   Cohen's kappa, computed directly from the simulated ``y_true``/``y_pred``
   columns with a small dependency-free numpy implementation. These are honest
   functions of the simulated predictions (no external libraries, no fabricated
   numbers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .params import Schema
from .simulation import METHODS

__all__ = [
    "CostModel",
    "method_cost",
    "aguc",
    "confusion_matrix",
    "classification_metrics",
    "post_drift_summary",
    "mean_ci",
    "summarize_runs",
]

#: Methods billed at the human supervision rate on their (single-tier) queries.
_HUMAN_BILLED = {"SAL", "ADWIN-SAL"}


@dataclass(frozen=True)
class CostModel:
    """Per-event supervision costs.

    Defaults match the original experiments: the edge model is effectively free,
    an oracle query costs 1, and a human annotation costs 10.
    """

    cost_edge_step: float = 0.0
    cost_oracle: float = 1.0
    cost_human: float = 10.0


def method_cost(
    method: str,
    *,
    n_steps: int,
    q_oracle: int,
    q_human: int,
    cost: CostModel = CostModel(),
) -> float:
    """Total supervision cost for one method over ``n_steps`` post-drift steps.

    * ``Static`` -- no supervision, cost 0.
    * ``SAL`` / ``ADWIN-SAL`` -- single-tier; their queries are billed at the
      human rate (they have no cheap oracle tier in the original costing).
    * ``Symbiosis-Edge`` -- edge cost per step + oracle queries + human queries.
    """
    if method == "Static":
        return 0.0
    if method in _HUMAN_BILLED:
        return float(cost.cost_human * q_oracle)
    return float(
        cost.cost_edge_step * n_steps + cost.cost_oracle * q_oracle + cost.cost_human * q_human
    )


def aguc(mean_acc: float, static_acc: float, total_cost: float) -> float:
    """Accuracy Gain per Unit Cost: ``(acc - static_acc) / total_cost``.

    Returns ``nan`` when ``total_cost <= 0`` (the gain is undefined per unit of
    zero cost).
    """
    if total_cost <= 0.0:
        return float("nan")
    return float((mean_acc - static_acc) / total_cost)


# --------------------------------------------------------------------------- #
# Classification quality (dependency-free)
# --------------------------------------------------------------------------- #

def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, *, k: Optional[int] = None
) -> np.ndarray:
    """Integer confusion matrix ``C`` with ``C[i, j]`` = #(true ``i``, pred ``j``)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if k is None:
        k = int(max(y_true.max(initial=-1), y_pred.max(initial=-1))) + 1
    k = max(k, 1)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, k: Optional[int] = None
) -> Dict[str, float]:
    """Accuracy, balanced accuracy, macro-F1, MCC and Cohen's kappa.

    Implemented from the confusion matrix so the package has no hard dependency
    on scikit-learn. Classes with zero support are excluded from macro averages.
    """
    cm = confusion_matrix(y_true, y_pred, k=k).astype(float)
    n = cm.sum()
    if n == 0:
        nan = float("nan")
        return {
            "accuracy": nan,
            "balanced_accuracy": nan,
            "macro_f1": nan,
            "mcc": nan,
            "cohen_kappa": nan,
        }

    tp = np.diag(cm)
    support = cm.sum(axis=1)        # true totals per class
    pred_tot = cm.sum(axis=0)       # predicted totals per class

    accuracy = float(tp.sum() / n)

    present = support > 0
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    balanced_accuracy = float(recall[present].mean()) if present.any() else float("nan")

    precision = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    macro_f1 = float(f1[present].mean()) if present.any() else float("nan")

    # Cohen's kappa
    p_o = accuracy
    p_e = float((support * pred_tot).sum() / (n * n))
    cohen_kappa = float((p_o - p_e) / (1.0 - p_e)) if (1.0 - p_e) > 1e-12 else float("nan")

    # Multiclass Matthews correlation coefficient (Gorodkin formulation):
    #   MCC = (c*s - sum_k p_k t_k) / sqrt((s^2 - sum_k p_k^2)(s^2 - sum_k t_k^2))
    # with c = #correct, s = #samples, p_k = predicted total, t_k = true total.
    c_correct = float(tp.sum())
    sum_pk2 = float((pred_tot ** 2).sum())
    sum_tk2 = float((support ** 2).sum())
    cov_ytyp = c_correct * n - float((support * pred_tot).sum())
    mcc_denom = np.sqrt((n * n - sum_pk2) * (n * n - sum_tk2))
    mcc = float(cov_ytyp / mcc_denom) if mcc_denom > 1e-12 else float("nan")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "cohen_kappa": cohen_kappa,
    }


# --------------------------------------------------------------------------- #
# Post-drift summary
# --------------------------------------------------------------------------- #

def _post_drift(df: pd.DataFrame, *, t_col: str, drift_t: Optional[int]) -> pd.DataFrame:
    if drift_t is None:
        return df
    return df[df[t_col].astype(int) >= int(drift_t)]


def _one_run_method_row(
    dm: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    schema: Schema,
    cost: CostModel,
    k: Optional[int],
) -> Dict[str, float]:
    n_steps = int(dm[schema.t].nunique())
    q_oracle = int(dm[schema.q_oracle].fillna(False).astype(bool).sum())
    q_human = int(dm[schema.q_human].fillna(False).astype(bool).sum())
    qm = classification_metrics(
        dm[schema.y_true].to_numpy(),
        dm[schema.y_pred].to_numpy(),
        k=k,
    )
    total_cost = method_cost(
        method, n_steps=n_steps, q_oracle=q_oracle, q_human=q_human, cost=cost
    )
    return {
        "dataset": dataset,
        "method": method,
        "n_steps": n_steps,
        "q_oracle": q_oracle,
        "q_human": q_human,
        "n_queries": q_oracle + q_human,
        "total_cost": total_cost,
        **qm,
    }


def post_drift_summary(
    df: pd.DataFrame,
    *,
    drift_t: Optional[int],
    schema: Schema = Schema(),
    cost: CostModel = CostModel(),
    k: Optional[int] = None,
) -> pd.DataFrame:
    """One summary row per ``(dataset, method)`` over the post-drift segment.

    If a ``seed`` column is present, per-seed rows are averaged. AGUC is computed
    against the ``Static`` accuracy *within each dataset*.

    Returns columns: ``dataset, method, n_steps, q_oracle, q_human, n_queries,
    total_cost, accuracy, balanced_accuracy, macro_f1, mcc, cohen_kappa, aguc``.
    """
    d = df.copy()
    d[schema.t] = d[schema.t].astype(int)
    d[schema.method] = d[schema.method].astype(str)
    d = _post_drift(d, t_col=schema.t, drift_t=drift_t)
    if d.empty:
        raise ValueError("No post-drift rows to summarise.")

    has_seed = "seed" in d.columns
    rows: List[Dict[str, float]] = []
    for dataset, dd in d.groupby(schema.dataset):
        for method in METHODS:
            dmeth = dd[dd[schema.method] == method]
            if dmeth.empty:
                continue
            if has_seed:
                per_seed = [
                    _one_run_method_row(
                        ds_group, dataset=str(dataset), method=method,
                        schema=schema, cost=cost, k=k,
                    )
                    for _, ds_group in dmeth.groupby("seed")
                ]
                agg = pd.DataFrame(per_seed).drop(columns=["dataset", "method"]).mean()
                row = {"dataset": str(dataset), "method": method, **agg.to_dict()}
            else:
                row = _one_run_method_row(
                    dmeth, dataset=str(dataset), method=method,
                    schema=schema, cost=cost, k=k,
                )
            rows.append(row)

    out = pd.DataFrame(rows)
    # AGUC relative to Static accuracy within each dataset.
    aguc_vals: List[float] = []
    for _, r in out.iterrows():
        static = out[(out["dataset"] == r["dataset"]) & (out["method"] == "Static")]
        static_acc = float(static["accuracy"].iloc[0]) if not static.empty else 0.0
        aguc_vals.append(aguc(float(r["accuracy"]), static_acc, float(r["total_cost"])))
    out["aguc"] = aguc_vals
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Confidence intervals for time-series bands
# --------------------------------------------------------------------------- #

def mean_ci(mat: np.ndarray, ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise mean and normal-approx CI band for a ``(runs, T)`` matrix.

    Returns ``(mean, lo, hi)`` each of length ``T``.
    """
    mat = np.asarray(mat, dtype=float)
    mean = np.nanmean(mat, axis=0)
    n = mat.shape[0]
    if n <= 1:
        return mean, mean.copy(), mean.copy()
    sd = np.nanstd(mat, axis=0, ddof=1)
    se = sd / np.sqrt(n)
    z = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}.get(round(ci, 2), 1.9599640)
    half = z * se
    return mean, mean - half, mean + half


def summarize_runs(
    df: pd.DataFrame,
    *,
    drift_t: Optional[int],
    schema: Schema = Schema(),
    cost: CostModel = CostModel(),
    k: Optional[int] = None,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Per-``(dataset, method)`` mean and CI half-width across seeds.

    Requires a ``seed`` column. Reports mean/CI for accuracy, total cost and
    AGUC -- the headline quantities -- as ``*_mean`` / ``*_ci`` pairs.
    """
    if "seed" not in df.columns:
        raise ValueError("summarize_runs requires a 'seed' column (multi-seed run).")

    d = df.copy()
    d[schema.t] = d[schema.t].astype(int)
    d = _post_drift(d, t_col=schema.t, drift_t=drift_t)

    per_seed_frames: List[pd.DataFrame] = []
    for seed, dd in d.groupby("seed"):
        s = post_drift_summary(
            dd.drop(columns=["seed"]), drift_t=None, schema=schema, cost=cost, k=k
        )
        s["seed"] = seed
        per_seed_frames.append(s)
    allruns = pd.concat(per_seed_frames, ignore_index=True)

    z = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}.get(round(ci, 2), 1.9599640)
    out_rows: List[Dict[str, float]] = []
    for (dataset, method), g in allruns.groupby(["dataset", "method"]):
        row: Dict[str, float] = {"dataset": dataset, "method": method, "n_seeds": int(len(g))}
        for col in ["accuracy", "macro_f1", "mcc", "total_cost", "aguc", "n_queries"]:
            vals = g[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_ci"] = float("nan")
            elif finite.size == 1:
                row[f"{col}_mean"] = float(finite[0])
                row[f"{col}_ci"] = 0.0
            else:
                row[f"{col}_mean"] = float(finite.mean())
                row[f"{col}_ci"] = float(z * finite.std(ddof=1) / np.sqrt(finite.size))
        out_rows.append(row)
    return pd.DataFrame(out_rows).reset_index(drop=True)
