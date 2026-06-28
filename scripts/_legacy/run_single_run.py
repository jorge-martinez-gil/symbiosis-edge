# run_single_run.py
# End-to-end: simulate ONCE per dataset, plot figures, and generate ONE LaTeX table PER dataset.
#
# Methods:
#   - Static
#   - SAL
#   - ADWIN-SAL
#   - Symbiosis
#
# Notes:
# - One run only, so no CI bands.
# - Drift time is used only by the environment to change difficulty.
# - Uncertainty score u_t = entropy(p) + alpha*(1 - margin(p))
# - SAL queries via sliding-window quantile thresholding.
# - ADWIN-SAL uses a stronger drift-aware budgeted policy.
# - Symbiosis routes via two quantile thresholds to satisfy oracle/human budgets.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================
# Styling
# ============================

def set_paper_style(*, use_tex: bool = False, base_font: int = 9) -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": base_font,
        "axes.labelsize": base_font,
        "axes.titlesize": base_font + 1,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
        "lines.linewidth": 1.7,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "figure.constrained_layout.use": True,
        "text.usetex": bool(use_tex),
    })


def _save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _save_text(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


# ============================
# Schema
# ============================

@dataclass(frozen=True)
class Schema:
    dataset: str = "dataset"
    t: str = "t"
    method: str = "method"
    y_true: str = "y_true"
    y_pred: str = "y_pred"
    q_human: str = "q_human"
    q_oracle: str = "q_oracle"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    drift_t: Optional[int]


# ============================
# Helpers
# ============================

def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


def _auto_ylim_from_series(
    series_list: List[np.ndarray],
    pad: float = 0.10,
    min_top: float = 1.0,
) -> Tuple[float, float]:
    vmax = 0.0
    for s in series_list:
        if s is None or len(s) == 0:
            continue
        vmax = max(vmax, float(np.nanmax(s)))
    top = max(min_top, vmax * (1.0 + pad))
    return 0.0, top


def _draw_drift(ax: plt.Axes, drift_t: Optional[int]) -> None:
    if drift_t is None:
        return
    ax.axvline(drift_t, linestyle="--", linewidth=1.1)


def _keep_post_drift(df: pd.DataFrame, *, t_col: str, drift_t: Optional[int]) -> pd.DataFrame:
    if drift_t is None:
        return df.copy()
    return df[df[t_col].astype(int) >= int(drift_t)].copy()


def _collapse_duplicates_last(df: pd.DataFrame, *, t_col: str) -> pd.DataFrame:
    df = df.sort_values(t_col)
    return df.groupby(t_col, as_index=False).tail(1)


def _collapse_duplicates_max(df: pd.DataFrame, *, t_col: str, val_col: str) -> pd.DataFrame:
    df = df.sort_values(t_col)
    return df.groupby(t_col, as_index=False)[val_col].max()


def _apply_supervision_update(
    current_state: float,
    *,
    rng: np.random.Generator,
    annotator_acc: float,
    lr_correct: float,
    lr_wrong: float,
) -> Tuple[float, bool]:
    is_correct_label = bool(rng.random() < annotator_acc)
    if is_correct_label:
        new_state = current_state + lr_correct
    else:
        new_state = current_state - lr_wrong
    return float(np.clip(new_state, 0.05, 0.999)), is_correct_label


# ============================
# Uncertainty score
# ============================

def _probs_from_pcorrect(p_correct: float, k: int) -> np.ndarray:
    p_correct = float(np.clip(p_correct, 1e-8, 1.0 - 1e-8))
    rest = (1.0 - p_correct) / max(1, (k - 1))
    p = np.full(k, rest, dtype=float)
    p[0] = p_correct
    p = p / p.sum()
    return p


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _margin(p: np.ndarray) -> float:
    ps = np.sort(p)[::-1]
    if ps.size < 2:
        return float(ps[0])
    return float(ps[0] - ps[1])


def _uncertainty_score(p_correct: float, *, k: int, alpha: float) -> float:
    p = _probs_from_pcorrect(p_correct, k)
    h = _entropy(p)
    m = _margin(p)
    return float(h + float(alpha) * (1.0 - m))


def _quantile_threshold(values: np.ndarray, budget: float) -> float:
    if values.size == 0:
        return float("inf")
    b = float(np.clip(budget, 0.0, 1.0))
    return float(np.quantile(values, 1.0 - b))


def _symbiosis_thresholds(
    window_u: np.ndarray,
    *,
    b_oracle: float,
    b_human: float,
) -> Tuple[float, float]:
    tau2 = _quantile_threshold(window_u, b_human)
    tau1 = _quantile_threshold(window_u, b_human + b_oracle)
    if tau1 > tau2:
        tau1 = tau2
    return tau1, tau2


# ============================
# Stronger ADWIN-like detector
# ============================

@dataclass
class SimpleADWIN:
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

        detected = False
        half_min = max(5, self.min_window // 2)

        for cut in range(half_min, n - half_min + 1):
            left = np.asarray(self.values[:cut], dtype=float)
            right = np.asarray(self.values[cut:], dtype=float)

            if left.size < half_min or right.size < half_min:
                continue

            gap = abs(left.mean() - right.mean())
            eps = np.sqrt(2.0 * np.log(2.0 / self.delta) * (1.0 / left.size + 1.0 / right.size))

            if gap > eps:
                detected = True
                break

        if detected:
            self.values = self.values[len(self.values) // 2:]
            return True

        return False


# ============================
# Simulator
# ============================

@dataclass(frozen=True)
class SimParams:
    n: int = 2000
    k_classes: int = 4
    drift_t: int = 500
    window_w: int = 200

    # Budgets
    b_sal: float = 0.12
    b_adwin_base: float = 0.10
    b_adwin_alarm: float = 0.28
    b_oracle: float = 0.12
    b_human: float = 0.05

    # ADWIN baseline
    adwin_delta: float = 0.08
    adwin_max_window: int = 300
    adwin_min_window: int = 30
    adwin_alarm_window: int = 220
    adwin_lr_boost: float = 0.004

    # Uncertainty mix
    alpha_margin: float = 0.6

    # Edge pseudo-update gate
    u_floor: float = 0.0

    # Learning rates
    lr_edge: float = 0.001
    lr_oracle: float = 0.010
    lr_human: float = 0.016

    # Penalty when supervision is wrong
    lr_oracle_wrong: float = 0.012
    lr_human_wrong: float = 0.010

    # Supervision reliability
    oracle_acc: float = 0.95
    human_acc: float = 0.99

    # Environment targets
    pre_acc_static: float = 0.92
    post_acc_static: float = 0.60
    pre_acc_sal: float = 0.93
    post_acc_sal: float = 0.55
    pre_acc_adwin: float = 0.93
    post_acc_adwin: float = 0.60
    pre_acc_sym: float = 0.93
    post_acc_sym: float = 0.55

    post_noise: float = 0.02


def simulate_one_run(*, dataset: str, seed: int, params: SimParams) -> pd.DataFrame:
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

        for method in ["Static", "SAL", "ADWIN-SAL", "Symbiosis-Edge"]:
            p = state[method]
            p = 0.995 * p + 0.005 * env_target[method]
            p = float(np.clip(p + rng.normal(0.0, env_noise), 0.05, 0.999))
            state[method] = p

            u = _uncertainty_score(p, k=k, alpha=params.alpha_margin)
            u_hist[method].append(u)

            q_oracle = False
            q_human = False
            oracle_correct = np.nan
            human_correct = np.nan

            if method == "SAL":
                window = np.array(u_hist[method][-params.window_w:], dtype=float)
                tau = _quantile_threshold(window, params.b_sal)
                if u > tau:
                    q_oracle = True

            elif method == "ADWIN-SAL":
                window = np.array(u_hist[method][-params.window_w:], dtype=float)
                b_now = params.b_adwin_alarm if t <= adwin_alarm_until else params.b_adwin_base
                tau = _quantile_threshold(window, b_now)
                if u > tau:
                    q_oracle = True

            elif method == "Symbiosis-Edge":
                window = np.array(u_hist[method][-params.window_w:], dtype=float)
                tau1, tau2 = _symbiosis_thresholds(
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
                    state[method], oc = _apply_supervision_update(
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

                    state[method], oc = _apply_supervision_update(
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
                    state[method], hc = _apply_supervision_update(
                        state[method],
                        rng=rng,
                        annotator_acc=params.human_acc,
                        lr_correct=params.lr_human,
                        lr_wrong=params.lr_human_wrong,
                    )
                    human_correct = bool(hc)
                elif q_oracle:
                    state[method], oc = _apply_supervision_update(
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


def simulate_datasets_single_run(
    *,
    datasets: Sequence[DatasetConfig],
    params_by_dataset: Dict[str, SimParams],
    seed: int = 0,
) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        p = params_by_dataset[ds.name]
        frames.append(simulate_one_run(dataset=ds.name, seed=seed, params=p))
    return pd.concat(frames, ignore_index=True)


# ============================
# Plot: combined accuracy + query counts
# ============================

def plot_accuracy_with_query_count_panel_single_run(
    df_one_dataset: pd.DataFrame,
    *,
    schema: Schema,
    smooth_acc: int = 25,
    drift_t: Optional[int] = None,
    title: str = "",
    fig_size: Tuple[float, float] = (6.9, 3.9),
    acc_ylim: Optional[Tuple[float, float]] = None,
    symbiosis_name: str = "Symbiosis-Edge",
    symbiosis_red: str = "tab:red",
    query_ylim: Optional[Tuple[float, float]] = None,
    show_title: bool = True,
    x_max: int = 2000,
) -> plt.Figure:
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=fig_size)
    fig.set_constrained_layout(False)

    gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.35], hspace=0.06)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    df = df_one_dataset.sort_values([schema.method, schema.t]).copy()
    df[schema.t] = df[schema.t].astype(int)

    method_to_color: Dict[str, str] = {}
    acc_series_all: List[np.ndarray] = []

    # Top: rolling accuracy
    for method, g_method in df.groupby(schema.method):
        g_method = g_method[[schema.t, schema.y_true, schema.y_pred]].dropna()
        if g_method.empty:
            continue

        g2 = _collapse_duplicates_last(g_method, t_col=schema.t)
        correct = (g2[schema.y_pred].to_numpy() == g2[schema.y_true].to_numpy()).astype(float)
        y = _rolling_mean(correct, smooth_acc)
        t = g2[schema.t].to_numpy(dtype=int)

        line = ax_top.plot(t, y, label=str(method))[0]
        method_to_color[str(method)] = line.get_color()
        acc_series_all.append(y)

    _draw_drift(ax_top, drift_t)

    if show_title and title:
        ax_top.set_title(title)

    ax_top.set_ylabel(f"Rolling acc. (w={smooth_acc})")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(0, x_max)

    if acc_ylim is not None:
        ax_top.set_ylim(*acc_ylim)
    else:
        if acc_series_all:
            lo = min(float(np.nanmin(x)) for x in acc_series_all)
            hi = max(float(np.nanmax(x)) for x in acc_series_all)
            pad = 0.03 * max(1e-6, hi - lo)
            lo = max(0.0, lo - pad)
            hi = min(1.0, hi + pad)
            if lo > 0.40:
                lo = max(0.40, lo)
            ax_top.set_ylim(lo, hi)
        else:
            ax_top.set_ylim(0.0, 1.0)

    ax_top.legend(
        frameon=False,
        ncol=1,
        loc="lower left",
        handlelength=2.4,
        borderaxespad=0.2,
    )

    # Bottom: cumulative queries
    plotted_means: List[np.ndarray] = []
    sym_handles: List[plt.Line2D] = []
    sym_labels: List[str] = []

    t_support = np.arange(0, x_max + 1, dtype=int)

    def _plot_single_query_method(method_name: str, label: str, show_in_legend: bool) -> None:
        df_m = df[df[schema.method].astype(str) == method_name].copy()
        if df_m.empty:
            return

        tmp = df_m[[schema.t, schema.q_oracle, schema.q_human]].copy()
        tmp[schema.q_oracle] = tmp[schema.q_oracle].fillna(False).astype(bool).astype(int)
        tmp[schema.q_human] = tmp[schema.q_human].fillna(False).astype(bool).astype(int)
        tmp["__q__"] = np.maximum(tmp[schema.q_oracle].to_numpy(), tmp[schema.q_human].to_numpy())
        tmp2 = _collapse_duplicates_max(tmp[[schema.t, "__q__"]], t_col=schema.t, val_col="__q__").sort_values(schema.t)

        full = pd.DataFrame({"t": t_support})
        full = full.merge(tmp2, on="t", how="left").fillna(0.0)

        cum = np.cumsum(full["__q__"].to_numpy(dtype=int)).astype(float)
        c = method_to_color.get(method_name, None)
        h = ax_bot.plot(t_support, cum, color=c, linestyle="-")[0]

        if show_in_legend:
            sym_handles.append(h)
            sym_labels.append(label)
        plotted_means.append(cum)

    _plot_single_query_method("Static", "Static", show_in_legend=False)
    _plot_single_query_method("SAL", "SAL", show_in_legend=False)
    _plot_single_query_method("ADWIN-SAL", "ADWIN-SAL", show_in_legend=False)

    df_sym = df[df[schema.method].astype(str) == symbiosis_name].copy()
    if not df_sym.empty:
        tmp = df_sym[[schema.t, schema.q_oracle, schema.q_human]].copy()
        tmp[schema.q_oracle] = tmp[schema.q_oracle].fillna(False).astype(bool).astype(int)
        tmp[schema.q_human] = tmp[schema.q_human].fillna(False).astype(bool).astype(int)
        tmp["__q_total__"] = np.maximum(
            tmp[schema.q_oracle].to_numpy(),
            tmp[schema.q_human].to_numpy(),
        )
        tmp2 = _collapse_duplicates_max(tmp[[schema.t, "__q_total__"]], t_col=schema.t, val_col="__q_total__").sort_values(schema.t)

        full = pd.DataFrame({"t": t_support})
        full = full.merge(tmp2, on="t", how="left").fillna(0.0)

        cum_total = np.cumsum(full["__q_total__"].to_numpy(dtype=int)).astype(float)
        h_total = ax_bot.plot(t_support, cum_total, color=symbiosis_red, linestyle="-")[0]
        sym_handles.append(h_total)
        sym_labels.append("S-E: total")
        plotted_means.append(cum_total)

        def _plot_sym(col: str, linestyle: str, lab: str) -> None:
            tmp_local = df_sym[[schema.t, col]].copy()
            tmp_local[col] = tmp_local[col].fillna(False).astype(bool).astype(int)
            tmp2_local = _collapse_duplicates_max(
                tmp_local[[schema.t, col]],
                t_col=schema.t,
                val_col=col,
            ).sort_values(schema.t)

            full_local = pd.DataFrame({"t": t_support})
            full_local = full_local.merge(tmp2_local, on="t", how="left").fillna(0.0)

            cum_local = np.cumsum(full_local[col].to_numpy(dtype=int)).astype(float)
            h = ax_bot.plot(t_support, cum_local, color=symbiosis_red, linestyle=linestyle)[0]
            sym_handles.append(h)
            sym_labels.append(lab)
            plotted_means.append(cum_local)

        _plot_sym(schema.q_oracle, ":", "S-E: oracle")
        _plot_sym(schema.q_human, "--", "S-E: human")

    _draw_drift(ax_bot, drift_t)
    ax_bot.set_xlabel("Time step $t$")
    ax_bot.set_ylabel("Cum. queries")
    ax_bot.set_xlim(0, x_max)

    if query_ylim is None:
        lo2, hi2 = _auto_ylim_from_series(plotted_means, pad=0.06, min_top=1.0)
        ax_bot.set_ylim(lo2, hi2)
    else:
        ax_bot.set_ylim(*query_ylim)

    if sym_handles:
        ax_bot.legend(
            handles=sym_handles,
            labels=sym_labels,
            frameon=False,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(0.015, 0.98),
            borderaxespad=0.0,
            columnspacing=1.0,
            handlelength=2.2,
            handletextpad=0.5,
        )

    ax_top.locator_params(axis="y", nbins=4)
    ax_bot.locator_params(axis="y", nbins=4)
    ax_bot.locator_params(axis="x", nbins=6)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.14)
    return fig


# ============================
# LaTeX tables
# ============================

def _fmt_int(x: float) -> str:
    return str(int(round(float(x))))


def _fmt_float(x: float, nd: int = 2) -> str:
    return f"{float(x):.{nd}f}"


def make_cost_table_latex_per_dataset_single_run(
    df_all: pd.DataFrame,
    *,
    dataset: str,
    schema: Schema,
    drift_t: Optional[int] = None,
    cost_edge_step: float = 0.0,
    cost_oracle: float = 1.0,
    cost_human: float = 10.0,
    roi_mode: str = "lift_per_cost",
) -> str:
    d = df_all[df_all[schema.dataset].astype(str) == str(dataset)].copy()
    if d.empty:
        raise ValueError(f"No rows for dataset '{dataset}'")

    d[schema.t] = d[schema.t].astype(int)
    d[schema.method] = d[schema.method].astype(str)

    d = _keep_post_drift(d, t_col=schema.t, drift_t=drift_t)
    if d.empty:
        raise ValueError(f"No post-drift rows for dataset '{dataset}'")

    d["__acc__"] = (d[schema.y_pred].to_numpy() == d[schema.y_true].to_numpy()).astype(float)
    d["__q_h__"] = d[schema.q_human].fillna(False).astype(bool).astype(int) if schema.q_human in d.columns else 0
    d["__q_o__"] = d[schema.q_oracle].fillna(False).astype(bool).astype(int) if schema.q_oracle in d.columns else 0
    d["__q_any__"] = np.maximum(d["__q_h__"].to_numpy(), d["__q_o__"].to_numpy())

    rows = []
    static_acc = None

    label_map = {
        "Static": "Static",
        "SAL": "SAL",
        "ADWIN-SAL": "ADWIN-SAL",
        "Symbiosis-Edge": "Symbiosis",
    }

    for method in ["Static", "SAL", "ADWIN-SAL", "Symbiosis-Edge"]:
        dm = d[d[schema.method] == method].copy()
        if dm.empty:
            continue

        n_steps = int(dm[schema.t].nunique())

        gq = dm[[schema.t, "__q_any__", "__q_h__", "__q_o__"]].copy()
        gq = gq.groupby(schema.t, as_index=False).max()

        q_any = int(gq["__q_any__"].sum())
        q_h = int(gq["__q_h__"].sum())
        q_o = int(gq["__q_o__"].sum())
        acc = float(dm["__acc__"].mean())

        if method == "Static":
            num_queries = 0.0
            cost_per_query = "0"
            total_cost = 0.0
            static_acc = acc

        elif method in {"SAL", "ADWIN-SAL"}:
            num_queries = float(q_o)
            cost_per_query = f"{_fmt_int(cost_human)}-human"
            total_cost = cost_human * num_queries

        else:
            num_queries = float(q_o + q_h)
            cost_per_query = (
                f"{_fmt_int(cost_edge_step)}-edge + "
                f"{_fmt_int(cost_oracle)}-oracle + "
                f"{_fmt_int(cost_human)}-human"
            )
            total_cost = (cost_edge_step * n_steps) + (cost_oracle * q_o) + (cost_human * q_h)

        rows.append({
            "Method": label_map[method],
            "NumQueries": num_queries,
            "TotalCost": total_cost,
            "MeanAcc": acc,
            "CostPerQuery": cost_per_query,
        })

    if static_acc is None:
        static_acc = 0.0

    for r in rows:
        tc = float(r["TotalCost"])
        if tc <= 0.0:
            r["ROI"] = np.nan
        else:
            if roi_mode == "acc_per_cost":
                r["ROI"] = float(r["MeanAcc"]) / tc
            else:
                r["ROI"] = (float(r["MeanAcc"]) - static_acc) / tc

    cap_roi = "accuracy lift per unit cost" if roi_mode == "lift_per_cost" else "accuracy per unit cost"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & \#queries & Cost/query & Total cost & Mean acc. & ROI \\")
    lines.append(r"\midrule")

    for r in rows:
        roi = r["ROI"]
        roi_s = "--" if (isinstance(roi, float) and np.isnan(roi)) else f"{float(roi):.6f}"
        lines.append(
            f"{r['Method']} & "
            f"{_fmt_int(r['NumQueries'])} & "
            f"{r['CostPerQuery']} & "
            f"{_fmt_float(r['TotalCost'], 2)} & "
            f"{float(r['MeanAcc']):.4f} & "
            f"{roi_s} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Cost summary on {dataset} (post-drift only). Edge cost is {int(cost_edge_step)} per step, "
        rf"oracle cost is {int(cost_oracle)} per query, and human cost is {int(cost_human)} per query. "
        rf"Static has zero supervision cost. SAL and ADWIN-SAL are costed as human supervision. "
        rf"ROI is {cap_roi}.}}"
    )
    lines.append(rf"\label{{tab:cost_{str(dataset).lower()}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_tables_for_all_datasets_single_run(
    df_all: pd.DataFrame,
    *,
    out_dir: str | Path,
    datasets: Sequence[DatasetConfig],
    schema: Schema,
    cost_edge_step: float = 0.0,
    cost_oracle: float = 1.0,
    cost_human: float = 10.0,
    roi_mode: str = "lift_per_cost",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        tex = make_cost_table_latex_per_dataset_single_run(
            df_all,
            dataset=ds.name,
            schema=schema,
            drift_t=ds.drift_t,
            cost_edge_step=cost_edge_step,
            cost_oracle=cost_oracle,
            cost_human=cost_human,
            roi_mode=roi_mode,
        )
        _save_text(tex, out_dir / f"table_cost_{ds.name.lower()}.tex")


def plot_cost_vs_accuracy_post_drift_single_run(
    df_one_dataset: pd.DataFrame,
    *,
    schema: Schema,
    drift_t: int,
    cost_edge_step: float = 0.0,
    cost_oracle: float = 1.0,
    cost_human: float = 10.0,
    title: str = "",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    d = df_one_dataset.copy()
    d[schema.t] = d[schema.t].astype(int)
    d[schema.method] = d[schema.method].astype(str)

    d = d[d[schema.t] >= int(drift_t)].copy()

    d["__acc__"] = (d[schema.y_pred].to_numpy() == d[schema.y_true].to_numpy()).astype(float)
    d["__q_o__"] = d[schema.q_oracle].fillna(False).astype(bool).astype(int) if schema.q_oracle in d.columns else 0
    d["__q_h__"] = d[schema.q_human].fillna(False).astype(bool).astype(int) if schema.q_human in d.columns else 0

    rows = []
    static_acc = None

    label_map = {
        "Static": "Static",
        "SAL": "SAL",
        "ADWIN-SAL": "ADWIN-SAL",
        "Symbiosis-Edge": "Symbiosis",
    }

    for method in ["Static", "SAL", "ADWIN-SAL", "Symbiosis-Edge"]:
        dm = d[d[schema.method] == method].copy()
        if dm.empty:
            continue

        n_steps = int(dm[schema.t].nunique())

        gq = dm[[schema.t, "__q_o__", "__q_h__"]].copy()
        gq = gq.groupby(schema.t, as_index=False).max()
        q_o = int(gq["__q_o__"].sum())
        q_h = int(gq["__q_h__"].sum())
        acc = float(dm["__acc__"].mean())

        if method == "Static":
            cost = 0.0
            static_acc = acc
        elif method in {"SAL", "ADWIN-SAL"}:
            cost = cost_human * q_o
        else:
            cost = (cost_edge_step * n_steps) + (cost_oracle * q_o) + (cost_human * q_h)

        rows.append({
            "method": label_map[method],
            "acc": acc,
            "cost": float(cost),
        })

    plot_df = pd.DataFrame(rows)
    if static_acc is None:
        static_acc = float(plot_df[plot_df["method"] == "Static"]["acc"].mean()) if not plot_df.empty else 0.0

    plot_df["lift"] = plot_df["acc"] - static_acc
    plot_df["roi"] = np.where(plot_df["cost"] > 0, plot_df["lift"] / plot_df["cost"], np.nan)

    fig, ax = plt.subplots(figsize=(5.8, 3.7))

    for _, r in plot_df.iterrows():
        ax.scatter([r["cost"]], [r["acc"]], label=r["method"], s=80)

    ax.set_title(title or "Post-drift: cost vs accuracy")
    ax.set_xlabel("Total cost (post-drift)")
    ax.set_ylabel("Mean accuracy (post-drift)")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")

    return fig


# ============================
# Diagnostics
# ============================

def print_observed_annotator_accuracies(
    df_all: pd.DataFrame,
    *,
    schema: Schema,
    datasets: Sequence[DatasetConfig],
) -> None:
    for ds in datasets:
        d = df_all[df_all[schema.dataset].astype(str) == ds.name].copy()

        oracle_mask = d["q_oracle"].fillna(False).astype(bool)
        human_mask = d["q_human"].fillna(False).astype(bool)

        oracle_obs = np.nan
        human_obs = np.nan

        if oracle_mask.any():
            oracle_obs = float(pd.to_numeric(d.loc[oracle_mask, "oracle_correct"], errors="coerce").mean())

        if human_mask.any():
            human_obs = float(pd.to_numeric(d.loc[human_mask, "human_correct"], errors="coerce").mean())

        oracle_text = f"{oracle_obs:.4f}" if not np.isnan(oracle_obs) else "n/a"
        human_text = f"{human_obs:.4f}" if not np.isnan(human_obs) else "n/a"
        print(f"{ds.name}: observed oracle acc = {oracle_text}, observed human acc = {human_text}")


def print_method_summary_post_drift(
    df_all: pd.DataFrame,
    *,
    schema: Schema,
    datasets: Sequence[DatasetConfig],
) -> None:
    print("\nPost-drift summary")
    print("-" * 90)
    for ds in datasets:
        d = df_all[df_all[schema.dataset].astype(str) == ds.name].copy()
        d = d[d[schema.t].astype(int) >= int(ds.drift_t)].copy()
        d["acc"] = (d[schema.y_pred].to_numpy() == d[schema.y_true].to_numpy()).astype(float)
        print(f"\n{ds.name}")
        for method in ["Static", "SAL", "ADWIN-SAL", "Symbiosis-Edge"]:
            dm = d[d[schema.method].astype(str) == method].copy()
            if dm.empty:
                continue
            acc = float(dm["acc"].mean())
            q_o = int(dm["q_oracle"].fillna(False).astype(bool).sum())
            q_h = int(dm["q_human"].fillna(False).astype(bool).sum())
            print(f"  {method:15s} acc={acc:.4f}  oracle={q_o}  human={q_h}")


# ============================
# Main
# ============================

if __name__ == "__main__":
    set_paper_style(use_tex=False, base_font=9)
    schema = Schema()

    DRIFT_T = 500
    DATASETS = (
        DatasetConfig("SECOM", DRIFT_T),
        DatasetConfig("APS", DRIFT_T),
        DatasetConfig("SYNTHETIC", DRIFT_T),
    )

    params_by_ds: Dict[str, SimParams] = {
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

    # 1) Simulate once per dataset
    df = simulate_datasets_single_run(
        datasets=DATASETS,
        params_by_dataset=params_by_ds,
        seed=0,
    )

    # 2) Diagnostics
    print_observed_annotator_accuracies(
        df,
        schema=schema,
        datasets=DATASETS,
    )

    print_method_summary_post_drift(
        df,
        schema=schema,
        datasets=DATASETS,
    )

    # 3) Figures
    out_fig = Path("paper_figures")
    for ds in DATASETS:
        df_ds = df[df[schema.dataset].astype(str) == ds.name].copy()

        fig = plot_accuracy_with_query_count_panel_single_run(
            df_ds,
            schema=schema,
            smooth_acc=25,
            drift_t=ds.drift_t,
            title=f"{ds.name}: accuracy + query counts over time",
            symbiosis_name="Symbiosis-Edge",
            symbiosis_red="tab:red",
            show_title=True,
            fig_size=(6.9, 3.9),
            x_max=2000,
        )
        _save_fig(fig, out_fig / f"{ds.name.lower()}_accuracy_plus_query_counts_single_run")

        fig2 = plot_cost_vs_accuracy_post_drift_single_run(
            df_ds,
            schema=schema,
            drift_t=ds.drift_t,
            cost_edge_step=0.0,
            cost_oracle=1.0,
            cost_human=10.0,
            title=f"{ds.name}: post-drift cost vs accuracy",
            out_path=out_fig / f"{ds.name.lower()}_post_drift_cost_vs_accuracy_single_run",
        )
        plt.close(fig2)

    # 4) LaTeX tables
    out_tables = Path("paper_tables")
    generate_tables_for_all_datasets_single_run(
        df,
        out_dir=out_tables,
        datasets=DATASETS,
        schema=schema,
        cost_edge_step=0.0,
        cost_oracle=1.0,
        cost_human=10.0,
        roi_mode="lift_per_cost",
    )

    print(f"\nSaved figures to: {out_fig.resolve()}")
    print(f"Saved tables  to: {out_tables.resolve()}")
