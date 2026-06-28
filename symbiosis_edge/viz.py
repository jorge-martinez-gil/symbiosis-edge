"""Publication-quality figures and LaTeX/CSV tables.

All figures are rendered through a non-interactive Agg backend so they work in
headless CI. Every plotting helper returns the Matplotlib figure and, if an
output path is given, writes both ``.pdf`` (vector, for papers) and ``.png``
(raster, for the README and the web).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import CostModel  # noqa: E402
from .params import Schema  # noqa: E402
from .simulation import METHODS  # noqa: E402

__all__ = [
    "set_paper_style",
    "save_fig",
    "plot_accuracy_over_time",
    "plot_cost_vs_accuracy",
    "latex_cost_table",
    "summary_to_csv",
]

#: Stable colour per method so figures are visually consistent across datasets.
METHOD_COLORS: Dict[str, str] = {
    "Static": "#6c757d",
    "SAL": "#1f77b4",
    "ADWIN-SAL": "#ff7f0e",
    "Symbiosis-Edge": "#2E7D32",
}


def set_paper_style(*, use_tex: bool = False, base_font: int = 9) -> None:
    """Apply a compact, paper-friendly Matplotlib style."""
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


def save_fig(fig: plt.Figure, out_base: Path) -> List[Path]:
    """Save ``fig`` as both PDF and PNG at ``out_base`` (without suffix)."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in (".pdf", ".png"):
        p = out_base.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


def plot_accuracy_over_time(
    df_one_dataset: pd.DataFrame,
    *,
    schema: Schema = Schema(),
    drift_t: Optional[int] = None,
    smooth: int = 25,
    title: str = "",
    out_base: Optional[Path] = None,
) -> plt.Figure:
    """Rolling accuracy per method over the stream, with mean +/- CI across seeds.

    A dashed vertical line marks the drift point. If a ``seed`` column is
    present, the band shows the across-seed 95% normal-approx interval.
    """
    d = df_one_dataset.copy()
    d[schema.t] = d[schema.t].astype(int)
    d["__correct__"] = (d[schema.y_pred].to_numpy() == d[schema.y_true].to_numpy()).astype(float)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for method in METHODS:
        dm = d[d[schema.method] == method]
        if dm.empty:
            continue
        if "seed" in dm.columns:
            curves = []
            for _, g in dm.groupby("seed"):
                g = g.sort_values(schema.t)
                curves.append(_rolling_mean(g["__correct__"].to_numpy(), smooth))
            T = min(len(c) for c in curves)
            mat = np.vstack([c[:T] for c in curves])
            mean = mat.mean(axis=0)
            t_axis = np.sort(dm[schema.t].unique())[:T]
            ax.plot(t_axis, mean, label=method, color=METHOD_COLORS.get(method))
            if mat.shape[0] > 1:
                se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
                ax.fill_between(t_axis, mean - 1.96 * se, mean + 1.96 * se,
                                alpha=0.16, linewidth=0, color=METHOD_COLORS.get(method))
        else:
            g = dm.sort_values(schema.t)
            ax.plot(g[schema.t].to_numpy(), _rolling_mean(g["__correct__"].to_numpy(), smooth),
                    label=method, color=METHOD_COLORS.get(method))

    if drift_t is not None:
        ax.axvline(drift_t, linestyle="--", linewidth=1.1, color="#b00020")
        ax.annotate("drift", xy=(drift_t, ax.get_ylim()[0]), xytext=(4, 4),
                    textcoords="offset points", color="#b00020", fontsize=7)

    ax.set_title(title or "Accuracy over time")
    ax.set_xlabel("Stream step $t$")
    ax.set_ylabel(f"Rolling accuracy (w={smooth})")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    if out_base is not None:
        save_fig(fig, out_base)
    return fig


def plot_cost_vs_accuracy(
    summary: pd.DataFrame,
    *,
    dataset: Optional[str] = None,
    title: str = "",
    out_base: Optional[Path] = None,
) -> plt.Figure:
    """Cost vs post-drift accuracy scatter (the cost-quality trade-off).

    Expects the output of :func:`symbiosis_edge.metrics.post_drift_summary`.
    """
    d = summary if dataset is None else summary[summary["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for _, r in d.iterrows():
        ax.scatter([r["total_cost"]], [r["accuracy"]], s=90,
                   color=METHOD_COLORS.get(r["method"], "#333"), zorder=3)
        ax.annotate(str(r["method"]), xy=(r["total_cost"], r["accuracy"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.set_title(title or "Post-drift cost vs accuracy")
    ax.set_xlabel("Total supervision cost (post-drift)")
    ax.set_ylabel("Mean accuracy (post-drift)")
    ax.grid(True, alpha=0.22)
    if out_base is not None:
        save_fig(fig, out_base)
    return fig


def _fmt_int(x: float) -> str:
    return str(int(round(float(x))))


def latex_cost_table(
    summary: pd.DataFrame,
    *,
    dataset: str,
    cost: CostModel = CostModel(),
) -> str:
    """Render a per-dataset LaTeX cost/quality table from a summary frame."""
    d = summary[summary["dataset"] == dataset]
    if d.empty:
        raise ValueError(f"No summary rows for dataset '{dataset}'")

    label_map = {"Symbiosis-Edge": "Symbiosis"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & \#queries & Total cost & Mean acc. & Macro-F1 & AGUC \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = d[d["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        aguc_s = "--" if not np.isfinite(r["aguc"]) else f"{float(r['aguc']):.6f}"
        lines.append(
            f"{label_map.get(method, method)} & "
            f"{_fmt_int(r['n_queries'])} & "
            f"{float(r['total_cost']):.1f} & "
            f"{float(r['accuracy']):.4f} & "
            f"{float(r['macro_f1']):.4f} & "
            f"{aguc_s} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            rf"\caption{{Post-drift cost and quality on {dataset}. Oracle cost "
            rf"{int(cost.cost_oracle)} per query, human cost {int(cost.cost_human)} "
            rf"per query; Static has zero supervision cost. AGUC is accuracy gain "
            rf"over Static per unit cost. Numbers are means across seeds.}}"
        ),
        rf"\label{{tab:cost_{dataset.lower()}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def summary_to_csv(summary: pd.DataFrame, out_path: Path) -> Path:
    """Write a summary frame to CSV and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    return out_path
