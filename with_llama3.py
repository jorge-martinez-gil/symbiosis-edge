# with_llama3.py
# Replaces the "Oracle" with a direct Llama 3 call via the Groq API, plus a DEV_MODE switch
# to reduce cost during development (fewer seeds, shorter stream, fewer datasets).
#
# Requirements:
#   pip install aiohttp numpy pandas matplotlib
#
# Secrets (do NOT hardcode):
#   export GROQ_API_KEY="..."
#
# DEV_MODE behavior:
# - seeds: 2 (instead of 10)
# - n: 1500 (instead of 1200)
# - window_w: 50 (instead of 200)
# - datasets: SYNTHETIC only (instead of SYNTHETIC+SECOM+APS)
#
# Chatbot output contract:
# The model must return STRICT JSON only: {"label": <int>} where label in [0, k_classes-1].

from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aiohttp


# ============================
# Styling (paper-friendly)
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


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


# ============================
# Schema
# ============================

@dataclass(frozen=True)
class MultiRunSchema:
    dataset: str = "dataset"
    seed: str = "seed"
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


def _mean_ci(mat: np.ndarray, ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mat.ndim != 2:
        raise ValueError("mat must be 2D: (runs, points)")
    n = mat.shape[0]
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=1) if n > 1 else np.zeros_like(mean)

    if abs(ci - 0.95) < 1e-9:
        z = 1.96
    elif abs(ci - 0.90) < 1e-9:
        z = 1.645
    elif abs(ci - 0.99) < 1e-9:
        z = 2.576
    else:
        z = 1.96

    half = z * (std / max(1.0, np.sqrt(n)))
    return mean, mean - half, mean + half


def _align_runs(
    keep2: pd.DataFrame,
    *,
    t_col: str = "t",
    run_col: str = "run_id",
    y_col: str = "__y__",
) -> Tuple[np.ndarray, np.ndarray]:
    t_all = np.sort(keep2[t_col].unique())
    runs = []
    for _, gg in keep2.groupby(run_col):
        s = gg.set_index(t_col).reindex(t_all)
        runs.append(s[y_col].to_numpy(dtype=float))
    mat = np.vstack(runs)
    return t_all, mat


def _auto_ylim_from_series(series_list: List[np.ndarray], pad: float = 0.10, min_top: float = 1.0) -> Tuple[float, float]:
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


def _collapse_duplicates_last(df: pd.DataFrame, *, t_col: str) -> pd.DataFrame:
    df = df.sort_values(t_col)
    return df.groupby(t_col, as_index=False).tail(1)


def _collapse_duplicates_mean(df: pd.DataFrame, *, t_col: str, val_col: str) -> pd.DataFrame:
    df = df.sort_values(t_col)
    return df.groupby(t_col, as_index=False)[val_col].mean()


def _collapse_duplicates_max(df: pd.DataFrame, *, t_col: str, val_col: str) -> pd.DataFrame:
    df = df.sort_values(t_col)
    return df.groupby(t_col, as_index=False)[val_col].max()


# ============================
# Uncertainty + routing (paper-style)
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
    q = 1.0 - b
    return float(np.quantile(values, q))


def _symbiosis_thresholds(window_u: np.ndarray, *, b_oracle: float, b_human: float) -> Tuple[float, float]:
    tau2 = _quantile_threshold(window_u, b_human)
    tau1 = _quantile_threshold(window_u, b_human + b_oracle)
    if tau1 > tau2:
        tau1 = tau2
    return tau1, tau2


# ============================
# Llama 3 oracle (via Groq API)
# ============================

@dataclass(frozen=True)
class OracleReply:
    raw_text: str
    label: Optional[int]


class LlamaOracle:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_url: str = "https://api.groq.com/openai/v1/chat/completions",
        model: str = "llama3-70b-8192",
        timeout_s: float = 30.0,
        max_retries: int = 3,
        concurrency: int = 6,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = api_url
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(concurrency)
        self._cache: Dict[str, OracleReply] = {}

        if not self.api_key:
            raise ValueError("Missing GROQ_API_KEY environment variable.")

    @staticmethod
    def _extract_text(data: object) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        # OpenAI-compatible response: choices[0].message.content
        choices = data.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
        return None

    async def label(self, *, cache_key: str, item_text: str, k_classes: int) -> OracleReply:
        if cache_key in self._cache:
            return self._cache[cache_key]

        system_prompt = (
            "You are a labeling assistant.\n"
            "Return ONLY a JSON object like {\"label\": <int>}.\n"
            f"label must be an integer in [0, {k_classes - 1}].\n"
            "No extra keys. No extra text.\n"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item_text},
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_err: Optional[Exception] = None

        async with self._sem:
            for attempt in range(self.max_retries):
                try:
                    timeout = aiohttp.ClientTimeout(total=self.timeout_s)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(self.api_url, headers=headers, json=payload) as resp:
                            text_body = await resp.text()

                            if resp.status < 200 or resp.status >= 300:
                                raise RuntimeError(f"Groq HTTP {resp.status}: {text_body[:500]}")

                            model_text = None
                            try:
                                data = json.loads(text_body)
                                model_text = self._extract_text(data)
                            except Exception:
                                model_text = None

                            if model_text is None:
                                model_text = text_body

                            label_val: Optional[int] = None
                            try:
                                obj = json.loads(model_text)
                                label_val = int(obj.get("label"))
                                if label_val < 0 or label_val >= k_classes:
                                    label_val = None
                            except Exception:
                                label_val = None

                            reply = OracleReply(raw_text=model_text, label=label_val)
                            self._cache[cache_key] = reply
                            return reply

                except Exception as e:
                    last_err = e
                    await asyncio.sleep(0.4 * (attempt + 1))

        raise RuntimeError(f"Llama oracle failed after retries: {last_err}")


def _sample_cache_key(*, dataset: str, t: int, k_classes: int) -> str:
    # Stable key per dataset + time step, so repeated seeds/methods reuse the same label.
    return f"{dataset}|t={t}|k={k_classes}"


def _make_item_text(*, dataset: str, t: int, k_classes: int) -> str:
    # Replace this with your real sample representation (features/text) when you have it.
    # IMPORTANT: do not include y_pred here if you want cache reuse.
    return (
        f"Task: classify the item into one of {k_classes} classes labeled 0..{k_classes - 1}.\n"
        f"Dataset: {dataset}\n"
        f"Time step: {t}\n"
        "Return ONLY JSON: {\"label\": <int>}."
    )


# ============================
# Stream simulator (with Llama oracle)
# ============================

@dataclass(frozen=True)
class SimParams:
    n: int = 1200
    k_classes: int = 4
    drift_t: int = 500
    window_w: int = 200

    b_sal: float = 0.12
    b_oracle: float = 0.12
    b_human: float = 0.05

    alpha_margin: float = 0.6

    u_floor: float = 0.0

    lr_edge: float = 0.000
    lr_oracle: float = 0.010
    lr_human: float = 0.016

    pre_acc_static: float = 0.92
    post_acc_static: float = 0.60

    pre_acc_sal: float = 0.93
    post_acc_sal: float = 0.55

    pre_acc_sym: float = 0.93
    post_acc_sym: float = 0.55

    post_noise: float = 0.02


async def simulate_one_run(
    *,
    dataset: str,
    seed: int,
    params: SimParams,
    oracle: LlamaOracle,
    use_llama_oracle: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = params.n
    k = params.k_classes

    y_true = rng.integers(0, k, size=n)

    state = {
        "Static": float(params.pre_acc_static),
        "SAL": float(params.pre_acc_sal),
        "Symbiosis-Edge": float(params.pre_acc_sym),
    }

    u_hist: Dict[str, List[float]] = {m: [] for m in state.keys()}
    rows: List[dict] = []

    for t in range(n):
        is_post = t >= params.drift_t
        env_noise = params.post_noise * (1.0 if is_post else 0.0)

        env_target = {
            "Static": params.post_acc_static if is_post else params.pre_acc_static,
            "SAL": params.post_acc_sal if is_post else params.pre_acc_sal,
            "Symbiosis-Edge": params.post_acc_sym if is_post else params.pre_acc_sym,
        }

        for method in ["Static", "SAL", "Symbiosis-Edge"]:
            p = state[method]
            p = 0.995 * p + 0.005 * env_target[method]
            p = float(np.clip(p + rng.normal(0.0, env_noise), 0.05, 0.999))
            state[method] = p

            u = _uncertainty_score(p, k=k, alpha=params.alpha_margin)
            u_hist[method].append(u)

            q_oracle = False
            q_human = False

            if method == "SAL":
                window = np.array(u_hist[method][-params.window_w:], dtype=float)
                tau_sal = _quantile_threshold(window, params.b_sal)
                if u > tau_sal:
                    q_oracle = True

            elif method == "Symbiosis-Edge":
                window = np.array(u_hist[method][-params.window_w:], dtype=float)
                tau1, tau2 = _symbiosis_thresholds(window, b_oracle=params.b_oracle, b_human=params.b_human)
                if u > tau2:
                    q_human = True
                elif u > tau1:
                    q_oracle = True

            correct = rng.random() < state[method]
            if correct:
                y_pred = int(y_true[t])
            else:
                other = [c for c in range(k) if c != int(y_true[t])]
                y_pred = int(rng.choice(other))

            # Online adaptation
            if method == "SAL":
                if q_oracle:
                    if use_llama_oracle:
                        cache_key = _sample_cache_key(dataset=dataset, t=t, k_classes=k)
                        item_text = _make_item_text(dataset=dataset, t=t, k_classes=k)
                        reply = await oracle.label(cache_key=cache_key, item_text=item_text, k_classes=k)
                        if reply.label is not None:
                            state[method] = float(np.clip(state[method] + params.lr_oracle, 0.05, 0.999))
                    else:
                        state[method] = float(np.clip(state[method] + params.lr_oracle, 0.05, 0.999))

            elif method == "Symbiosis-Edge":
                if q_human:
                    state[method] = float(np.clip(state[method] + params.lr_human, 0.05, 0.999))
                elif q_oracle:
                    if use_llama_oracle:
                        cache_key = _sample_cache_key(dataset=dataset, t=t, k_classes=k)
                        item_text = _make_item_text(dataset=dataset, t=t, k_classes=k)
                        reply = await oracle.label(cache_key=cache_key, item_text=item_text, k_classes=k)
                        if reply.label is not None:
                            state[method] = float(np.clip(state[method] + params.lr_oracle, 0.05, 0.999))
                    else:
                        state[method] = float(np.clip(state[method] + params.lr_oracle, 0.05, 0.999))
                else:
                    if u >= params.u_floor:
                        state[method] = float(np.clip(state[method] + params.lr_edge, 0.05, 0.999))

            rows.append({
                "dataset": dataset,
                "seed": seed,
                "t": t,
                "method": method,
                "y_true": int(y_true[t]),
                "y_pred": int(y_pred),
                "q_oracle": bool(q_oracle),
                "q_human": bool(q_human),
            })

    return pd.DataFrame(rows)


async def simulate_datasets(
    *,
    datasets: Sequence[DatasetConfig],
    seeds: int,
    params_by_dataset: Dict[str, SimParams],
    use_llama_oracle: bool = True,
    oracle_concurrency: int = 6,
) -> pd.DataFrame:
    oracle = LlamaOracle(concurrency=oracle_concurrency) if use_llama_oracle else LlamaOracle(concurrency=1)

    tasks: List[asyncio.Task[pd.DataFrame]] = []
    for ds in datasets:
        p = params_by_dataset[ds.name]
        for s in range(seeds):
            tasks.append(asyncio.create_task(
                simulate_one_run(dataset=ds.name, seed=s, params=p, oracle=oracle, use_llama_oracle=use_llama_oracle)
            ))

    frames = await asyncio.gather(*tasks)
    return pd.concat(frames, ignore_index=True)


# ============================
# Plot: combined accuracy + query counts with legend below x-label
# ============================

def plot_accuracy_with_query_count_panel_ci(
    df_one_dataset: pd.DataFrame,
    *,
    schema: MultiRunSchema,
    smooth_acc: int = 25,
    drift_t: Optional[int] = None,
    title: str = "",
    ci: float = 0.95,
    fig_size: Tuple[float, float] = (6.6, 3.8),
    acc_ylim: Tuple[float, float] = (0.0, 1.02),
    acc_collapse_rule: str = "last",
    symbiosis_name: str = "Symbiosis-Edge",
    symbiosis_green: str = "tab:green",
    query_ylim: Optional[Tuple[float, float]] = None,
    show_static_zero_line: bool = True,
) -> plt.Figure:
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=fig_size)
    fig.set_constrained_layout(False)

    gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.25], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    df = df_one_dataset.sort_values([schema.method, schema.seed, schema.t]).copy()
    df[schema.t] = df[schema.t].astype(int)
    df[schema.seed] = df[schema.seed].astype(int)

    method_to_color: Dict[str, str] = {}

    for method, g_method in df.groupby(schema.method):
        method = str(method)
        per_run = []
        for seed, gg in g_method.groupby(schema.seed):
            gg = gg[[schema.t, schema.y_true, schema.y_pred]].dropna()
            if gg.empty:
                continue

            if acc_collapse_rule == "mean":
                gg["__y__"] = (gg[schema.y_pred].to_numpy() == gg[schema.y_true].to_numpy()).astype(float)
                gg2 = _collapse_duplicates_mean(gg[[schema.t, "__y__"]], t_col=schema.t, val_col="__y__")
                y = _rolling_mean(gg2["__y__"].to_numpy(dtype=float), smooth_acc)
                t = gg2[schema.t].to_numpy(dtype=int)
            else:
                gg2 = _collapse_duplicates_last(gg, t_col=schema.t)
                correct = (gg2[schema.y_pred].to_numpy() == gg2[schema.y_true].to_numpy()).astype(float)
                y = _rolling_mean(correct, smooth_acc)
                t = gg2[schema.t].to_numpy(dtype=int)

            per_run.append(pd.DataFrame({"t": t, "__y__": y, "run_id": f"{seed}"}))

        if not per_run:
            continue

        keep2 = pd.concat(per_run, ignore_index=True)
        t_all, mat = _align_runs(keep2)
        mean, lo, hi = _mean_ci(mat, ci=ci)

        line = ax_top.plot(t_all, mean, label=method)[0]
        method_to_color[method] = line.get_color()
        ax_top.fill_between(t_all, lo, hi, alpha=0.18, linewidth=0)

    _draw_drift(ax_top, drift_t)

    ax_top.set_title(title)
    ax_top.set_ylabel(f"Rolling accuracy (w={smooth_acc})")
    ax_top.set_ylim(*acc_ylim)
    ax_top.legend(frameon=False, ncol=1, loc="lower left")
    ax_top.tick_params(labelbottom=False)

    plotted_means: List[np.ndarray] = []
    handles: List[plt.Line2D] = []
    labels: List[str] = []

    methods = [str(m) for m in df[schema.method].unique()]

    if show_static_zero_line and "Static" in methods:
        t_support = np.sort(df[schema.t].unique())
        zeros = np.zeros_like(t_support, dtype=float)
        h = ax_bot.plot(
            t_support, zeros,
            color=method_to_color.get("Static", None),
            linestyle="-",
        )[0]
        handles.append(h)
        labels.append("Static: total")
        plotted_means.append(zeros)

    if "SAL" in methods:
        df_sal = df[df[schema.method].astype(str) == "SAL"].copy()
        per_run_total = []
        for seed, g_seed in df_sal.groupby(schema.seed):
            gg = g_seed[[schema.t, schema.q_oracle, schema.q_human]].copy()
            gg[schema.q_oracle] = gg[schema.q_oracle].fillna(False).astype(bool).astype(int)
            gg[schema.q_human] = gg[schema.q_human].fillna(False).astype(bool).astype(int)
            gg["__total__"] = np.maximum(gg[schema.q_oracle].to_numpy(), gg[schema.q_human].to_numpy())

            gg2 = _collapse_duplicates_max(gg[[schema.t, "__total__"]], t_col=schema.t, val_col="__total__").sort_values(schema.t)
            t = gg2[schema.t].to_numpy(dtype=int)
            cum = np.cumsum(gg2["__total__"].to_numpy(dtype=int)).astype(float)
            per_run_total.append(pd.DataFrame({"t": t, "__y__": cum, "run_id": f"{seed}"}))

        if per_run_total:
            keep2 = pd.concat(per_run_total, ignore_index=True)
            t_all, mat = _align_runs(keep2)
            mean, lo, hi = _mean_ci(mat, ci=ci)

            c_sal = method_to_color.get("SAL", None)
            h = ax_bot.plot(t_all, mean, color=c_sal, linestyle="-")[0]
            ax_bot.fill_between(t_all, lo, hi, alpha=0.18, linewidth=0, color=c_sal)
            handles.append(h)
            labels.append("SAL: total")
            plotted_means.append(mean)

    if symbiosis_name in methods:
        df_sym = df[df[schema.method].astype(str) == symbiosis_name].copy()

        def _plot_sym(col: str, linestyle: str) -> Optional[np.ndarray]:
            per_run = []
            for seed, g_seed in df_sym.groupby(schema.seed):
                gg = g_seed[[schema.t, col]].dropna(subset=[col]).copy()
                if gg.empty:
                    continue
                gg[col] = gg[col].astype(bool).astype(int)
                gg2 = _collapse_duplicates_max(gg, t_col=schema.t, val_col=col).sort_values(schema.t)
                t = gg2[schema.t].to_numpy(dtype=int)
                cum = np.cumsum(gg2[col].to_numpy(dtype=int)).astype(float)
                per_run.append(pd.DataFrame({"t": t, "__y__": cum, "run_id": f"{seed}"}))

            if not per_run:
                return None

            keep2 = pd.concat(per_run, ignore_index=True)
            t_all, mat = _align_runs(keep2)
            mean, lo, hi = _mean_ci(mat, ci=ci)

            h = ax_bot.plot(t_all, mean, color=symbiosis_green, linestyle=linestyle)[0]
            ax_bot.fill_between(t_all, lo, hi, alpha=0.18, linewidth=0, color=symbiosis_green)
            handles.append(h)
            return mean

        m_or = _plot_sym(schema.q_oracle, ":")
        if m_or is not None:
            labels.append(f"{symbiosis_name}: Oracle")
            plotted_means.append(m_or)

        m_hu = _plot_sym(schema.q_human, "--")
        if m_hu is not None:
            labels.append(f"{symbiosis_name}: Human")
            plotted_means.append(m_hu)

    _draw_drift(ax_bot, drift_t)

    ax_bot.set_xlabel("Time step $t$")
    ax_bot.set_ylabel("Cum. queries")

    if query_ylim is None:
        lo, hi = _auto_ylim_from_series(plotted_means, pad=0.10, min_top=1.0)
        ax_bot.set_ylim(lo, hi)
    else:
        ax_bot.set_ylim(*query_ylim)

    ax_bot.legend(
        handles=handles,
        labels=labels,
        frameon=False,
        ncol=min(3, max(1, len(labels))),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        borderaxespad=0.0,
    )

    fig.subplots_adjust(bottom=0.28)
    return fig


# ============================
# Orchestrator
# ============================

def generate_paper_figures_for_datasets(
    df_all: pd.DataFrame,
    out_dir: str | Path,
    *,
    schema: MultiRunSchema,
    datasets: Sequence[DatasetConfig],
    smooth_acc: int = 25,
    ci: float = 0.95,
    acc_collapse_rule: str = "last",
    symbiosis_name: str = "Symbiosis-Edge",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = df_all.copy()
    for col in [schema.dataset, schema.method]:
        df_all[col] = df_all[col].astype(str)
    for col in [schema.seed, schema.t]:
        df_all[col] = df_all[col].astype(int)

    for ds in datasets:
        df_ds = df_all[df_all[schema.dataset] == ds.name].copy()
        if df_ds.empty:
            print(f"[WARN] No rows for dataset '{ds.name}'")
            continue

        fig = plot_accuracy_with_query_count_panel_ci(
            df_ds,
            schema=schema,
            smooth_acc=smooth_acc,
            drift_t=ds.drift_t,
            ci=ci,
            title=f"{ds.name}: accuracy + query counts (mean ± {int(ci*100)}% CI)",
            acc_collapse_rule=acc_collapse_rule,
            symbiosis_name=symbiosis_name,
            symbiosis_green="tab:green",
            query_ylim=None,
            show_static_zero_line=True,
        )
        _save(fig, out_dir / f"{ds.name.lower()}_accuracy_plus_query_counts")

    print(f"Saved figures to: {out_dir.resolve()}")


# ============================
# Main
# ============================

async def main() -> None:
    set_paper_style(use_tex=False, base_font=9)

    schema = MultiRunSchema()

    # Toggle this to switch between cheap dev runs and full runs.
    DEV_MODE = True

    DRIFT_T = 500

    # Default datasets
    DATASETS_FULL = (
        DatasetConfig("SECOM", DRIFT_T),
        DatasetConfig("APS", DRIFT_T),
        DatasetConfig("SYNTHETIC", DRIFT_T),
    )
    DATASETS_DEV = (DatasetConfig("SYNTHETIC", DRIFT_T),)

    DATASETS = DATASETS_DEV if DEV_MODE else DATASETS_FULL

    params_by_ds: Dict[str, SimParams] = {
        "SYNTHETIC": SimParams(
            drift_t=DRIFT_T,
            b_sal=0.12, b_oracle=0.12, b_human=0.05,
            alpha_margin=0.6,
            pre_acc_static=0.93, post_acc_static=0.60,
            pre_acc_sal=0.93, post_acc_sal=0.55,
            pre_acc_sym=0.93, post_acc_sym=0.55,
            post_noise=0.02,
        ),
        "SECOM": SimParams(
            drift_t=DRIFT_T,
            b_sal=0.12, b_oracle=0.12, b_human=0.05,
            alpha_margin=0.6,
            pre_acc_static=0.92, post_acc_static=0.62,
            pre_acc_sal=0.93, post_acc_sal=0.58,
            pre_acc_sym=0.94, post_acc_sym=0.58,
            post_noise=0.02,
        ),
        "APS": SimParams(
            drift_t=DRIFT_T,
            b_sal=0.12, b_oracle=0.12, b_human=0.05,
            alpha_margin=0.6,
            pre_acc_static=0.90, post_acc_static=0.60,
            pre_acc_sal=0.92, post_acc_sal=0.56,
            pre_acc_sym=0.93, post_acc_sym=0.56,
            post_noise=0.02,
        ),
    }

    # Oracle usage toggle
    USE_LLAMA_ORACLE = True

    # Dev cost controls
    SEEDS = 2 if DEV_MODE else 10
    ORACLE_CONCURRENCY = 2 if DEV_MODE else 6

    # Shorter stream and smaller uncertainty window in dev mode
    if DEV_MODE:
        new_params: Dict[str, SimParams] = {}
        for name, p in params_by_ds.items():
            new_params[name] = replace(p, n=1500, window_w=50)
        params_by_ds = new_params

    df = await simulate_datasets(
        datasets=DATASETS,
        seeds=SEEDS,
        params_by_dataset=params_by_ds,
        use_llama_oracle=USE_LLAMA_ORACLE,
        oracle_concurrency=ORACLE_CONCURRENCY,
    )

    generate_paper_figures_for_datasets(
        df,
        out_dir="paper_figures",
        schema=schema,
        datasets=DATASETS,
        smooth_acc=25,
        ci=0.95,
        acc_collapse_rule="last",
        symbiosis_name="Symbiosis-Edge",
    )


if __name__ == "__main__":
    asyncio.run(main())
