"""End-to-end experiment runner with a reproducibility manifest.

``run_experiment`` is the single entry point behind ``symbiosis-edge run``: it
simulates every ``(dataset, seed)`` combination, computes the post-drift summary
and across-seed confidence intervals, renders figures and LaTeX tables, writes
tidy CSVs, and finally emits a ``manifest.json`` that records exactly how the
artifacts were produced -- versions, git commit, seeds, parameters, cost model,
and a SHA-256 for every output file. This makes any result fully traceable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import __version__
from .datasets import DATASET_PRESETS, default_datasets
from .metrics import CostModel, per_seed_summary, post_drift_summary, summarize_runs
from .params import DatasetConfig, Schema, SimParams
from .simulation import simulate_datasets
from .stats import compare_methods
from .viz import (
    latex_cost_table,
    latex_significance_table,
    plot_accuracy_over_time,
    plot_cost_vs_accuracy,
    plot_pareto_frontier,
    set_paper_style,
    summary_to_csv,
)

__all__ = ["ExperimentResult", "run_experiment"]


@dataclass
class ExperimentResult:
    """Paths and in-memory frames produced by :func:`run_experiment`."""

    out_dir: Path
    summary: pd.DataFrame
    ci_summary: pd.DataFrame
    raw: pd.DataFrame
    comparisons: Optional[pd.DataFrame] = None
    files: List[Path] = field(default_factory=list)
    manifest_path: Optional[Path] = None


def _git_commit(repo_dir: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_experiment(
    *,
    out_dir: Path | str = "results",
    datasets: Optional[Sequence[DatasetConfig]] = None,
    params_by_dataset: Optional[Dict[str, SimParams]] = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    cost: CostModel = CostModel(),
    drift_t: Optional[int] = None,
    make_figures: bool = True,
    schema: Schema = Schema(),
    ci: float = 0.95,
    n_permutations: int = 10_000,
) -> ExperimentResult:
    """Run the full benchmark and write all artifacts under ``out_dir``.

    Parameters
    ----------
    out_dir:
        Destination directory (created if missing). Sub-dirs ``figures/`` and
        ``tables/`` hold figures and LaTeX tables.
    datasets / params_by_dataset:
        Streams to run and their parameters. Defaults to the three built-in
        presets.
    seeds:
        Random seeds; one stream per seed enables confidence intervals.
    cost:
        Supervision cost model.
    drift_t:
        Drift time used to delimit the post-drift evaluation window. Defaults to
        the first dataset's ``drift_t``.
    make_figures:
        If ``False``, skip figure rendering (useful for fast CI smoke runs).
    ci:
        Confidence level for Student-t intervals (0.90, 0.95 or 0.99).
    n_permutations:
        Monte-Carlo permutations for the paired significance tests (an exact
        test is used automatically when there are <= 14 seeds).
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = default_datasets()
    if params_by_dataset is None:
        params_by_dataset = DATASET_PRESETS
    if drift_t is None:
        drift_t = datasets[0].drift_t

    raw = simulate_datasets(
        datasets=datasets, params_by_dataset=params_by_dataset, seeds=seeds
    )
    summary = post_drift_summary(raw, drift_t=drift_t, schema=schema, cost=cost)

    multi_seed = len(seeds) > 1
    comparisons: Optional[pd.DataFrame] = None
    if multi_seed:
        ci_summary = summarize_runs(raw, drift_t=drift_t, schema=schema, cost=cost, ci=ci)
        per_seed = per_seed_summary(raw, drift_t=drift_t, schema=schema, cost=cost)
        comparisons = compare_methods(
            per_seed, ci=ci, n_permutations=n_permutations,
            metrics=("accuracy", "macro_f1", "total_cost"),
        )
    else:
        ci_summary = summary.copy()

    files: List[Path] = []
    files.append(summary_to_csv(summary, out_dir / "summary.csv"))
    files.append(summary_to_csv(ci_summary, out_dir / "summary_ci.csv"))
    files.append(summary_to_csv(raw, out_dir / "raw_runs.csv.gz"))
    if comparisons is not None:
        files.append(summary_to_csv(comparisons, out_dir / "significance.csv"))

    dataset_names = [ds.name for ds in datasets]
    tab_dir.mkdir(parents=True, exist_ok=True)
    for name in dataset_names:
        tex = latex_cost_table(summary, dataset=name, cost=cost)
        p = tab_dir / f"table_cost_{name.lower()}.tex"
        p.write_text(tex, encoding="utf-8")
        files.append(p)
        if comparisons is not None:
            tex = latex_significance_table(comparisons, dataset=name, metric="accuracy")
            p = tab_dir / f"table_significance_{name.lower()}.tex"
            p.write_text(tex, encoding="utf-8")
            files.append(p)

    if make_figures:
        set_paper_style()
        for name in dataset_names:
            dsub = raw[raw[schema.dataset] == name]
            files += _save_accuracy(dsub, schema, drift_t, name, fig_dir)
            files += _save_cost_accuracy(summary, name, fig_dir)
            if multi_seed:
                files += _save_pareto(ci_summary, name, fig_dir)

    manifest_path = _write_manifest(
        out_dir=out_dir, files=files, datasets=datasets,
        params_by_dataset=params_by_dataset, seeds=seeds, cost=cost, drift_t=drift_t,
        ci=ci, n_permutations=n_permutations,
    )

    return ExperimentResult(
        out_dir=out_dir, summary=summary, ci_summary=ci_summary,
        raw=raw, comparisons=comparisons,
        files=files + [manifest_path], manifest_path=manifest_path,
    )


def _save_accuracy(dsub, schema, drift_t, name, fig_dir) -> List[Path]:
    plot_accuracy_over_time(
        dsub, schema=schema, drift_t=drift_t,
        title=f"{name}: accuracy over time",
        out_base=fig_dir / f"accuracy_{name.lower()}",
    )
    base = fig_dir / f"accuracy_{name.lower()}"
    return [base.with_suffix(".pdf"), base.with_suffix(".png")]


def _save_cost_accuracy(summary, name, fig_dir) -> List[Path]:
    plot_cost_vs_accuracy(
        summary, dataset=name, title=f"{name}: cost vs accuracy",
        out_base=fig_dir / f"cost_accuracy_{name.lower()}",
    )
    base = fig_dir / f"cost_accuracy_{name.lower()}"
    return [base.with_suffix(".pdf"), base.with_suffix(".png")]


def _save_pareto(ci_summary, name, fig_dir) -> List[Path]:
    plot_pareto_frontier(
        ci_summary, dataset=name, title=f"{name}: cost-accuracy Pareto frontier",
        out_base=fig_dir / f"pareto_{name.lower()}",
    )
    base = fig_dir / f"pareto_{name.lower()}"
    return [base.with_suffix(".pdf"), base.with_suffix(".png")]


def _write_manifest(
    *,
    out_dir: Path,
    files: Sequence[Path],
    datasets: Sequence[DatasetConfig],
    params_by_dataset: Dict[str, SimParams],
    seeds: Sequence[int],
    cost: CostModel,
    drift_t: Optional[int],
    ci: float = 0.95,
    n_permutations: int = 10_000,
) -> Path:
    import numpy as np

    manifest = {
        "schema": "symbiosis-edge/manifest@1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbiosis_edge_version": __version__,
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "config": {
            "seeds": list(int(s) for s in seeds),
            "drift_t": drift_t,
            "datasets": [ds.name for ds in datasets],
            "cost_model": asdict(cost),
            "statistics": {
                "ci_level": ci,
                "ci_method": "student-t",
                "significance_test": "paired sign-flip permutation "
                                     "(exact for <=14 seeds)",
                "n_permutations": n_permutations,
                "multiple_comparison_correction": "holm-bonferroni",
            },
            "params_by_dataset": {
                ds.name: asdict(params_by_dataset[ds.name]) for ds in datasets
            },
        },
        "outputs": [
            {
                "path": str(p.relative_to(out_dir)) if out_dir in p.parents else str(p.name),
                "sha256": _sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in files if Path(p).exists()
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
