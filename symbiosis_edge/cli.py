"""Command-line interface for Symbiosis-Edge.

Examples
--------
Reproduce the full benchmark (figures + tables + manifest) into ``results/``::

    symbiosis-edge run

A fast smoke run (few seeds, short streams, no figures)::

    symbiosis-edge run --quick --out /tmp/se-smoke

Only the SECOM preset, custom costs, 10 seeds::

    symbiosis-edge run --datasets SECOM --seeds 10 --cost-human 20

Inspect the installed version and available presets::

    symbiosis-edge info
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .datasets import DATASET_PRESETS, DRIFT_T, default_datasets
from .metrics import CostModel
from .params import DatasetConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="symbiosis-edge",
        description="Cost-aware supervision routing under concept drift: "
                    "reproducible simulation benchmark.",
    )
    p.add_argument("--version", action="version", version=f"symbiosis-edge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the benchmark and write all artifacts.")
    run.add_argument("--out", default="results", type=Path,
                     help="Output directory (default: results).")
    run.add_argument("--datasets", nargs="+", default=None, metavar="NAME",
                     help=f"Subset of presets to run (default: all). "
                          f"Choices: {sorted(DATASET_PRESETS)}.")
    run.add_argument("--seeds", type=int, default=5,
                     help="Number of seeds 0..N-1 (default: 5).")
    run.add_argument("--n", type=int, default=None,
                     help="Override stream length per dataset.")
    run.add_argument("--cost-oracle", type=float, default=1.0,
                     help="Cost per oracle query (default: 1).")
    run.add_argument("--cost-human", type=float, default=10.0,
                     help="Cost per human annotation (default: 10).")
    run.add_argument("--cost-edge", type=float, default=0.0,
                     help="Cost per edge step (default: 0).")
    run.add_argument("--no-figures", action="store_true",
                     help="Skip figure rendering (faster).")
    run.add_argument("--quick", action="store_true",
                     help="Fast smoke run: 2 seeds, short streams, no figures.")

    sub.add_parser("info", help="Show version, presets, and methods.")
    return p


def _resolve_datasets(names: Optional[Sequence[str]]) -> List[DatasetConfig]:
    if not names:
        return default_datasets()
    out: List[DatasetConfig] = []
    for raw in names:
        key = raw.upper()
        if key not in DATASET_PRESETS:
            raise SystemExit(
                f"Unknown dataset '{raw}'. Available: {sorted(DATASET_PRESETS)}"
            )
        out.append(DatasetConfig(key, DRIFT_T))
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    # Imported lazily so `info`/`--version` stay fast and dependency-light.
    from .report import run_experiment

    datasets = _resolve_datasets(args.datasets)
    seeds = list(range(max(1, args.seeds)))
    n_override = args.n
    make_figures = not args.no_figures

    if args.quick:
        seeds = [0, 1]
        n_override = n_override or 400
        make_figures = False

    params_by_dataset = dict(DATASET_PRESETS)
    if n_override is not None:
        # Keep the drift point inside the (possibly shortened) stream so the
        # post-drift evaluation window is non-empty.
        drift_override = max(1, min(DRIFT_T, n_override // 2))
        params_by_dataset = {
            name: replace(p, n=n_override, drift_t=drift_override)
            for name, p in params_by_dataset.items()
        }
        datasets = [DatasetConfig(d.name, drift_override) for d in datasets]

    cost = CostModel(
        cost_edge_step=args.cost_edge,
        cost_oracle=args.cost_oracle,
        cost_human=args.cost_human,
    )

    print(f"symbiosis-edge {__version__}: running {[d.name for d in datasets]} "
          f"x {len(seeds)} seed(s) -> {args.out}", file=sys.stderr)

    result = run_experiment(
        out_dir=args.out,
        datasets=datasets,
        params_by_dataset=params_by_dataset,
        seeds=seeds,
        cost=cost,
        make_figures=make_figures,
    )

    cols = ["dataset", "method", "accuracy", "macro_f1", "mcc", "n_queries", "total_cost", "aguc"]
    print(result.summary[cols].to_string(index=False))
    print(f"\nArtifacts written to: {result.out_dir.resolve()}")
    print(f"Manifest: {result.manifest_path}")
    print(f"{len(result.files)} files generated.")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    from .simulation import METHODS

    print(f"symbiosis-edge {__version__}")
    print(f"Methods : {', '.join(METHODS)}")
    print(f"Presets : {', '.join(sorted(DATASET_PRESETS))}")
    print(f"Drift t : {DRIFT_T}")
    print("Metrics : accuracy, balanced_accuracy, macro_f1, mcc, cohen_kappa, "
          "total_cost, n_queries, aguc")
    print("Docs    : docs/methodology.md  |  Reproduce: symbiosis-edge run")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "info":
        return _cmd_info(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
