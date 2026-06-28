#!/usr/bin/env python3
"""Main multi-dataset, multi-seed experiment (Static, SAL, ADWIN-SAL, Symbiosis-Edge).

This is now a thin wrapper around the installable ``symbiosis_edge`` package,
which is the single tested source of truth for the simulation logic. The
canonical, fully configurable entry point is the CLI::

    symbiosis-edge run --seeds 5 --out results

Running this file directly reproduces the paper-style multi-seed experiment and
writes figures, LaTeX tables, tidy CSVs, and a reproducibility manifest under
``results/``. The original monolithic implementation is preserved verbatim in
``scripts/_legacy/`` for reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/run_multi_dataset.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from symbiosis_edge.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run", "--seeds", "5", "--out", "results"]))
