#!/usr/bin/env python3
"""Fast single-seed experiment for quick figure generation and smoke tests.

Thin wrapper around the ``symbiosis_edge`` package. Equivalent to::

    symbiosis-edge run --seeds 1 --out results/single_run

The original monolithic implementation is preserved in ``scripts/_legacy/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from symbiosis_edge.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run", "--seeds", "1", "--out", "results/single_run"]))
