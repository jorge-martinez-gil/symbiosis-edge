#!/usr/bin/env python3
"""Ablation: report results without the ADWIN-SAL baseline.

Thin wrapper around the ``symbiosis_edge`` package. The full stream is still
simulated for every method (so the random stream and the remaining methods are
identical to the main experiment), and ADWIN-SAL is simply excluded from the
ablation summary -- this keeps the comparison fair and deterministic.

Outputs are written under ``results/without_adwin``. The original monolithic
implementation is preserved in ``scripts/_legacy/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from symbiosis_edge.report import run_experiment  # noqa: E402

if __name__ == "__main__":
    result = run_experiment(out_dir="results/without_adwin", seeds=range(5))

    ablation = result.summary[result.summary["method"] != "ADWIN-SAL"].copy()
    out_path = Path(result.out_dir) / "summary_without_adwin.csv"
    ablation.to_csv(out_path, index=False)

    cols = ["dataset", "method", "accuracy", "macro_f1", "mcc", "total_cost", "aguc"]
    print("\nAblation summary (ADWIN-SAL excluded):")
    print(ablation[cols].to_string(index=False))
    print(f"\nWritten to: {out_path}")
