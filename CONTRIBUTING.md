# Contributing to Symbiosis-Edge

Thanks for your interest in improving Symbiosis-Edge! This project aims to be a
reproducible, well-tested benchmark for **cost-aware supervision routing under
concept drift**. Contributions of code, baselines, datasets/presets, docs, and
bug reports are all welcome.

## Development setup

```bash
git clone https://github.com/jorge-martinez-gil/symbiosis-edge
cd symbiosis-edge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
```

Run the checks the CI runs:

```bash
ruff check symbiosis_edge tests   # lint
pytest -q                         # tests
symbiosis-edge run --quick --out /tmp/se-smoke   # end-to-end smoke test
```

## Project layout

```
symbiosis_edge/        # the installable library (single source of truth)
  uncertainty.py       # edge-model uncertainty scoring
  routing.py           # budget-aware quantile routing thresholds
  drift.py             # SimpleADWIN change detector
  simulation.py        # the stream simulator (simulate_one_run / simulate_datasets)
  metrics.py           # cost model, AGUC, classification metrics, CIs
  datasets.py          # SimParams presets (SYNTHETIC, SECOM, APS)
  viz.py               # publication-quality figures + LaTeX tables
  report.py            # end-to-end runner + reproducibility manifest
  cli.py               # `symbiosis-edge` command-line interface
tests/                 # pytest suite
scripts/               # thin wrappers around the package
scripts/_legacy/       # original monolithic scripts, kept for reference
docs/                  # methodology and guides
```

## What makes a good pull request

Every PR should **increase scientific usefulness** and keep results reproducible:

- Add or update tests for any behaviour you change. Numerical changes to the
  simulator must explain how determinism and comparability are preserved.
- Keep the public API documented with docstrings.
- Run `ruff` and `pytest` locally; CI must be green.
- Never hand-tune results to look better. The cost/quality numbers must emerge
  from an actual run of the engine.

## Adding a baseline or routing policy

The routing logic lives in `simulation.simulate_one_run` and the threshold
helpers in `routing.py`. New baselines should:

1. Compute uncertainty the same way (`uncertainty.uncertainty_score`) so methods
   are compared on equal footing.
2. Respect an explicit supervision budget.
3. Be added to `simulation.METHODS` in a fixed position (the order is part of the
   RNG-determinism contract).

See `docs/extending.md` for a worked example and the planned policy plug-in API.

## Adding a dataset preset

Add a `SimParams` entry to `datasets.DATASET_PRESETS` (see `docs/methodology.md`
for what each field means). If you wire in a *real* streaming dataset loader,
please document the source, license, and preprocessing so results stay traceable.

## Reporting bugs / requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE`. For benchmark-result
submissions, include the `manifest.json` produced by your run so results are
verifiable.

## License

By contributing you agree that your contributions are licensed under the MIT
License (see `LICENSE`).
