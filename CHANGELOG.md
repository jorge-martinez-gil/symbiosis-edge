# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-28

### Added
- **Installable `symbiosis_edge` Python package** that is now the single,
  tested source of truth for the simulation engine (previously the same logic
  was duplicated across six standalone scripts).
- **Command-line interface** `symbiosis-edge` with `run` and `info` commands,
  exposed as a console entry point.
- **Reproducibility manifest** (`manifest.json`): records package/library
  versions, git commit, seeds, per-dataset parameters, the cost model, and a
  SHA-256 for every generated artifact.
- **Test suite** (pytest) covering uncertainty scoring, routing thresholds,
  ADWIN change detection, simulation determinism, the cost/AGUC model, and the
  CLI; **continuous integration** (GitHub Actions) across Python 3.9-3.12 with
  linting (ruff) and an end-to-end CLI smoke test.
- **Additional honest metrics** computed from the simulated predictions:
  balanced accuracy, macro-F1, Matthews correlation coefficient, and Cohen's
  kappa (dependency-free implementations).
- **`docs/methodology.md`** documenting the exact simulation model and its
  limitations, and **`docs/extending.md`** describing how to add baselines.
- Packaging metadata (`pyproject.toml`), contributor guide, issue/PR templates.

### Changed
- The ADWIN change-detector split scan is now vectorised with prefix sums
  (~14x faster per full run) while remaining numerically identical to the
  original double loop (locked by a regression test).
- The three offline scripts (`run_multi_dataset.py`, `run_single_run.py`,
  `run_without_adwin.py`) are now thin, back-compatible wrappers around the
  package. The original monolithic implementations are preserved in
  `scripts/_legacy/`.
- README and docs clarify that reported numbers come from a reproducible
  parametric **simulation**, not from training models on raw datasets.

### Notes
- No change to the simulation's numerical behaviour: a given
  `(dataset, seed, params)` produces exactly the same stream as before.

## [0.1.0] - 2026-04
- Initial public release: simulation scripts, baselines, LLM-oracle variants,
  and citation metadata.
