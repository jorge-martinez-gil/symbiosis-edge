# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-05

### Added
- **Statistics module** (`symbiosis_edge.stats`, numpy-only): Student-t
  confidence intervals (`t_critical`, `mean_ci_t`), percentile-bootstrap CIs
  (`bootstrap_ci`), **paired sign-flip permutation tests**
  (`paired_permutation_test`; exact enumeration for <=14 seeds, Monte Carlo
  with add-one correction beyond), effect sizes (`paired_cohens_d`,
  `cliffs_delta`), **Holm-Bonferroni** correction (`holm_bonferroni`), a full
  method-comparison battery (`compare_methods`), and Pareto-frontier
  extraction (`pareto_frontier`).
- **New artifacts** from `symbiosis-edge run` (multi-seed): `significance.csv`
  (p-values, Holm-adjusted p-values, effect sizes per dataset/metric/baseline),
  `tables/table_significance_*.tex`, and `figures/pareto_*.{pdf,png}`
  (cost-accuracy Pareto frontier with CI error bars).
- **CLI flags** `--ci {0.90,0.95,0.99}` and `--permutations N`; the run
  summary now prints paired significance vs every baseline.
- `metrics.per_seed_summary`: unaggregated per-(dataset, method, seed) rows,
  the input for paired tests.
- Statistical settings (CI method/level, test, correction) recorded in
  `manifest.json`; "Statistical methodology" section in
  `docs/experiments.md` with seed-count guidance (>=10 seeds for
  significance claims); "Statistical rigor" section in the README.
- Test suite for all statistical routines (exact permutation p-values,
  t critical values, Holm monotonicity, Pareto dominance, end-to-end
  `compare_methods` on a simulated run).

### Changed
- **Confidence intervals now use the Student-t critical value** instead of the
  normal z approximation everywhere (`summarize_runs`, `mean_ci`, figure CI
  bands). At the default 5 seeds this widens CIs by ~40% -- the previous
  z-intervals were anti-conservative for small samples. Means are unchanged.

### Earlier unreleased additions
- README citation section with BibTeX for the accompanying DEXA 2026 paper
  and for the software.
- `preferred-citation` entry in `CITATION.cff` pointing to the DEXA 2026 paper.
- Result preview figures embedded in the README (served from `docs/assets/`,
  since `results/` is gitignored).

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
