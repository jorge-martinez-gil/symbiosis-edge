# Extending Symbiosis-Edge

This guide shows how to evaluate your own ideas with the benchmark and outlines
the planned plug-in API.

## Add a dataset preset

A "dataset" is a `SimParams` preset. Register one and run it:

```python
from dataclasses import replace
from symbiosis_edge import DATASET_PRESETS, DatasetConfig
from symbiosis_edge.report import run_experiment

my_params = replace(DATASET_PRESETS["SYNTHETIC"], post_acc_static=0.50, post_noise=0.04)
presets = {**DATASET_PRESETS, "MYSTREAM": my_params}

run_experiment(
    out_dir="results/mystream",
    datasets=[DatasetConfig("MYSTREAM", my_params.drift_t)],
    params_by_dataset=presets,
    seeds=range(5),
)
```

See `docs/methodology.md` for the meaning of every `SimParams` field.

## Use the library directly

```python
from symbiosis_edge import SimParams, simulate_one_run, post_drift_summary

df = simulate_one_run(dataset="SYNTHETIC", seed=0, params=SimParams())
print(post_drift_summary(df, drift_t=500))
```

`simulate_datasets(...)` runs many `(dataset, seed)` combinations and adds a
`seed` column; `summarize_runs(...)` aggregates them with confidence intervals.

## Add a routing policy or baseline

Today the four methods are implemented inline in
`symbiosis_edge/simulation.simulate_one_run`, and the budget-aware thresholds
live in `symbiosis_edge/routing.py`. To add a baseline:

1. Reuse `uncertainty.uncertainty_score` so methods are compared on equal terms.
2. Decide escalation from a budget via `routing.quantile_threshold` (single
   tier) or `routing.symbiosis_thresholds` (oracle + human tiers).
3. Add the method name to `simulation.METHODS` in a fixed position — the order
   is part of the determinism contract, so appending is safest.
4. Add a test (determinism + that the realised query rate respects the budget).

### Roadmap: a policy plug-in API

A near-term goal is to factor routing into a small `Policy` protocol
(`decide(u, window, budgets) -> {edge, oracle, human}`) so third-party policies
can be dropped in without editing the simulator, and so a real online learner
(producing genuine probabilities and uncertainty) can replace the parametric
accuracy state. Issues and PRs toward this are very welcome.
