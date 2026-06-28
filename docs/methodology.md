# Methodology: the Symbiosis-Edge simulation

This document specifies exactly what the Symbiosis-Edge benchmark computes, so
that results are transparent, reproducible, and easy to extend. It covers the
stream model, the uncertainty score, the routing policies, the cost model, and
the evaluation metrics — and is explicit about the scope and limitations of the
current simulation.

## What this is (and what it is not)

Symbiosis-Edge studies **cost-aware supervision routing under concept drift**:
when an edge model becomes uncertain, should the system trust the edge model,
query an LLM oracle, or spend scarce human-expert attention?

The bundled experiments are a **controlled parametric simulation of that routing
problem**, not the training of real classifiers on raw datasets:

- Each method maintains a scalar accuracy *state* that is pulled toward
  configurable pre-/post-drift targets. Supervision (oracle/human) nudges that
  state up (correct annotation) or down (wrong annotation).
- The dataset names `SYNTHETIC`, `SECOM`, and `APS` select **parameter presets**
  (`SimParams`). No external data files are read in the simulation path; the
  names are kept for continuity with the accompanying paper, which documents
  them as parametric stream settings.

This design makes the *dynamics of routing under a supervision budget* fully
transparent and deterministic, which is what the benchmark is about. It is
**not** a claim about the accuracy of a specific trained model on the raw UCI
SECOM or Scania APS data. Integrating real streaming-dataset loaders and real
online learners is on the roadmap (see `docs/extending.md`); contributions are
welcome.

Every number in this repository can be regenerated with one command
(`symbiosis-edge run`) and is accompanied by a `manifest.json` recording the
versions, seeds, parameters, and checksums used to produce it.

## Stream model

For a stream of length `n` with `k` classes and drift time `drift_t`, each
method `m` keeps an accuracy state `p_m`. At every step `t` and for each method:

1. **Environment pull.** `p_m <- 0.995 * p_m + 0.005 * target_m(t)`, where
   `target_m(t)` is the method's pre-drift target for `t < drift_t` and its
   post-drift target otherwise. Post-drift, Gaussian noise of std `post_noise`
   is added. State is clipped to `[0.05, 0.999]`.
2. **Uncertainty.** A probability vector is reconstructed from `p_m` and scored
   (below). The score feeds the routing policy.
3. **Routing.** The policy may query the oracle and/or the human, subject to a
   supervision budget.
4. **Supervision update.** A queried annotator is correct with probability
   `oracle_acc` / `human_acc`; a correct annotation adds `lr_*` to the state, a
   wrong one subtracts `lr_*_wrong`.
5. **Prediction.** The realised prediction is correct with probability `p_m`;
   otherwise a uniformly-random wrong class is emitted.

The order of random draws is fixed, so a given `(dataset, seed, params)` yields
exactly the same stream every time (verified by tests).

## Uncertainty score

From the scalar `p_correct`, a length-`k` probability vector places mass
`p_correct` on the correct class and spreads the rest uniformly. The score is

```
u(x) = H(p) + alpha * (1 - margin(p))
```

where `H` is Shannon entropy and `margin` is the top-two probability gap. Higher
`u` means the edge model is less certain.

## Routing policies (baselines + method)

All methods share the same uncertainty score and sliding-window quantile
thresholding, so they are compared on equal footing. The escalation threshold
for budget `b` is the `1 - b` empirical quantile of the recent window, which
keeps the realised query rate close to `b` even as the uncertainty distribution
drifts.

| Method | Tiers | Policy |
| --- | --- | --- |
| **Static** | none | Never queries; baseline accuracy floor. |
| **SAL** | oracle | Query when `u > tau(b_sal)`. |
| **ADWIN-SAL** | oracle | Drift-aware: an ADWIN detector temporarily raises the budget (and learning rate) after a detected change. |
| **Symbiosis-Edge** | oracle + human | Two nested thresholds split the budget: `u > tau_human` -> human; `tau_oracle < u <= tau_human` -> oracle; else stay on the edge. |

## Cost model and AGUC

Per-event costs default to: edge `0`, oracle `1`, human `10`.

- `Static`: cost `0`.
- `SAL` / `ADWIN-SAL`: single-tier; their queries are billed at the **human**
  rate (they have no cheap oracle tier in the costing).
- `Symbiosis-Edge`: `edge*steps + oracle*q_oracle + human*q_human`.

The headline efficiency metric is **Accuracy Gain per Unit Cost**:

```
AGUC = (accuracy_method - accuracy_static) / total_cost
```

computed over the **post-drift** segment, with `total_cost` in the same units as
the cost model (so AGUC is small in absolute terms when costs are in the
hundreds/thousands). Costs are configurable via `--cost-oracle/--cost-human/--cost-edge`.

## Metrics

Over the post-drift segment the runner reports, per method (means across seeds):
accuracy, balanced accuracy, macro-F1, Matthews correlation coefficient,
Cohen's kappa, number of oracle/human queries, total supervision cost, and AGUC.
The classification metrics are computed directly from the simulated
`y_true`/`y_pred` with a dependency-free confusion-matrix implementation.

## Reproducing

```bash
pip install -e .
symbiosis-edge run --seeds 5 --out results
```

This writes `summary.csv`, `summary_ci.csv`, gzipped raw runs, per-dataset
LaTeX tables, publication-quality figures, and `manifest.json`.
