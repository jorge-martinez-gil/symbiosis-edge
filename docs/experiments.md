# Experiment Guide

Symbiosis-Edge evaluates **cost-aware supervision routing under concept drift**.
All experiments are driven by the installable `symbiosis_edge` package and are
fully reproducible from a single command. See `docs/methodology.md` for the
exact model and `docs/extending.md` for how to add your own baselines/presets.

## One-command reproduction

```bash
pip install -e .
symbiosis-edge run --seeds 5 --out results
```

This simulates every (dataset, seed) combination and writes, under `results/`:

| Artifact | Contents |
| --- | --- |
| `summary.csv` | Per-(dataset, method) post-drift metrics (means across seeds). |
| `summary_ci.csv` | Means with 95% Student-t confidence-interval half-widths across seeds. |
| `significance.csv` | Paired permutation tests of Symbiosis-Edge vs every baseline (p-values, Holm-adjusted p-values, Cohen's d_z, Cliff's delta), per dataset and metric. |
| `raw_runs.csv.gz` | Every per-step record (for custom analysis). |
| `tables/table_cost_*.tex` | Publication-ready LaTeX cost/quality tables. |
| `tables/table_significance_*.tex` | Publication-ready LaTeX significance tables. |
| `figures/accuracy_*.{pdf,png}` | Rolling accuracy over time with CI bands. |
| `figures/cost_accuracy_*.{pdf,png}` | Post-drift cost-vs-accuracy trade-off. |
| `figures/pareto_*.{pdf,png}` | Cost-accuracy Pareto frontier with CI error bars. |
| `manifest.json` | Versions, git commit, seeds, parameters, statistical settings, and SHA-256 of every output. |

Useful flags: `--datasets SECOM APS`, `--seeds 10`, `--n 4000`,
`--cost-human 20 --cost-oracle 2`, `--ci 0.99`, `--permutations 100000`,
`--no-figures`, and `--quick` (a fast smoke run).

## Statistical methodology

All aggregation across seeds is done with small-sample-honest statistics:

- **Confidence intervals** use the Student-t critical value (`t_4 = 2.776` at
  5 seeds), not the normal approximation (`z = 1.96`), which would be ~40% too
  narrow at that sample size.
- **Significance** is assessed with a **paired sign-flip permutation test** on
  per-seed differences: distribution-free, valid at any sample size, and
  *exact* (full enumeration of all `2^n` sign patterns) for up to 14 seeds.
  Beyond that, 10,000 Monte-Carlo permutations with an add-one correction.
- **Multiple comparisons** across baselines are Holm-Bonferroni adjusted
  (`p_holm` in `significance.csv`), controlling the family-wise error rate.
- **Effect sizes** are reported alongside p-values: Cohen's d_z (paired) and
  Cliff's delta (ordinal, outlier-robust).
- **Pareto analysis**: `figures/pareto_*` marks which methods are
  non-dominated in the (cost, accuracy) plane; dominated methods are hollow.

> **Seed-count guidance.** An exact permutation test with 5 seeds cannot
> produce a two-sided p-value below `2/2^5 = 0.0625` — so no comparison can
> reach the conventional 0.05 level regardless of how large the effect is.
> Use `--seeds 10` or more (min. achievable p = 0.002) when you intend to make
> significance claims; effect sizes and CIs remain informative at any count.

## Offline experiment wrappers

These thin wrappers call the package and remain for backward compatibility; the
original monolithic implementations are preserved in `scripts/_legacy/`.

| Script | Equivalent CLI |
| --- | --- |
| `scripts/run_multi_dataset.py` | `symbiosis-edge run --seeds 5 --out results` |
| `scripts/run_single_run.py` | `symbiosis-edge run --seeds 1 --out results/single_run` |
| `scripts/run_without_adwin.py` | runs all methods, reports the ADWIN-SAL ablation |

## LLM oracle experiments (optional)

The provider scripts in `scripts/` (Chatbase, Groq Llama 3, Mistral) explore
using an LLM as the oracle annotator. They require credentials (see
`.env.example`) and the `llm` extra (`pip install -e ".[llm]"`). They currently
send a placeholder item representation; wiring real per-instance features is part
of the roadmap in `docs/extending.md`.

| Script | Provider | Environment variables |
| --- | --- | --- |
| `scripts/run_chatbase_oracle.py` | Chatbase | `CHATBASE_API_KEY`, `CHATBASE_CHATBOT_ID` |
| `scripts/run_llama3_oracle.py` | Groq Llama 3 | `GROQ_API_KEY` |
| `scripts/run_mistral_oracle.py` | Mistral AI | `MISTRAL_API_KEY` |
