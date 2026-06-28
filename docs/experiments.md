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
| `summary_ci.csv` | Means with 95% confidence-interval half-widths across seeds. |
| `raw_runs.csv.gz` | Every per-step record (for custom analysis). |
| `tables/table_cost_*.tex` | Publication-ready LaTeX cost/quality tables. |
| `figures/accuracy_*.{pdf,png}` | Rolling accuracy over time with CI bands. |
| `figures/cost_accuracy_*.{pdf,png}` | Post-drift cost-vs-accuracy trade-off. |
| `manifest.json` | Versions, git commit, seeds, parameters, and SHA-256 of every output. |

Useful flags: `--datasets SECOM APS`, `--seeds 10`, `--n 4000`,
`--cost-human 20 --cost-oracle 2`, `--no-figures`, and `--quick` (a fast smoke run).

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
