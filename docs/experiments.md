# Experiment Guide

This repository contains research scripts for evaluating Symbiosis-Edge under synthetic and real-world drift scenarios.

## Offline Experiments

| Script | Purpose |
| --- | --- |
| `scripts/run_multi_dataset.py` | Main multi-dataset, multi-seed experiment with Static, SAL, ADWIN-SAL, and Symbiosis-Edge. |
| `scripts/run_single_run.py` | Faster one-run experiment for quick figure generation and smoke tests. |
| `scripts/run_without_adwin.py` | Ablation script that excludes the ADWIN-SAL baseline. |

## LLM Oracle Experiments

| Script | Provider | Required environment variables |
| --- | --- | --- |
| `scripts/run_chatbase_oracle.py` | Chatbase | `CHATBASE_API_KEY`, `CHATBASE_CHATBOT_ID` |
| `scripts/run_llama3_oracle.py` | Groq Llama 3 | `GROQ_API_KEY` |
| `scripts/run_mistral_oracle.py` | Mistral AI | `MISTRAL_API_KEY` |

The oracle scripts expect strict JSON responses with the shape `{"label": <int>}`.

## Outputs

Most scripts write figures to `paper_figures/` and tables to `paper_tables/`. These directories are generated artifacts and are ignored by Git.

## Recommended Workflow

1. Run `python scripts/run_single_run.py` as a quick smoke test.
2. Run `python scripts/run_multi_dataset.py` for paper-quality aggregate results.
3. Use the LLM-backed scripts only after confirming credentials and development-mode settings.
