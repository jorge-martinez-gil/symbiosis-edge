# Symbiosis-Edge

Cost-theoretic drift adaptation with edge models, LLM oracles, and human experts.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-Compatible-FF6F00.svg)](https://www.tensorflow.org/lite)

## Overview

Industrial edge systems degrade under concept drift, but the supervision needed to adapt them is scarce, expensive, and heterogeneous. Symbiosis-Edge frames drift adaptation as a supervision allocation problem under resource constraints.

Each uncertain instance is routed to the least costly agent that can resolve it with sufficient expected utility:

| Agent | Cost | Strength | Role |
| --- | --- | --- | --- |
| Edge model | Approx. 0 | Fast local inference | Handles routine predictions |
| LLM oracle | Moderate | Zero-shot reasoning | Resolves ambiguous cases and generates explanations |
| Human expert | High | Ground-truth authority | Validates critical decisions |

This turns drift adaptation from query selection into cost-aware routing with measurable economic impact.

## Repository Layout

```text
.
├── scripts/
│   ├── run_multi_dataset.py       # Main multi-seed experiment with ADWIN-SAL
│   ├── run_single_run.py          # One-run version for fast figure generation
│   ├── run_without_adwin.py       # Ablation without the ADWIN-SAL baseline
│   ├── run_chatbase_oracle.py     # Chatbase-backed oracle experiment
│   ├── run_llama3_oracle.py       # Groq Llama 3-backed oracle experiment
│   └── run_mistral_oracle.py      # Mistral-backed oracle experiment
├── docs/
│   └── experiments.md             # Experiment notes and script guide
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

Generated artifacts are written to `paper_figures/` and `paper_tables/`.

## Getting Started

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the main offline experiment:

```bash
python scripts/run_multi_dataset.py
```

Run a faster single-run version:

```bash
python scripts/run_single_run.py
```

## LLM Oracle Runs

The LLM-backed scripts require API keys. Copy `.env.example`, fill in the relevant values, and export them in your shell before running the script.

```bash
python scripts/run_chatbase_oracle.py
python scripts/run_llama3_oracle.py
python scripts/run_mistral_oracle.py
```

The LLM scripts include a development mode in the source code to reduce cost while testing.

## Core Mechanisms

The routing policy minimizes total cost over a sliding window:

```text
J = sum_t [1(y_hat_t != y_t) * lambda_err + C_pi(x_t)]
```

Shannon entropy from the edge model softmax acts as the routing signal:

```text
u_E(x) = -sum_k p_k(x) log p_k(x)
```

Dynamic thresholds from sliding-window quantiles enforce oracle and human budgets without manual tuning.

## Key Results

Across synthetic drift, SECOM semiconductor manufacturing, and APS vehicle failure streams, Symbiosis-Edge reduces supervision cost by more than 50 percent compared with single-oracle approaches while improving post-drift accuracy.

## Citation

If you use this project in academic work, please cite the associated paper or repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

Questions or collaboration: [jorge.martinez-gil@scch.at](mailto:jorge.martinez-gil@scch.at)

Software Competence Center Hagenberg (SCCH), Softwarepark 32a, 4232 Hagenberg, Austria
