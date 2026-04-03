# Symbiosis-Edge

### Cost-Theoretic Drift Adaptation via Edge Models, LLM Oracles, and Human Experts

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-Compatible-FF6F00.svg)](https://www.tensorflow.org/lite)



[Paper](#citation) · [Overview](#overview) · [Architecture](#architecture) · [Results](#key-results) · [Getting Started](#getting-started) · [Citation](#citation)

</div>

---

## Overview

Industrial edge systems degrade under **concept drift**, yet the supervision needed to adapt them is scarce, expensive, and heterogeneous. Existing approaches treat drift handling as uncertainty-based sampling under a single oracle — overlooking latency, financial cost, and constrained human attention.

**Symbiosis-Edge** recasts drift adaptation as a **supervision allocation problem** under resource constraints. Each uncertain instance is routed to the *least costly agent* that can resolve it with sufficient expected utility:

| Agent | Cost | Strength | Role |
|---|---|---|---|
| **Edge Model** | ≈ 0 | Fast, local inference | Handles routine predictions |
| **LLM Oracle** | Moderate | Zero-shot reasoning | Resolves ambiguous cases + generates explanations |
| **Human Expert** | High | Ground-truth authority | Validates critical decisions |

This transforms drift adaptation from a *query selection* problem into a **cost-aware routing problem** with measurable economic impact.

---

## Core Mechanisms

### Cost-Aware Routing Policy

The routing policy π(xₜ) ∈ {E, O, H} minimizes total cost over a sliding window:

$$J = \sum_{t \in W} \bigl[\mathbb{1}(\hat{y}_t \neq y_t) \cdot \lambda_{\text{err}} + C_{\pi(x_t)}\bigr]$$

### Entropy-Based Uncertainty Signal

Shannon entropy from the edge model's softmax output serves as the routing signal — requiring only that higher entropy correlates with higher error risk on average, not perfect calibration:

$$u_E(x) = -\sum_{k=1}^{K} p_k(x) \log p_k(x)$$

### Budget-Aware Threshold Adaptation

Dynamic thresholds derived from sliding-window quantiles enforce strict supervision budgets without manual tuning:

$$\tau_2^{(t)} = \text{Quantile}_{1-B_H}(\mathcal{D}_W), \qquad \tau_1^{(t)} = \text{Quantile}_{1-(B_H+B_O)}(\mathcal{D}_W)$$

This guarantees human load ≤ B_H and oracle load ≤ B_O regardless of drift severity.

### Edge-Local Online Transfer

Only the classification head θ_ψ is updated; the feature extractor θ_φ remains frozen — enabling fast adaptation while preventing catastrophic forgetting.

---

## Key Results

Evaluated on three streams (synthetic drift, SECOM semiconductor manufacturing, APS vehicle failure) across 20 seeds with statistical significance at α = 0.05:

| Dataset | Method | Total Cost | Mean Accuracy | AGUC |
|---|---|---:|---:|---:|
| **Synthetic** | SAL | 2900 | 0.871 | 0.082 |
| | ADWIN-SAL | 2870 | 0.906 | 0.095 |
| | **Symbiosis-Edge** | **1273** | **0.935** | **0.237** |
| **SECOM** | SAL | 3000 | 0.868 | 0.080 |
| | ADWIN-SAL | 2980 | 0.906 | 0.093 |
| | **Symbiosis-Edge** | **1291** | **0.950** | **0.250** |
| **APS** | SAL | 3060 | 0.851 | 0.071 |
| | ADWIN-SAL | 2960 | 0.901 | 0.090 |
| | **Symbiosis-Edge** | **1354** | **0.940** | **0.225** |

**Highlights:**
- Recovery after abrupt drift within **~40 samples**
- Up to **2× cost-efficiency** (AGUC) over active-learning baselines
- Supervision cost reduced by **>50%** compared to single-oracle approaches
- Results validated with a live LLM oracle (ChatGPT-5.2, Llama 3, Mistral) — deviations < 2%

**AGUC** (*Accuracy Gain per Unit Cost*) = (Acc_method − Acc_static) / TotalCost — measures how efficiently supervision budget converts into post-drift accuracy.

---

## Acknowledgments

This work was funded by the EU via the **ASTRID** action (oc1-2025-TIS-01), implemented under the **ENFIELD** project, grant agreement No 101120657.

---


## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Questions or collaboration?** Open an issue or contact [jorge.martinez-gil@scch.at](mailto:jorge.martinez-gil@scch.at)

Software Competence Center Hagenberg (SCCH) · Softwarepark 32a · 4232 Hagenberg, Austria

</div>