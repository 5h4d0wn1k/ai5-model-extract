# AI5 — Model Extraction Tool

Query-based model stealing, decision boundary mapping, and parameter estimation.

## Overview

This project demonstrates how to extract knowledge from black-box ML models through API queries:
- **Query-based model stealing**: Building surrogate models from query responses
- **Decision boundary mapping**: Locating classification boundaries via binary search
- **Architecture inference**: Inferring model properties from query behavior
- **Parameter estimation**: Recovering model weights through regression

## Features

- Query budget management system
- Random and adaptive sampling strategies
- Decision boundary detection via binary search
- Linear parameter estimation via pseudo-inverse
- Fidelity scoring between target and surrogate models

## Installation

```bash
# No external dependencies - numpy only
python3 -c "import numpy; print('numpy available')"
```

## Usage

```bash
# Offline demo (no API, no network) — black-box target is simulated locally, exit 0
python3 model_extract.py

# Tunable experiment
python3 model_extract.py --features 10 --classes 4 --queries 5000 --seed 42

# JSON report to reports/ (gitignored)
python3 model_extract.py --output reports/ai5-report.json

# Quiet CI mode + JSON
python3 model_extract.py --quiet --output reports/ai5-report.json
```

### Exit Codes

- `0` — experiment completed cleanly
- `1` — error (bad arguments / report write failure)

### Live Lab Test Plan

Runs entirely offline — the black-box target model, query budget, and surrogate
are all generated locally; no external inference API is contacted.

1. **Demo**: `python3 model_extract.py` — expect architecture inference, random/adaptive sampling extraction, boundary mapping, linear parameter estimation, and query-budget blocks. Exit `0`.
2. **Accuracy transfer**: verify `accuracy_transfer.fidelity_adaptive_sampling` shows the surrogate agreeing with the target on held-out inputs, and compare `fidelity_random_sampling` for sampling-strategy effectiveness.
3. **Budget discipline**: `query_budget.used <= query_budget.max_queries` always — extraction is throttled by the simulated budget.
4. **JSON report**: `python3 model_extract.py --output reports/ai5-report.json` — verify `accuracy_transfer`, `query_budget`, `parameter_estimation` present.
5. **Unit tests**: `python3 -m unittest discover -s tests -v` — all pass (target predict shapes, budget accounting, architecture inference fields, surrogate training, fidelity bounds, budget clamp on tight budgets, CLI JSON write).

## Metrics

- Real extraction code paths exercised offline: `TargetModel.query/query_confidence`, `QueryBudget.can_query/consume`, `ArchitectureInferrer.infer_by_probing/infer_nonlinearity`, `ModelExtractor.random_sampling/adaptive_sampling/boundary_mapping`, `SurrogateModel.forward/train/predict`, `ParameterEstimator.estimate_linear_params/compute_confidence`, `fidelity_score`
- Metrics emitted: classes detected, mean confidence, scale sensitivity, estimated depth, nonlinearity; per-strategy fidelity (accuracy transfer); linear-fit accuracy/confidence; query budget used/remaining/ratio
- Budget clamping hardened so extraction never exceeds `max_queries`
- 8 unit tests; exit-code contract `0` clean / `1` error
- Zero runtime cloud/network dependencies; offline demo needs only numpy

## Example Output

```
============================================================
AI5 — Model Extraction Tool Demonstration
============================================================

[1] Architecture inference (budget: 5000)...
    Classes detected: 4
    Mean confidence:  0.7800
    Scale sensitivity: 0.1200
    Estimated depth:  deep
    Budget remaining: 4200

[2] Random sampling extraction (budget: 4200)...
    Collected 500 samples
    Fidelity (random): 0.7200

[3] Adaptive sampling extraction (budget: 3700)...
    Collected 400 samples total
    Fidelity (adaptive): 0.8500

[4] Decision boundary mapping (budget: 3100)...
    Boundary points mapped: 60
    Budget remaining: 2500

[5] Parameter estimation (linear fit)...
    Weight matrix shape: (10, 4)
    Linear model accuracy: 0.6500

[6] Query budget summary...
    Max queries:     5000
    Used queries:    2500
    Remaining:       2500
    Usage ratio:     50.00%
```

## Legal Disclaimer

**IMPORTANT: Read before use.**

This project is provided for **educational and authorized security testing purposes only**.

### Authorization Requirements
- You MUST have explicit written permission from the model owner before using this tool
- Unauthorized model extraction may violate intellectual property laws
- This tool should ONLY be used on models you own or have written authorization to test

### Legal Framework
- **Computer Fraud and Abuse Act (CFAA)**: Unauthorized access to computer systems is a federal crime
- **Intellectual Property Law**: Model extraction may constitute trade secret misappropriation
- **Terms of Service**: Querying APIs may violate service terms
- **Emerging AI Regulations**: Model theft may be regulated under new AI laws

### Acceptable Use
- Testing security of your own ML models
- Authorized red team exercises with written scope
- Academic research in controlled lab environments
- Security education and training

### Prohibited Use
- Extracting proprietary models without authorization
- Violating API terms of service
- Any activity that violates applicable laws or regulations
- Commercial use without proper licensing

### No Warranty
This software is provided "AS IS" without warranty of any kind. The author is not responsible for any misuse or damage caused by this software.

### Responsible Disclosure
If you discover vulnerabilities using this tool, follow responsible disclosure practices:
1. Report to the vendor/owner privately
2. Allow reasonable time for remediation
3. Do not exploit beyond proof of concept

## License

MIT
