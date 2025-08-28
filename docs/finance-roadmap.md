# Finance Evaluation (TB-CSPN vs LangGraph)

This compares a typed, single-call TB-CSPN pipeline against a LangGraph-style 3-call pipeline on finance headlines.

## What’s included
- `evaluation/tb_finance_baseline.py` – TB-CSPN pipeline (topic extraction → normalization → deterministic policy).
- `evaluation/lg_finance_baseline.py` – LangGraph-ish pipeline (consultant → supervisor → worker).
- `evaluation/finance_taxonomy.py` – normalizes raw topic strings (e.g., “AI chip demand” → `ai_chip_demand`).
- `evaluation/finance_batch_eval.py` – batch comparison; writes `runs/finance_batch_results.csv`.
- `evaluation/finance_ablate.py` – (optional) model/temperature/policy ablations.

## Requirements
- Python 3.11–3.13
- Option A (recommended): `pip install tb-cspn-observe`  
  Option B: rely on the repo’s fallback shim under `fallback/tb_cspn_observe/`.

## Real API vs Simulation
- **Simulation (default, no cost)**
  - Windows CMD: `set TB_USE_REAL_LLM=false`
  - Linux/macOS: `export TB_USE_REAL_LLM=false`
- **Real API** (costs tokens): set `OPENAI_API_KEY`, and either unset `TB_USE_REAL_LLM` or set it to `true`.

## Quickstart (single runs)
From the repo root:
```bash
# Linux/macOS
export TB_USE_REAL_LLM=false
python evaluation/tb_finance_baseline.py
python evaluation/lg_finance_baseline.py

