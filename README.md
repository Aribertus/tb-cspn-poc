
# TB-CSPN PoC — Agentic Orchestration & Evaluation

This repository hosts a proof-of-concept of **Topic-Based CSPN** (TB-CSPN) for
agentic orchestration, together with a lightweight **evaluation scaffold** that
compares TB-CSPN vs. a LangGraph (LG) baseline on two micro-benchmarks:

- **Incidents** (operational response / triage)
- **Finance** (news → topics → directive → action)

> The evaluation code is intentionally minimal and reproducible. It can run in a
> **mock mode** (no API calls) or with a **real LLM** (OpenAI) for latency/tokens.

---

## What’s in this repo

- Core logic: `agents/`, `core/`, `cpn_engine/`, `lo_engine/`
- **Evaluation scaffold** (restored):
  - `evaluation/benchlib.py` — helpers for JSONL/CSV, stats & manifest
  - `evaluation/make_graphs.py` — plots from one or more JSONL runs
  - `evaluation/observe_utils.py` — tiny I/O helpers (used by benches)
  - `evaluation/incidents/bench_incidents_tb_lg.py`
  - `evaluation/finance/bench_finance_tb_lg.py`
  - `evaluation/finance/data/` — seed datasets (you can extend)
  - `evaluation/requirements.txt` — minimal deps to run the benches
- **Logger fallback** used by both benches:
  - `tb_cspn_observe/` → `logger.py`, `__init__.py`
- **Figures output**: `docs/figures/` (created by `make_graphs.py`)

---

## Quick start (evaluation only)

> Use a fresh virtualenv; the core TB-CSPN code has its own deps, but the
> evaluation scaffold’s deps are isolated in `evaluation/requirements.txt`.

```bash
# Windows (PowerShell) or Unix
python -m venv .venv
# Windows:
. .venv/Scripts/Activate.ps1
# Unix:
source .venv/bin/activate

pip install -r evaluation/requirements.txt

