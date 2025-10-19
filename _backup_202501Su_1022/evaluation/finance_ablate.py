# evaluation/finance_ablate.py
from __future__ import annotations
import csv, os, time, json
from pathlib import Path
from typing import List, Dict, Any

from evaluation import lg_finance_baseline as LG
from evaluation import tb_finance_baseline as TB

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

HEADLINES = [
    "NVIDIA surges on AI chip demand; Fed hints at easing",
    "Central bank holds rates; inflation moderates",
]

TEMPS = [0.1, 0.2, 0.3]

def _run_with_temp(mod, headline: str, temp: float) -> Dict[str, Any]:
    # override via env for each call (our helper reads env each time through openai shim)
    os.environ["TB_USE_REAL_LLM"] = "true"  # keep real LLM on if desired
    os.environ["OPENAI_TEMPERATURE_OVERRIDE"] = str(temp)  # optional, if you add support later
    return mod.process_news_item(headline)

def main():
    out_csv = RUNS_DIR / "finance_ablate_temp.csv"
    rows = []
    t0 = time.time()
    for h in HEADLINES:
        for t in TEMPS:
            lg = _run_with_temp(LG, h, t)
            tb = _run_with_temp(TB, h, t)
            for R in (lg, tb):
                rows.append({
                    "headline": h,
                    "arch": R.get("architecture"),
                    "temp": t,
                    "llm_calls": R.get("llm_calls"),
                    "processing_time": round(R.get("processing_time", 0.0), 3),
                    "confidence": round(R.get("confidence", 0.0), 3),
                    "action_taken": R.get("action_taken"),
                    "topics_extracted": json.dumps(R.get("topics_extracted", {}), ensure_ascii=False),
                })
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_csv}")
    print(f"Done in {round(time.time()-t0,2)}s")

if __name__ == "__main__":
    main()
