# -*- coding: utf-8 -*-
"""
Runs both incident baselines and writes runs/incident_compare.jsonl and .csv
- Uses lg_incident_baseline (deterministic) and tb_incident_baseline (deterministic).
- Run with:  python evaluation\incident_compare.py
"""

from pathlib import Path
import time
import json
import pandas as pd
import sys
import os

# Make the current folder (evaluation/) importable when run as a script
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import lg_incident_baseline as lg
import tb_incident_baseline as tb

Path("runs").mkdir(exist_ok=True)

def run_lg():
    lg.ensure_sample_data()
    t0 = time.time()
    kb = lg._load_jsonl(lg.KB_PATH)
    events = lg.collect_events()
    if not events:
        return None
    ev = lg.merge_normalize(events)[0]
    ctx = lg.retrieve_context(ev, kb)
    narr = lg.summarize(ev, ctx)          # deterministic in baseline
    narr = lg.validate_schema(narr)
    narr = lg.score_severity(narr)
    res  = lg.action_plan(narr)
    res  = lg.persist(res)
    return {
        "arch": "LangGraph",
        "llm_calls": 0,  # deterministic baseline
        "processing_time": time.time() - t0,
        "confidence": res.get("severity", 0.0),
        "directive": (res.get("summary","")[:120] + "...") if len(res.get("summary","")) > 120 else res.get("summary",""),
        "action_taken": (res.get("actions", [])[:1] or ["N/A"])[0],
        "topics_extracted": {"signals": ", ".join(res.get("summary","").split("Signals: ")[-1].split(".")[0].split(", "))} if "Signals:" in res.get("summary","") else {},
    }

def run_tb():
    out = tb.run_pipeline()
    res = out["result"]
    return {
        "arch": "TB-CSPN",
        "llm_calls": out["llm_calls"],
        "processing_time": out["processing_time"],
        "confidence": res.get("severity", 0.0),
        "directive": (res.get("summary","")[:120] + "...") if len(res.get("summary","")) > 120 else res.get("summary",""),
        "action_taken": (res.get("actions", [])[:1] or ["N/A"])[0],
        "topics_extracted": {"signals": ", ".join(res.get("summary","").split("Signals: ")[-1].split(".")[0].split(", "))} if "Signals:" in res.get("summary","") else {},
    }

if __name__ == "__main__":
    rows = []
    tb_row = run_tb(); rows.append(tb_row)
    lg_row = run_lg(); rows.append(lg_row)

    # save JSONL
    with open("runs/incident_compare.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # save CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("runs/incident_compare.csv", index=False)
    print("Wrote runs/incident_compare.jsonl and runs/incident_compare.csv")
