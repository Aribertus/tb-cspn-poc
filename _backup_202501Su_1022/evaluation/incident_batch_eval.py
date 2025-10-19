# -*- coding: utf-8 -*-
"""
Batch-compare TB-CSPN vs LangGraph incident baselines.
Writes runs/incident_batch_results.csv and prints summary.
"""
from __future__ import annotations

import csv
import time
from typing import Dict, Any, Tuple, Callable, List
from pathlib import Path
import importlib

Path("runs").mkdir(exist_ok=True)

def _load_callable(modname: str) -> Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    mod = importlib.import_module(f"evaluation.{modname}")
    if hasattr(mod, "process_incident"):
        return ("function", getattr(mod, "process_incident"))
    raise RuntimeError(f"{modname} must expose process_incident(incident: dict) -> dict")

def _synth_incidents() -> List[Dict[str, Any]]:
    return [
        {"title": "CPU spike A", "signals": ["cpu_spike"], "assets": ["cluster-A"], "timestamp": "2025-08-01T09:15:00Z"},
        {"title": "Pod restarts B", "signals": ["kube_pod_restart"], "assets": ["svc-B"], "timestamp": "2025-08-01T10:05:00Z"},
        {"title": "Disk pressure C", "signals": ["disk_io_surge"], "assets": ["db-C"], "timestamp": "2025-08-01T11:22:00Z"},
        {"title": "Latency D", "signals": ["net_latency"], "assets": ["edge-D"], "timestamp": "2025-08-01T12:40:00Z"},
        {"title": "CPU+Restart E", "signals": ["cpu_spike", "kube_pod_restart"], "assets": ["svc-E"], "timestamp": "2025-08-01T13:12:00Z"},
        {"title": "Quiet F", "signals": [], "assets": ["misc-F"], "timestamp": "2025-08-01T14:09:00Z"},
        {"title": "CPU+Disk G", "signals": ["cpu_spike", "disk_io_surge"], "assets": ["db-G"], "timestamp": "2025-08-01T15:33:00Z"},
        {"title": "Restart+Latency H", "signals": ["kube_pod_restart", "net_latency"], "assets": ["svc-H"], "timestamp": "2025-08-01T16:55:00Z"},
        {"title": "All signals I", "signals": ["cpu_spike", "kube_pod_restart", "disk_io_surge", "net_latency"], "assets": ["cluster-I"], "timestamp": "2025-08-01T18:01:00Z"},
        {"title": "Generic J", "signals": ["misc"], "assets": ["misc-J"], "timestamp": "2025-08-01T19:45:00Z"},
    ]

def _band_to_escalate(band: str) -> bool:
    return band.upper() in ("MEDIUM", "HIGH")

def main() -> None:
    kind_tb, call_tb = _load_callable("tb_incident_baseline")
    kind_lg, call_lg = _load_callable("lg_incident_baseline")
    incs = _synth_incidents()

    rows: List[Dict[str, Any]] = []
    for idx, inc in enumerate(incs, start=1):
        r_tb = call_tb(inc)
        r_lg = call_lg(inc)
        rows.append({
            "idx": idx, "arch": "TB-CSPN", "llm_calls": r_tb.get("llm_calls", 0),
            "processing_time": round(r_tb.get("processing_time", 0.0), 3),
            "severity": r_tb.get("severity"), "band": r_tb.get("band"),
            "escalate": _band_to_escalate(r_tb.get("band", "")),
            "directive": r_tb.get("summary", "")[:80],
        })
        rows.append({
            "idx": idx, "arch": "LangGraph", "llm_calls": r_lg.get("llm_calls", 0),
            "processing_time": round(r_lg.get("processing_time", 0.0), 3),
            "severity": r_lg.get("severity"), "band": r_lg.get("band"),
            "escalate": _band_to_escalate(r_lg.get("band", "")),
            "directive": r_lg.get("summary", "")[:80],
        })

    out_csv = Path("runs/incident_batch_results.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def _summ(arch: str) -> Dict[str, Any]:
        subset = [r for r in rows if r["arch"] == arch]
        n = len(subset)
        avg_t = sum(r["processing_time"] for r in subset) / n if n else 0.0
        avg_s = sum(r["severity"] for r in subset if r["severity"] is not None) / n if n else 0.0
        esc = sum(1 for r in subset if r["escalate"]) / n if n else 0.0
        avg_calls = sum(r["llm_calls"] for r in subset) / n if n else 0.0
        return {"count": n, "avg_runtime_s": round(avg_t, 3), "avg_severity": round(avg_s, 2),
                "escalation_rate": f"{esc*100:.1f}%", "avg_llm_calls": round(avg_calls, 2)}

    print("\n--- Summary (10 incidents) ---")
    print("TB-CSPN :", _summ("TB-CSPN"))
    print("LangGraph:", _summ("LangGraph"))
    print(f"\nCSV written -> {out_csv.resolve()}")

if __name__ == "__main__":
    main()
