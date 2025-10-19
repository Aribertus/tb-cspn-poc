
# -*- coding: utf-8 -*-
"""
TB-CSPN-like incident baseline (deterministic; 0 LLM calls).
Exposes: process_incident(incident: dict) -> dict
"""
from __future__ import annotations

import time
import json
from typing import Dict, Any, List
from pathlib import Path

# robust logger import
try:
    from tb_cspn_observe.logger import open_jsonl
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fallback"))
    from tb_cspn_observe.logger import open_jsonl  # type: ignore

Path("runs").mkdir(exist_ok=True)
try:
    OBS_LOG  # type: ignore[name-defined]
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")
THREAD_ID = "run-incident"

def _severity_from_signals(signals: List[str]) -> float:
    sev = 0.2
    s = set([x.lower() for x in signals])
    if "cpu_spike" in s: sev += 0.25
    if "kube_pod_restart" in s: sev += 0.20
    if "disk_io_surge" in s: sev += 0.15
    if "net_latency" in s: sev += 0.10
    return max(0.0, min(1.0, sev))

def _band(sev: float) -> str:
    return "LOW" if sev < 0.34 else ("MEDIUM" if sev < 0.67 else "HIGH")

def _actions_for(sev: float) -> list[str]:
    if sev >= 0.67:
        return ["Page service owner", "Escalate incident", "Correlate metrics & logs", "Open RCA task"]
    if sev >= 0.34:
        return ["Notify on-call SRE", "Correlate metrics & logs", "Apply KB remediation if applicable", "Open problem ticket for RCA"]
    return ["Notify on-call SRE", "Monitor", "Collect additional telemetry if symptoms persist"]

def process_incident(incident: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    title = incident.get("title") or "Incident"
    signals = incident.get("signals", [])
    assets = incident.get("assets", [])
    ts = incident.get("timestamp", "TBD")

    sev = _severity_from_signals(signals)
    band = _band(sev)
    actions = _actions_for(sev)
    root = "Resource saturation (CPU contention)" if "cpu_spike" in [s.lower() for s in signals] else "TBD"

    out = {
        "title": title,
        "summary": f"{title}. Signals: {', '.join(signals)}. Assets: {', '.join(assets)}.",
        "root_cause": root,
        "impacted_assets": assets,
        "timestamp": ts,
        "severity": round(sev, 2),
        "band": band,
        "actions": actions,
        "llm_calls": 0,
        "architecture": "TB-CSPN",
        "processing_time": time.time() - t0,
        "success": True,
    }
    return out

if __name__ == "__main__":
    demo = {
        "title": "Anomalia CPU cluster A",
        "signals": ["cpu_spike", "kube_pod_restart"],
        "assets": ["cluster-A"],
        "timestamp": "2025-08-01T09:15:00Z",
    }
    print(json.dumps(process_incident(demo), indent=2, ensure_ascii=False))
