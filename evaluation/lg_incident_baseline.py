# -*- coding: utf-8 -*-
"""
LangGraph-ish incident baseline.
- If TB_USE_REAL_LLM=true -> 3 LLM calls (consultant/supervisor/worker).
- Else -> deterministic simulated pipeline (0 calls).
Exposes: process_incident(incident: dict) -> dict
"""
from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, List
from pathlib import Path

# ---- robust logger import ----
try:
    from tb_cspn_observe.logger import open_jsonl
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fallback"))
    from tb_cspn_observe.logger import open_jsonl  # type: ignore

# ---- helpers from enhanced_fair_comparison (with fallback) ----
try:
    from evaluation.enhanced_fair_comparison import (
        with_llm_logging, _create_chat_completion, LLM_MODEL
    )
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from enhanced_fair_comparison import (  # type: ignore
        with_llm_logging, _create_chat_completion, LLM_MODEL
    )

Path("runs").mkdir(exist_ok=True)
try:
    OBS_LOG  # type: ignore[name-defined]
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")
THREAD_ID = "run-incident"

USE_REAL_LLM = (os.environ.get("TB_USE_REAL_LLM", "false").lower() == "true")


# ---------------- simulators ----------------

def _simulate_topics(incident: Dict[str, Any]) -> Dict[str, float]:
    """Cheap, deterministic features -> pseudo topic scores."""
    sigs = [s.lower() for s in incident.get("signals", [])]
    topics: Dict[str, float] = {}
    if "cpu_spike" in sigs:
        topics["resource_contention"] = 0.7
    if "kube_pod_restart" in sigs:
        topics["service_instability"] = 0.6
    if "disk_io_surge" in sigs:
        topics["storage_pressure"] = 0.5
    if not topics:
        topics["general_ops"] = 0.3
    return topics


def _simulate_directive(topics: Dict[str, float]) -> str:
    if "resource_contention" in topics and topics["resource_contention"] >= 0.7:
        return "Scale out node group; check noisy neighbor; pin workloads"
    if "service_instability" in topics and topics["service_instability"] >= 0.6:
        return "Investigate crash loops; verify rollout; rollback if needed"
    return "Correlate metrics and logs; consult KB"


def _simulate_action(directive: str) -> str:
    dl = directive.lower()
    if "scale out" in dl or "pin workload" in dl:
        return "SECTOR_ANALYSIS"  # keep action taxonomy aligned with finance
    if "rollback" in dl or "investigate" in dl:
        return "STANDARD_MONITORING"
    return "STANDARD_MONITORING"


# ---------------- real calls (optional) ----------------

def _consultant_llm(incident: Dict[str, Any]) -> Dict[str, float]:
    messages = [{
        "role": "user",
        "content": (
            "You are an SRE assistant. From this incident context, infer 2–3 root-cause topics with 0–1 scores.\n"
            "Respond ONLY JSON like {\"resource_contention\":0.7, \"service_instability\":0.6}.\n"
            f"Incident: {json.dumps(incident, ensure_ascii=False)}"
        ),
    }]
    resp = with_llm_logging(
        node_name=f"LLM_{LLM_MODEL}_inc_consultant",
        messages=messages,
        model=LLM_MODEL,
        temperature=0.2,
        call=lambda: _create_chat_completion(messages, model=LLM_MODEL, temperature=0.2, max_tokens=256),
        max_tokens=256,
    )
    msg = resp.choices[0].message
    text = msg["content"] if isinstance(msg, dict) else msg.content
    text = text.strip().strip("`").strip()
    return json.loads(text)


def _supervisor_llm(topics: Dict[str, float], incident: Dict[str, Any]) -> str:
    topics_str = ", ".join([f"{k}:{v:.2f}" for k, v in topics.items()])
    messages = [{
        "role": "user",
        "content": (
            f"Topics: {topics_str}. Provide a short directive (<= 20 words) for incident remediation."
        )
    }]
    resp = with_llm_logging(
        node_name=f"LLM_{LLM_MODEL}_inc_supervisor",
        messages=messages,
        model=LLM_MODEL,
        temperature=0.2,
        call=lambda: _create_chat_completion(messages, model=LLM_MODEL, temperature=0.2, max_tokens=128),
        max_tokens=128,
    )
    msg = resp.choices[0].message
    return (msg["content"] if isinstance(msg, dict) else msg.content).strip()


def _worker_llm(directive: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            f"Directive: '{directive}'. Choose ONE action: SECTOR_ANALYSIS or STANDARD_MONITORING or NO_ACTION."
        )
    }]
    resp = with_llm_logging(
        node_name=f"LLM_{LLM_MODEL}_inc_worker",
        messages=messages,
        model=LLM_MODEL,
        temperature=0.1,
        call=lambda: _create_chat_completion(messages, model=LLM_MODEL, temperature=0.1, max_tokens=64),
        max_tokens=64,
    )
    msg = resp.choices[0].message
    text = (msg["content"] if isinstance(msg, dict) else msg.content).upper()
    for k in ["SECTOR_ANALYSIS", "STANDARD_MONITORING", "NO_ACTION"]:
        if k in text:
            return k
    return "STANDARD_MONITORING"


# ---------------- public API ----------------

def process_incident(incident: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a normalized dict:
      title, summary, root_cause, impacted_assets, timestamp, severity [0..1], band, actions[]
    """
    t0 = time.time()
    llm_calls = 0
    title = incident.get("title") or "Incident"
    assets: List[str] = incident.get("assets", [])
    timestamp = incident.get("timestamp", "TBD")

    if not USE_REAL_LLM:
        topics = _simulate_topics(incident)
        directive = _simulate_directive(topics)
        action = _simulate_action(directive)
    else:
        topics = _consultant_llm(incident); llm_calls += 1
        directive = _supervisor_llm(topics, incident); llm_calls += 1
        action = _worker_llm(directive); llm_calls += 1

    # derive severity (simple mapping from top topic)
    sev = max(topics.values()) if topics else 0.3
    band = "LOW" if sev < 0.34 else ("MEDIUM" if sev < 0.67 else "HIGH")

    out = {
        "title": title,
        "summary": f"{title}. Signals: {', '.join(incident.get('signals', []))}. "
                   f"Assets: {', '.join(assets)}. {directive}.",
        "root_cause": "TBD",
        "impacted_assets": assets,
        "timestamp": timestamp,
        "severity": round(sev, 2),
        "band": band,
        "actions": [
            "Notify on-call SRE",
            "Correlate metrics & logs",
            "Apply known remediation from similar incidents",
            "Open problem ticket for RCA",
        ],
        "llm_calls": llm_calls,
        "architecture": "LangGraph",
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
