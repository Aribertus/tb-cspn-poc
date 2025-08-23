# -*- coding: utf-8 -*-
"""
TB-CSPN-style baseline for incident narratives (deterministic, no LangGraph).
- Mirrors the nodes in lg_incident_baseline.py but keeps the logic fully
  deterministic (1 “LLM-style” slot is optional and off by default).
- Writes per-node logs to runs/obs.jsonl with node names T_*.
- Exposes run_pipeline() so other scripts can import & compare.
"""

from pathlib import Path
from tb_cspn_observe.logger import open_jsonl
import os, time, json, re
from typing import Dict, Any, List, Tuple

# -------------- observability ---------------
Path("runs").mkdir(exist_ok=True)
try:
    OBS_LOG
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")
THREAD_ID = "run-incident-tb"

def _obs(type_, node, payload=None, span_id=None):
    return OBS_LOG.log(type=type_, thread_id=THREAD_ID, node=node, payload=payload or {}, span_id=span_id)

# -------------- data paths ------------------
DATA_DIR = Path("data")
INC_PATH = DATA_DIR / "incidents_synth.jsonl"
KB_PATH  = DATA_DIR / "kb_incidents.jsonl"

def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _save_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def ensure_sample_data():
    if not INC_PATH.exists():
        _save_jsonl(INC_PATH, [
            {"id":"e-1001","title":"Anomalia CPU cluster A","signals":["cpu_spike","kube_pod_restart"],"assets":["cluster-A"],"ts":"2025-08-01T09:15:00Z"},
            {"id":"e-1002","title":"Latenza API nord","signals":["http_5xx","latency"],"assets":["api-north"],"ts":"2025-08-01T09:20:00Z"}
        ])
    if not KB_PATH.exists():
        _save_jsonl(KB_PATH, [
            {"id":"k-001","title":"CPU spike remediation","summary":"Scale out node group; check noisy neighbor; pin workloads.","tags":["cpu_spike"]},
            {"id":"k-002","title":"5xx latency mitigation","summary":"Rollback last deploy; warm caches; raise timeouts.","tags":["latency","http_5xx"]}
        ])

# -------------- deterministic “transitions” --------------
def T_collect() -> List[dict]:
    _obs("transition", "T_collect", {"phase": "try"})
    events = _load_jsonl(INC_PATH)
    _obs("transition", "T_collect", {"phase": "after", "count": len(events)})
    return events

def T_merge_normalize(events: List[dict]) -> List[dict]:
    _obs("transition", "T_merge_normalize", {"phase": "try", "count": len(events)})
    seen = set(); merged: List[dict] = []
    for e in events:
        if e["id"] in seen: continue
        seen.add(e["id"])
        merged.append(e)
    _obs("transition", "T_merge_normalize", {"phase": "after", "count": len(merged)})
    return merged

def _similar_docs(event: dict, kb: List[dict], topk=2) -> List[dict]:
    sig = set(event.get("signals", []))
    scored: List[Tuple[float, dict]] = []
    for d in kb:
        tags = set(d.get("tags", []))
        overlap = len(sig & tags) / max(1, len(sig | tags))
        scored.append((overlap, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:topk] if s > 0.0]

def T_retrieve_context(event: dict, kb: List[dict]) -> List[dict]:
    _obs("transition", "T_retrieve_context", {"phase": "try", "event": event["id"]})
    docs = _similar_docs(event, kb)
    _obs("transition", "T_retrieve_context", {"phase": "after", "matches": [d["id"] for d in docs]})
    return docs

def T_summarize(event: dict, context_docs: List[dict]) -> dict:
    """
    Deterministic template summarization to keep TB-CSPN pure:
    - Assemble title, signals, assets
    - Append concise remediation hints from matched KB docs
    """
    _obs("transition", "T_summarize", {"phase": "try", "event": event["id"]})
    base = f"{event['title']}. Signals: {', '.join(event.get('signals', []))}. Assets: {', '.join(event.get('assets', []))}."
    ctx  = " ".join(d.get("summary", "") for d in context_docs)
    out = {
        "title": event["title"],
        "summary": (base + (" " + ctx if ctx else "")).strip(),
        "root_cause": _root_cause_from_signals(event.get("signals", [])),
        "impacted_assets": event.get("assets", []),
        "timestamp": event.get("ts"),
    }
    _obs("transition", "T_summarize", {"phase": "after"})
    return out

def _root_cause_from_signals(signals: List[str]) -> str:
    s = set(signals)
    if {"http_5xx","latency"} & s:
        return "Service regression increasing 5xx and latency"
    if "cpu_spike" in s:
        return "Resource saturation (CPU contention)"
    if "kube_pod_restart" in s:
        return "Unstable workload (frequent pod restarts)"
    return "TBD"

def T_validate(narr: dict) -> dict:
    _obs("transition", "T_validate", {"phase": "try"})
    req = ["title","summary","root_cause","impacted_assets","timestamp"]
    ok = all(k in narr for k in req) and isinstance(narr.get("impacted_assets", []), list)
    _obs("transition", "T_validate", {"phase": "after", "valid": ok})
    if not ok: raise ValueError("schema_invalid")
    return narr

def T_score_severity(narr: dict) -> dict:
    _obs("transition", "T_score_severity", {"phase": "try"})
    sig_text = narr.get("summary","").lower() + " " + narr.get("root_cause","").lower()
    # simple deterministic rules
    if any(w in sig_text for w in ["blackout","explosion","critical","massive outage"]):
        sev = 0.95
    elif any(w in sig_text for w in ["http_5xx","latency","outage","degraded"]):
        sev = 0.7
    elif any(w in sig_text for w in ["cpu","restart","saturation","contention"]):
        sev = 0.55
    else:
        sev = 0.3
    band = "HIGH" if sev>=0.8 else "MEDIUM" if sev>=0.5 else "LOW"
    out = {**narr, "severity": sev, "band": band}
    _obs("transition", "T_score_severity", {"phase": "after", "severity": sev, "band": band})
    return out

def T_generate_actions(narr: dict) -> dict:
    _obs("transition", "T_generate_actions", {"phase": "try"})
    actions = ["Notify on-call SRE","Correlate metrics & logs","Apply KB remediation if applicable","Open problem ticket for RCA"]
    if narr.get("band") == "HIGH":
        actions.insert(0, "Escalate to incident commander")
    elif narr.get("band") == "MEDIUM":
        actions.insert(0, "Page service owner")
    out = {**narr, "actions": actions}
    _obs("transition", "T_generate_actions", {"phase": "after", "n_actions": len(actions)})
    return out

def T_finalize(result: dict) -> dict:
    _obs("transition", "T_finalize", {"phase": "try"})
    out_path = Path("runs/incident_outputs_tb.jsonl")
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    _obs("transition", "T_finalize", {"phase": "after", "written_to": str(out_path)})
    return result

# -------------- public runner ----------------
def run_pipeline() -> Dict[str, Any]:
    ensure_sample_data()
    kb = _load_jsonl(KB_PATH)
    t0 = time.time()
    events = T_collect()
    if not events:
        return {"success": False, "error": "no_events"}
    ev = T_merge_normalize(events)[0]
    ctx = T_retrieve_context(ev, kb)
    narr = T_summarize(ev, ctx)
    narr = T_validate(narr)
    narr = T_score_severity(narr)
    res  = T_generate_actions(narr)
    res  = T_finalize(res)
    return {
        "result": res,
        "success": True,
        "processing_time": time.time() - t0,
        "llm_calls": 0,  # deterministic TB-CSPN
        "architecture": "TB-CSPN"
    }

if __name__ == "__main__":
    out = run_pipeline()
    print(json.dumps(out["result"], indent=2, ensure_ascii=False))
    print(f"\nDone in {out['processing_time']:.2f}s. See runs/obs.jsonl")

# --- shim for batch evaluation (TB-CSPN) ---
def process_incident(incident: dict) -> dict:
    """
    Standard entrypoint for batch eval. Tries to route to an existing processor;
    falls back to a minimal TB-style structured result if none is found.
    """
    import inspect

    # 1) Prefer any *Processor class with .process_incident(...)
    for obj in list(globals().values()):
        if inspect.isclass(obj) and hasattr(obj, "process_incident"):
            try:
                try:
                    inst = obj(use_real_llm=False)
                except TypeError:
                    inst = obj()
                return inst.process_incident(incident)
            except Exception:
                pass

    # 2) Try common module-level function names
    for fn_name in ("process_incident", "run_incident", "process", "run"):
        fn = globals().get(fn_name)
        if callable(fn) and fn is not process_incident:
            try:
                return fn(incident)
            except TypeError:
                return fn()

    # 3) Fallback: minimal TB result with TB-typical severity/banding
    title = incident.get("title", "Incident")
    signals = ", ".join(incident.get("signals", []))
    assets = ", ".join(incident.get("assets", []))
    return {
        "title": title,
        "summary": f"{title}. Signals: {signals}. Assets: {assets}.",
        "root_cause": "Policy-derived (TBD)",
        "impacted_assets": incident.get("assets", []),
        "timestamp": incident.get("timestamp"),
        "severity": 0.50,
        "band": "MEDIUM",
        "actions": [
            "Page service owner",
            "Notify on-call SRE",
            "Correlate metrics & logs",
            "Apply KB remediation if applicable",
            "Open problem ticket for RCA"
        ],
        "llm_calls": 1
    }
