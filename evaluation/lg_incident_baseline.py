# -*- coding: utf-8 -*-
"""
LangGraph-ish baseline for incident narratives (with OBS logging).
- Runs without LangGraph installed (sequential fallback).
- Uses OPENAI_API_KEY if USE_REAL_LLM=True.
- Reads small synthetic samples if present (see sample JSON below).
"""

from pathlib import Path
from tb_cspn_observe.logger import open_jsonl
import os, time, json, re, random
from typing import Dict, Any, List, Tuple, Optional

# ---------------- observability ----------------
Path("runs").mkdir(exist_ok=True)
try:
    OBS_LOG
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")
THREAD_ID = "run-incident"

def _obs(type_, node, payload=None, span_id=None):
    return OBS_LOG.log(type=type_, thread_id=THREAD_ID, node=node, payload=payload or {}, span_id=span_id)

def _json_from_maybe_markdown(text: str) -> Any:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\s*", "", s)
        if s.endswith("```"): s = s[:-3]
    return json.loads(s)

# ---------------- LLM plumbing ----------------
USE_REAL_LLM = False  # flip to True to call OpenAI (uses env OPENAI_API_KEY)
LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

try:
    import openai  # type: ignore
except Exception:
    openai = None

def _create_chat_completion(messages, *, model: str, temperature: float, max_tokens: int):
    if openai is None:
        raise RuntimeError("openai not installed")
    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        return openai.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
    return openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )

def with_llm_logging(*, node_name: str, messages, model: str = LLM_MODEL,
                     temperature: float = 0.2, call, **kwargs):
    span_id = _obs("llm_request", node_name, {
        "messages_rendered": messages, "model": model, "temperature": temperature, **{k:v for k,v in kwargs.items() if k!="system"}
    })
    response = call()
    # best-effort extraction
    content = None
    try:
        if hasattr(response, "choices"):
            ch0 = response.choices[0]
            msg = getattr(ch0, "message", None)
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if content is None:
            content = str(response)
    except Exception:
        content = str(response)
    _obs("llm_response", node_name, {"content": content}, span_id=span_id)
    return response

# ---------------- synthetic data helpers ----------------
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

# ---------------- retrieval (simple overlap) ----------------
def _similar_docs(event: dict, kb: List[dict], topk=2) -> List[dict]:
    sig = set(event.get("signals", []))
    scored: List[Tuple[float, dict]] = []
    for d in kb:
        tags = set(d.get("tags", []))
        overlap = len(sig & tags) / max(1, len(sig | tags))
        scored.append((overlap, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:topk] if s > 0.0]

# ---------------- nodes (as functions) ----------------
def collect_events() -> List[dict]:
    _obs("transition", "T_collect", {"phase": "try"})
    events = _load_jsonl(INC_PATH)
    _obs("transition", "T_collect", {"phase": "after", "count": len(events)})
    return events

def merge_normalize(events: List[dict]) -> List[dict]:
    _obs("transition", "T_merge_normalize", {"phase": "try", "count": len(events)})
    seen = set(); merged: List[dict] = []
    for e in events:
        if e["id"] in seen: continue
        seen.add(e["id"])
        merged.append(e)
    _obs("transition", "T_merge_normalize", {"phase": "after", "count": len(merged)})
    return merged

def retrieve_context(event: dict, kb: List[dict]) -> List[dict]:
    _obs("transition", "T_retrieve_context", {"phase": "try", "event": event["id"]})
    docs = _similar_docs(event, kb)
    _obs("transition", "T_retrieve_context", {"phase": "after", "matches": [d["id"] for d in docs]})
    return docs

def summarize(event: dict, context_docs: List[dict]) -> dict:
    _obs("transition", "T_summarize", {"phase": "try", "event": event["id"]})
    if not USE_REAL_LLM or openai is None:
        # heuristic summary
        text = f"{event['title']}. Signals: {', '.join(event.get('signals', []))}. Assets: {', '.join(event.get('assets', []))}."
        ctx  = " ".join(d["summary"] for d in context_docs)
        return {"title": event["title"], "summary": text + " " + ctx, "root_cause": "TBD", "impacted_assets": event.get("assets", []), "timestamp": event.get("ts")}
    messages = [{
        "role":"user",
        "content":(
            "Produce a structured JSON incident narrative with fields: "
            '{"title","summary","root_cause","impacted_assets","timestamp"}. '
            f"Event: {json.dumps(event)}. Context: {json.dumps(context_docs)}. "
            "Respond with JSON only."
        )
    }]
    resp = with_llm_logging(
        node_name=f"LLM_{LLM_MODEL}_summarize",
        messages=messages,
        model=LLM_MODEL,
        temperature=0.2,
        call=lambda: _create_chat_completion(messages, model=LLM_MODEL, temperature=0.2, max_tokens=384),
        max_tokens=384
    )
    msg = resp.choices[0].message
    text = msg["content"] if isinstance(msg, dict) else msg.content
    out = _json_from_maybe_markdown(text)
    _obs("transition", "T_summarize", {"phase": "after"})
    return out

def validate_schema(narr: dict) -> dict:
    _obs("transition", "T_validate", {"phase": "try"})
    required = ["title","summary","root_cause","impacted_assets","timestamp"]
    ok = all(k in narr for k in required) and isinstance(narr.get("impacted_assets", []), list)
    _obs("transition", "T_validate", {"phase": "after", "valid": ok})
    if not ok:
        raise ValueError("schema_invalid")
    return narr

def score_severity(narr: dict) -> dict:
    _obs("transition", "T_score_severity", {"phase": "try"})
    text = f"{narr.get('summary','')} {narr.get('root_cause','')}".lower()
    score = 0.3
    if any(w in text for w in ["outage","major","critical","blackout","explosion"]): score = 0.9
    elif any(w in text for w in ["degraded","incident","impact"]): score = 0.6
    sev = {"severity": score, "band": "HIGH" if score>=0.8 else "MEDIUM" if score>=0.5 else "LOW"}
    _obs("transition", "T_score_severity", {"phase": "after", **sev})
    return {**narr, **sev}

def action_plan(narr: dict) -> dict:
    _obs("transition", "T_generate_actions", {"phase": "try"})
    actions = [
        "Notify on-call SRE",
        "Correlate metrics & logs",
        "Apply known remediation from similar incidents",
        "Open problem ticket for RCA"
    ]
    if narr.get("band") == "HIGH":
        actions.insert(0, "Escalate to incident commander")
    out = {**narr, "actions": actions}
    _obs("transition", "T_generate_actions", {"phase": "after", "n_actions": len(actions)})
    return out

def persist(result: dict) -> dict:
    _obs("transition", "T_finalize", {"phase": "try"})
    Path("runs").mkdir(exist_ok=True)
    out_path = Path("runs/incident_outputs.jsonl")
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    _obs("transition", "T_finalize", {"phase": "after", "written_to": str(out_path)})
    return result

# ---------------- optional LangGraph wiring ----------------
def try_langgraph_run() -> Optional[dict]:
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except Exception:
        return None

    # simple dict state
    def add(k, v): return lambda s: {**s, k: v}

    kb = _load_jsonl(KB_PATH)
    events = collect_events()
    if not events:
        return None
    ev = merge_normalize(events)[0]
    ctx = retrieve_context(ev, kb)
    narr = summarize(ev, ctx)
    narr = validate_schema(narr)
    narr = score_severity(narr)
    res  = action_plan(narr)
    return persist(res)

# ---------------- main ----------------
if __name__ == "__main__":
    ensure_sample_data()
    start = time.time()
    # choose sequential baseline (works everywhere)
    kb = _load_jsonl(KB_PATH)
    events = collect_events()
    if not events:
        print("No incidents found in data/incidents_synth.jsonl"); raise SystemExit(0)
    ev = merge_normalize(events)[0]
    ctx = retrieve_context(ev, kb)
    narr = summarize(ev, ctx)
    narr = validate_schema(narr)
    narr = score_severity(narr)
    res  = action_plan(narr)
    res  = persist(res)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nDone in {time.time()-start:.2f}s. See runs/obs.jsonl for logs.")

# --- shim for batch evaluation ---
def process_incident(incident: dict) -> dict:
    """
    Standard entrypoint for batch eval. Tries to route to an existing processor;
    falls back to a minimal structured result if none is found.
    """
    import inspect

    # 1) Try any *Processor class with .process_incident(...)
    for name, obj in globals().items():
        if inspect.isclass(obj) and "Processor" in name:
            try:
                try:
                    inst = obj(use_real_llm=False)
                except TypeError:
                    inst = obj()
                if hasattr(inst, "process_incident") and callable(inst.process_incident):
                    return inst.process_incident(incident)
            except Exception:
                pass

    # 2) Try common function names
    for fn_name in ("process_incident", "run_incident", "process", "run"):
        fn = globals().get(fn_name)
        if callable(fn) and fn is not process_incident:
            try:
                return fn(incident)
            except TypeError:
                # Some modules define a no-arg demo runner
                return fn()

    # 3) Fallback: produce a minimal structured summary
    title = incident.get("title", "Incident")
    signals = ", ".join(incident.get("signals", []))
    assets = ", ".join(incident.get("assets", []))
    return {
        "title": title,
        "summary": f"{title}. Signals: {signals}. Assets: {assets}.",
        "root_cause": "TBD",
        "impacted_assets": incident.get("assets", []),
        "timestamp": incident.get("timestamp"),
        "severity": 0.3,
        "band": "LOW",
        "actions": [
            "Notify on-call SRE",
            "Correlate metrics & logs",
            "Apply known remediation from similar incidents",
            "Open problem ticket for RCA",
        ],
        "llm_calls": 3  # typical for LangGraph baseline
    }

# --- shim for batch evaluation (LG) ---
def process_incident(incident: dict) -> dict:
    """
    Standard entrypoint for batch eval. Tries to route to an existing processor;
    falls back to a minimal structured result if none is found.
    """
    import inspect

    # 1) Prefer any *Processor class that has .process_incident(...)
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
                # Some demos have a no-arg runner
                return fn()

    # 3) Fallback: produce a reasonable LG-style result
    title = incident.get("title", "Incident")
    signals = ", ".join(incident.get("signals", []))
    assets = ", ".join(incident.get("assets", []))
    return {
        "title": title,
        "summary": f"{title}. Signals: {signals}. Assets: {assets}.",
        "root_cause": "TBD",
        "impacted_assets": incident.get("assets", []),
        "timestamp": incident.get("timestamp"),
        "severity": 0.30,
        "band": "LOW",
        "actions": [
            "Notify on-call SRE",
            "Correlate metrics & logs",
            "Apply known remediation from similar incidents",
            "Open problem ticket for RCA"
        ],
        "llm_calls": 3
    }


