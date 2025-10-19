# -*- coding: utf-8 -*-
"""
LangGraph-ish baseline for finance (with OBS logging).
Runs as a script or module, with robust imports and LLM logging.
"""

from __future__ import annotations
import os, time, json
from typing import Dict, Any
from pathlib import Path

# --- logger import with fallback (works even if package not installed) ---
try:
    from tb_cspn_observe.logger import open_jsonl
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fallback"))
    from tb_cspn_observe.logger import open_jsonl  # type: ignore

# --- helpers import with fallback (module vs script) ---
try:
    # when run as a module: python -m evaluation.lg_finance_baseline
    from evaluation.enhanced_fair_comparison import (
        with_llm_logging, _create_chat_completion, LLM_MODEL
    )
except ModuleNotFoundError:
    # when run as a script: python evaluation\lg_finance_baseline.py
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from enhanced_fair_comparison import (  # type: ignore
        with_llm_logging, _create_chat_completion, LLM_MODEL
    )

# --- taxonomy import with fallback (module vs script) ---
try:
    import evaluation.finance_taxonomy as ftax  # type: ignore
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import finance_taxonomy as ftax  # type: ignore

# ---------------- observability ----------------
Path("runs").mkdir(exist_ok=True)
try:
    OBS_LOG  # type: ignore[name-defined]
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")
THREAD_ID = "run-finance-lg"

USE_REAL_LLM = (os.environ.get("TB_USE_REAL_LLM", "true").lower() != "false")

def _consultant_llm(content: str) -> Dict[str, float]:
    """Extract 2–3 finance topics with scores (0–1)."""
    if not USE_REAL_LLM:
        cl = content.lower()
        topics: Dict[str, float] = {}
        if any(w in cl for w in ("ai", "nvidia", "semiconductor")):
            topics["AI chip demand"] = 0.90
        if any(w in cl for w in ("fed", "rate", "monetary")):
            topics["monetary policy"] = 0.80
        if "inflation" in cl:
            topics["inflation trends"] = 0.60
        return topics or {"general market": 0.50}

    messages = [{
        "role": "user",
        "content": (
            "Extract 2-3 descriptive finance topics with scores 0-1. "
            "Respond ONLY as JSON (no prose, no code fences). Example: "
            '{"AI chip demand": 0.92, "monetary policy": 0.80}. '
            f"News: {content}"
        ),
    }]
    resp = with_llm_logging(
        node_name=f"LG_{LLM_MODEL}_finance_consultant",
        messages=messages,
        model=LLM_MODEL,
        temperature=0.3,
        call=lambda: _create_chat_completion(
            messages, model=LLM_MODEL, temperature=0.3, max_tokens=256
        ),
        max_tokens=256,
    )
    msg = resp.choices[0].message
    text = msg["content"] if isinstance(msg, dict) else msg.content
    text = text.strip().strip("`").strip()
    return json.loads(text)

def _supervisor_llm(topics: Dict[str, float], content: str) -> str:
    """Turn topics into a brief directive."""
    if not USE_REAL_LLM:
        return "Analyze sector dynamics and macro drivers"

    topics_str = ", ".join([f"{k}: {v:.2f}" for k, v in topics.items()])
    messages = [{
        "role": "user",
        "content": f"Based on topics ({topics_str}), write a brief (<=20 words) investment directive."
    }]
    resp = with_llm_logging(
        node_name=f"LG_{LLM_MODEL}_finance_supervisor",
        messages=messages,
        model=LLM_MODEL,
        temperature=0.2,
        call=lambda: _create_chat_completion(
            messages, model=LLM_MODEL, temperature=0.2, max_tokens=128
        ),
        max_tokens=128,
    )
    msg = resp.choices[0].message
    return (msg["content"] if isinstance(msg, dict) else msg.content).strip()

def _worker_llm(directive: str) -> str:
    """Map directive to an action code."""
    if not USE_REAL_LLM:
        dl = directive.lower()
        if "ai" in dl or "semiconductor" in dl:
            return "SECTOR_ANALYSIS"
        if "defensive" in dl or "risk" in dl:
            return "DEFENSIVE_POSITIONING"
        return "STANDARD_MONITORING"

    messages = [{
        "role": "user",
        "content": (
            f"Based on this directive: '{directive}', choose ONE action: "
            "MONITOR_AI_SECTOR, DEFENSIVE_POSITIONING, SECTOR_ANALYSIS, "
            "STANDARD_MONITORING, or NO_ACTION"
        ),
    }]
    resp = with_llm_logging(
        node_name=f"LG_{LLM_MODEL}_finance_worker",
        messages=messages,
        model=LLM_MODEL,
        temperature=0.1,
        call=lambda: _create_chat_completion(
            messages, model=LLM_MODEL, temperature=0.1, max_tokens=64
        ),
        max_tokens=64,
    )
    msg = resp.choices[0].message
    text = (msg["content"] if isinstance(msg, dict) else msg.content).strip()
    for a in ("MONITOR_AI_SECTOR","DEFENSIVE_POSITIONING","SECTOR_ANALYSIS","STANDARD_MONITORING","NO_ACTION"):
        if a in text:
            return a
    return "STANDARD_MONITORING"

def process_news_item(news: str) -> Dict[str, Any]:
    """
    Entry point for batch evals (finance).
    Returns a dict with topics_extracted (normalized), directive, action_taken, etc.
    """
    t0 = time.time()
    llm_calls = 0
    try:
        raw = _consultant_llm(news); llm_calls += 1
        norm = ftax.normalize_topics(raw)

        directive = _supervisor_llm(norm, news); llm_calls += 1
        action = _worker_llm(directive); llm_calls += 1

        conf = max(norm.values()) if norm else 0.5
        return {
            "input_text": news[:100] + "..." if len(news) > 100 else news,
            "topics_extracted": norm,
            "directive": directive,
            "action_taken": action,
            "confidence": conf,
            "processing_time": time.time() - t0,
            "success": True,
            "llm_calls": llm_calls,
            "architecture": "LangGraph",
        }
    except Exception as e:
        return {
            "input_text": news[:100] + "..." if len(news) > 100 else news,
            "topics_extracted": {},
            "directive": None,
            "action_taken": None,
            "confidence": 0.0,
            "processing_time": time.time() - t0,
            "success": False,
            "error_message": str(e),
            "llm_calls": llm_calls,
            "architecture": "LangGraph",
        }

if __name__ == "__main__":
    sample = "NVIDIA beats on earnings as AI chip demand surges; Fed hints at rate cuts amid cooling inflation."
    print(json.dumps(process_news_item(sample), indent=2))
