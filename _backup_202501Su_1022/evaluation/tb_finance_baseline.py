# -*- coding: utf-8 -*-
from __future__ import annotations

"""
TB-CSPN finance baseline (deterministic).
- One typed LLM call (or a simulator when TB_USE_REAL_LLM=false)
- Topic normalization via finance_taxonomy.normalize_topics
- Deterministic rules -> directive/action
"""

import os
import sys
import time
import json
from typing import Dict, Any
from pathlib import Path

# ------------------------ robust imports ------------------------

# logger (installed package or local fallback)
try:
    from tb_cspn_observe.logger import open_jsonl
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fallback"))
    from tb_cspn_observe.logger import open_jsonl  # type: ignore

# enhanced_fair_comparison helpers (module or sibling file)
try:
    from evaluation.enhanced_fair_comparison import (  # type: ignore
        with_llm_logging, _create_chat_completion, LLM_MODEL
    )
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from enhanced_fair_comparison import (  # type: ignore
        with_llm_logging, _create_chat_completion, LLM_MODEL
    )

# finance taxonomy (module or sibling file)
try:
    from evaluation.finance_taxonomy import normalize_topics  # type: ignore
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from finance_taxonomy import normalize_topics  # type: ignore

# ------------------------ setup ------------------------

Path("runs").mkdir(exist_ok=True)
try:
    OBS_LOG  # type: ignore[name-defined]
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")

THREAD_ID = "run-001"
USE_REAL_LLM = (os.environ.get("TB_USE_REAL_LLM", "true").lower() != "false")

# ------------------------ LLM / Simulator ------------------------

def _extract_topics_llm(content: str) -> Dict[str, float]:
    """
    Return raw topics (possibly unnormalized). When USE_REAL_LLM=false,
    return a cheap deterministic simulation for speed/repro.
    """
    if not USE_REAL_LLM:
        # simple simulator
        topics: Dict[str, float] = {}
        cl = content.lower()
        if any(w in cl for w in ["ai", "nvidia", "semiconductor", "chip"]):
            topics["AI chip demand"] = 0.95
        if any(w in cl for w in ["fed", "rate", "federal reserve", "policy"]):
            topics["monetary policy"] = 0.85
        if "inflation" in cl:
            topics["inflation trends"] = 0.60
        if "earnings" in cl or "beats" in cl or "misses" in cl:
            topics["earnings"] = 0.80
        return topics or {"general market": 0.60}

    system = (
        "You are a senior financial analyst. Identify the main topics and assign each a score 0–1. "
        "Respond ONLY with valid JSON as {\"topic_distribution\": {\"topic\": score}}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"News: {content}"},
    ]
    resp = with_llm_logging(
        node_name=f"TB_{LLM_MODEL}_finance_consultant",
        messages=messages,
        model=LLM_MODEL,
        temperature=0.2,
        call=lambda: _create_chat_completion(
            messages, model=LLM_MODEL, temperature=0.2, max_tokens=512
        ),
        max_tokens=512,
    )
    msg = resp.choices[0].message
    text = msg["content"] if isinstance(msg, dict) else msg.content
    text = text.strip().strip("`").strip()
    data = json.loads(text)
    return data.get("topic_distribution", {})

# ------------------------ Deterministic policy ------------------------

def _apply_rules(topics: Dict[str, float]) -> Dict[str, Any]:
    """
    topics are expected to be normalized to canonical keys like:
      ai_chip_demand, monetary_policy, inflation, earnings, ...
    """
    if not topics:
        return {
            "directive": "No actionable topics",
            "action": "NO_ACTION",
            "confidence": 0.0,
            "rule_fired": "default_rule",
        }

    ai = topics.get("ai_chip_demand", 0.0)
    mon = topics.get("monetary_policy", 0.0)
    infl = topics.get("inflation", 0.0)
    earn = topics.get("earnings", 0.0)

    if ai >= 0.80:
        return {
            "directive": "Analyze semiconductor positioning",
            "action": "SECTOR_ANALYSIS",
            "confidence": ai,
            "rule_fired": "ai_high",
        }

    if mon >= 0.70 and infl >= 0.60:
        return {
            "directive": "Tighten risk posture",
            "action": "DEFENSIVE_POSITIONING",
            "confidence": (mon + infl) / 2.0,
            "rule_fired": "macro_tighten",
        }

    if earn >= 0.75:
        return {
            "directive": "Review earnings-driven catalysts",
            "action": "SECTOR_ANALYSIS",
            "confidence": earn,
            "rule_fired": "earnings_high",
        }

    # fallback
    mx = max(topics.values())
    return {
        "directive": "Standard monitoring",
        "action": "STANDARD_MONITORING",
        "confidence": mx,
        "rule_fired": "standard",
    }

# ------------------------ Public entry point ------------------------

def process_news_item(news: str) -> Dict[str, Any]:
    t0 = time.time()
    llm_calls = 0
    try:
        raw = _extract_topics_llm(news); llm_calls += 1
        norm = normalize_topics(raw)
        decision = _apply_rules(norm)
        return {
            "input_text": (news[:100] + "...") if len(news) > 100 else news,
            "topics_extracted": norm,
            "directive": decision["directive"],
            "action_taken": decision["action"],
            "rule_fired": decision["rule_fired"],
            "confidence": decision["confidence"],
            "processing_time": time.time() - t0,
            "success": True,
            "llm_calls": llm_calls,
            "architecture": "TB-CSPN",
        }
    except Exception as e:
        return {
            "input_text": (news[:100] + "...") if len(news) > 100 else news,
            "topics_extracted": {},
            "directive": None,
            "action_taken": None,
            "rule_fired": None,
            "confidence": 0.0,
            "processing_time": time.time() - t0,
            "success": False,
            "error_message": str(e),
            "llm_calls": llm_calls,
            "architecture": "TB-CSPN",
        }

# ------------------------ CLI smoke test ------------------------

if __name__ == "__main__":
    sample = (
        "NVIDIA beats on earnings as AI chip demand surges; "
        "the Fed hints at rate cuts amid cooling inflation."
    )
    print(json.dumps(process_news_item(sample), indent=2))
