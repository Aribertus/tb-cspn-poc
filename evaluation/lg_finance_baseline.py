# evaluation/lg_finance_baseline.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

# ---- Config (env-driven, with sensible defaults)
LG_MODEL        = os.environ.get("TB_LG_MODEL", "gpt-4o-mini")
LG_TEMPERATURE  = float(os.environ.get("TB_LG_TEMPERATURE", "0.2"))
LG_TOP_P        = float(os.environ.get("TB_LG_TOP_P", "0.95"))
# Determinism: set TB_LG_SEED to an int (e.g. set TB_LG_SEED=7)
LG_SEED: Optional[int] = (
    int(os.environ["TB_LG_SEED"]) if os.environ.get("TB_LG_SEED") else None
)
# New knobs:
LG_MAX_TOKENS   = int(os.environ.get("TB_LG_MAX_TOKENS", "300"))
LG_TIMEOUT_SEC  = float(os.environ.get("TB_LG_TIMEOUT", "60"))

# Toggle real LLM vs. stub
USE_REAL_LLM    = os.environ.get("TB_USE_REAL_LLM", "false").lower() == "true"

# Optional: OpenAI key (LangChain will also read from env automatically)
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")

# --------- Optional logging helpers (no-op if package unavailable)
def _try_log_jsonl(filepath: str, obj: Dict[str, Any]) -> None:
    try:
        # prefer tb_cspn_observe.logger if present (any of the APIs we’ve seen)
        from tb_cspn_observe.logger import log_json, write_json, log_jsonl, open_jsonl  # type: ignore
        # priority: context manager if available
        if "log_jsonl" in globals():
            from pathlib import Path
            with log_jsonl(Path(filepath)) as f:  # type: ignore
                write_json(f, obj)               # type: ignore
            return
        # or open_jsonl (older helper)
        if "open_jsonl" in globals():
            with open_jsonl(filepath) as f:      # type: ignore
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            return
        # or direct single-line writer
        if "log_json" in globals():
            log_json(filepath, obj)              # type: ignore
            return
    except Exception:
        pass  # best-effort only


# --------- Real LLM (LangChain OpenAI chat) or heuristic stubs
def _chat_json(system: str, user: str) -> Dict[str, Any]:
    """
    Single JSON call with explicit params to avoid LC warnings.
    """
    from langchain_openai import ChatOpenAI  # lazy import

    kwargs: Dict[str, Any] = {
        "model": LG_MODEL,
        "temperature": LG_TEMPERATURE,
        "top_p": LG_TOP_P,
        "max_tokens": LG_MAX_TOKENS,
        "timeout": LG_TIMEOUT_SEC,
    }
    if LG_SEED is not None:
        kwargs["seed"] = LG_SEED

    llm = ChatOpenAI(**kwargs)
    # response_format requires “json” hint in messages — include it explicitly
    msgs = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "You MUST reply in strict JSON only. "
                "Return a valid JSON object (no prose). "
                "The output will be parsed by a machine."
                f"\n\n{user}"
            ),
        },
    ]
    resp = llm.invoke(msgs, response_format={"type": "json_object"})
    txt = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)
    return json.loads(txt)


# ---------- Normalization helpers
def _normalize_topics(scored: Dict[str, float]) -> Dict[str, float]:
    # strip/normalize topic keys, clamp scores into [0,1]
    out: Dict[str, float] = {}
    for k, v in scored.items():
        key = (k or "").strip()
        try:
            score = float(v)
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))
        if key:
            out[key] = score
    # sort by score desc (optional)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


# ---------- Heuristic (offline) path
def _consultant_llm_stub(content: str) -> Dict[str, float]:
    text = content.lower()
    scored: Dict[str, float] = {}
    if "nvidia" in text or "ai chip" in text:
        scored["NVIDIA earnings"] = 0.9
        scored["AI chip demand"] = 0.85
    if "fed" in text or "rate cut" in text:
        scored["Federal Reserve rate cuts"] = 0.8
    if "inflation" in text:
        scored["inflation trends"] = 0.75
    if not scored:
        scored["market news"] = 0.5
    return scored


def _supervisor_llm_stub(topics: Dict[str, float], content: str) -> str:
    if "AI chip demand" in topics:
        return "Monitor AI/semiconductors sector; assess earnings sensitivity."
    if "Federal Reserve rate cuts" in topics:
        return "Review duration risk and rate-sensitive exposures."
    return "Summarize key risks and opportunities."


def _worker_llm_stub(directive: str) -> str:
    if "Monitor AI" in directive:
        return "SECTOR_ANALYSIS"
    if "duration risk" in directive:
        return "DURATION_REVIEW"
    return "NOTE"


# ---------- Real LLM path
def _consultant_llm(content: str) -> Dict[str, float]:
    if not USE_REAL_LLM:
        return _consultant_llm_stub(content)

    system = (
        "You are a financial consultant extracting high-level topics from news. "
        "Return ONLY JSON with string keys and scores in [0,1]. Example:\n"
        '{ "NVIDIA earnings": 0.92, "AI chip demand": 0.88 }'
    )
    user = f"News text (extract finance topics as JSON):\n{content}"
    out = _chat_json(system, user)
    # tolerate either dict or wrapper with 'topics' key
    if isinstance(out, dict) and "topics" in out and isinstance(out["topics"], dict):
        return _normalize_topics(out["topics"])
    if isinstance(out, dict):
        return _normalize_topics(out)
    return {}


def _supervisor_llm(topics: Dict[str, float], content: str) -> str:
    if not USE_REAL_LLM:
        return _supervisor_llm_stub(topics, content)

    system = (
        "You are a portfolio supervisor. Read scored topics and the news, "
        "and produce ONE concise directive. Reply as JSON: {\"directive\": \"...\"}."
    )
    user = (
        f"Scored topics (JSON): {json.dumps(topics, ensure_ascii=False)}\n\n"
        f"News: {content}\n"
        "Return a JSON object with a single field 'directive'."
    )
    out = _chat_json(system, user)
    if isinstance(out, dict) and isinstance(out.get("directive"), str):
        return out["directive"].strip()
    return "Summarize key risks and opportunities."


def _worker_llm(directive: str) -> str:
    if not USE_REAL_LLM:
        return _worker_llm_stub(directive)

    system = (
        "You are a trading assistant. Map a supervisor directive to a single ACTION code. "
        "Allowed codes: WATCHLIST, NOTE, SECTOR_ANALYSIS, DURATION_REVIEW, HEDGE, REBALANCE. "
        "Reply as JSON: {\"action\": \"WATCHLIST\"}."
    )
    user = f"Directive: {directive}\nReturn JSON with key 'action'."
    out = _chat_json(system, user)
    action = out.get("action") if isinstance(out, dict) else None
    if isinstance(action, str) and action.strip():
        return action.strip().upper()
    return "NOTE"


# ---------- Public API used by the bench
def process_news_item(news: str) -> Dict[str, Any]:
    """
    Core pipeline: consultant -> supervisor -> worker.
    Returns a dict that the bench knows how to read.
    """
    t0 = time.time()
    llm_calls = 0
    try:
        raw = _consultant_llm(news)
        llm_calls += 0 if not USE_REAL_LLM else 1
        norm = _normalize_topics(raw)

        directive = _supervisor_llm(norm, news)
        llm_calls += 0 if not USE_REAL_LLM else 1

        action = _worker_llm(directive)
        llm_calls += 0 if not USE_REAL_LLM else 1

        result: Dict[str, Any] = {
            "input_text": (news[:100] + "...") if len(news) > 100 else news,
            "topics_scored": norm,
            "topics_extracted": list(norm.keys()),
            "directive": directive,
            "action_taken": action,
            "confidence": (max(norm.values()) if norm else 0.5),
            "processing_time": time.time() - t0,
            "success": True,
            "architecture": "LangGraph",
            "raw": {},
            "latency_ms": int((time.time() - t0) * 1000),
            "tokens_total": None,
            "llm_calls": llm_calls,
            "n_calls": llm_calls,
            "calls": llm_calls,
        }
        return result

    except Exception as e:
        result = {
            "input_text": (news[:100] + "...") if isinstance(news, str) and len(news) > 100 else str(news),
            "topics_scored": {},
            "topics_extracted": [],
            "directive": None,
            "action_taken": None,
            "confidence": 0.0,
            "processing_time": time.time() - t0,
            "success": False,
            "error_message": str(e),
            "architecture": "LangGraph",
            "raw": {},
            "latency_ms": int((time.time() - t0) * 1000),
            "tokens_total": None,
            "llm_calls": llm_calls,
            "n_calls": llm_calls,
            "calls": llm_calls,
        }
        return result


def process_item(item: Any, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Public entry point used by the bench. It extracts a plain-text news field
    from `item` and delegates to `process_news_item(news)`.
    """
    news = (
        (item.get("news") if isinstance(item, dict) else None)
        or (item.get("text") if isinstance(item, dict) else None)
        or (
            " ".join(
                f for f in [
                    (item.get("headline") if isinstance(item, dict) else None),
                    (item.get("summary") if isinstance(item, dict) else None),
                    (item.get("content") if isinstance(item, dict) else None),
                ] if f
            )
            if isinstance(item, dict) else None
        )
        or (str(item) if item is not None else "")
    )
    return process_news_item(news)


# ---------- Standalone smoke (python -m evaluation.lg_finance_baseline)
if __name__ == "__main__":
    demo = "NVIDIA beats on earnings as AI chip demand surges; Fed hints at rate cuts amid cooling inflation."
    out = process_news_item(demo)
    # best-effort: also emit to a jsonl file if the helper exists
    _try_log_jsonl("evaluation/runs/finance_lg_smoke.jsonl", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
