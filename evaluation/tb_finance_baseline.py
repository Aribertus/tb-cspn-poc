# --- Minimal TB finance baseline (deterministic & bench-friendly) ---

from __future__ import annotations
import json, time
from typing import Any, Dict, List, Optional

try:
    # Optional: if LangChain & OpenAI are installed, we’ll use them for real calls
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# Deterministic-ish defaults to mirror TB style
TB_MODEL       = "gpt-4o-mini"
TB_TEMPERATURE = 0.0
TB_TOP_P       = 1.0
TB_SEED: Optional[int] = 1234   # set to None to disable seed

FINANCE_TAXONOMY: List[str] = [
    "earnings", "mergers_acquisitions", "macroeconomy", "regulation",
    "markets", "commodities", "technology", "risk", "strategy",
]

def _extract_news(item: Any) -> str:
    """Robustly extract a text payload from various item shapes."""
    if isinstance(item, dict):
        for k in ("news", "text", "content"):
            if item.get(k):
                return str(item[k])
        # fallback: concatenate common fields
        parts = [item.get("headline"), item.get("summary"), item.get("body")]
        return " ".join([p for p in parts if p]) or json.dumps(item, ensure_ascii=False)
    return str(item)

def process_item_impl(item: Any, thread_id: str | None = None) -> Dict[str, Any]:
    """
    Real TB baseline implementation the bench will call.
    Returns a dict with keys expected by the bench:
      - topics_extracted: List[str]
      - raw: Dict (may include usage)
      - latency_ms: float
      - llm_calls: int
      - tokens_total: Optional[int]
    """
    news = _extract_news(item)
    start = time.perf_counter()

    topics: List[str] = []
    usage = None
    tokens_total = None
    llm_calls = 0

    if LLM_AVAILABLE:
        # Simple single-shot classification prompt (deterministic)
        system = SystemMessage(content=(
            "You are a precise finance topic tagger. "
            "Given a news item, output a JSON list of 3-6 topics chosen ONLY "
            f"from this taxonomy: {FINANCE_TAXONOMY}."
        ))
        human = HumanMessage(content=f"News:\n{news}\n\nRespond ONLY with a JSON list of strings.")
        llm = ChatOpenAI(
            model=TB_MODEL,
            temperature=TB_TEMPERATURE,
            top_p=TB_TOP_P, 
            seed=TB_SEED,
)       
        resp = llm.invoke([system, human])
        llm_calls = 1
        try:
            topics = json.loads(resp.content)
            if not isinstance(topics, list):
                topics = []
        except Exception:
            topics = []

        # Best-effort usage (langchain-openai may or may not expose it here)
        try:
            usage = getattr(resp, "response_metadata", {}).get("token_usage")
            if usage:
                tokens_total = usage.get("total_tokens")
        except Exception:
            usage = None
    else:
        # No LLM available: fallback heuristic (quick & deterministic)
        text = news.lower()
        if "fed" in text or "inflation" in text or "gdp" in text:
            topics.append("macroeconomy")
        if "merger" in text or "acquisition" in text or "deal" in text:
            topics.append("mergers_acquisitions")
        if "earnings" in text or "guidance" in text or "q" in text:
            topics.append("earnings")
        if "sec" in text or "regulator" in text or "policy" in text:
            topics.append("regulation")
        if not topics:
            topics.append("markets")

    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "topics_extracted": topics,
        "raw": {"usage": usage} if usage else {},
        "latency_ms": latency_ms,
        "llm_calls": llm_calls,
        "tokens_total": tokens_total,
    }

# Public entry point expected by the bench
def process_item(item: Any, thread_id: str | None = None) -> Dict[str, Any]:
    return process_item_impl(item, thread_id=thread_id)
