from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, Optional, List
from evaluation.benchlib import load_jsonl, write_jsonl, write_csv, summarize_rows, now_epoch
from evaluation.observe_utils import tokens_for_thread

# Import the existing baselines present in your tree
from evaluation.lg_finance_baseline import process_item as lg_proc   # type: ignore
from evaluation.tb_finance_baseline import process_item as tb_proc   # type: ignore

# OpenAI pricing (as of Oct 2025) - gpt-4o-mini
# Input: $0.150 / 1M tokens, Output: $0.600 / 1M tokens
PRICING = {
    "gpt-4o-mini": {
        "input": 0.150 / 1_000_000,   # $0.00000015 per token
        "output": 0.600 / 1_000_000,  # $0.00000060 per token
    }
}

def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost in USD for a given model and token usage.
    Returns 0.0 if model pricing is unknown.
    """
    pricing = PRICING.get(model, PRICING.get("gpt-4o-mini"))  # fallback to gpt-4o-mini
    if not pricing:
        return 0.0
    
    input_cost = prompt_tokens * pricing["input"]
    output_cost = completion_tokens * pricing["output"]
    return input_cost + output_cost

def _clean_topics_list(topics_extracted: Any) -> List[str]:
    """
    Clean and normalize topics_extracted field.
    Handles cases where LLM returns JSON wrapped in markdown code blocks.
    """
    if isinstance(topics_extracted, list):
        return topics_extracted
    
    if isinstance(topics_extracted, str):
        content = topics_extracted.strip()
        # Strip markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    
    return []

def _row(which: str, idx: int, item: Dict[str, Any], thread_prefix: str) -> Dict[str, Any]:
    tid = f"{thread_prefix}-{which.lower()}-{idx:04d}"
    t0 = time.perf_counter()
    proc = lg_proc if which == "LG" else tb_proc
    out = proc(item, thread_id=tid)  # expect keys: topics_extracted (list), raw (usage?), latency_ms?
    dt = (time.perf_counter() - t0) * 1000.0

    topics = _clean_topics_list(out.get("topics_extracted"))
    raw = out.get("raw") or {}
    
    # Try to read counters; fall back gracefully
    # Check both raw dict and root level for llm_calls
    llm_calls = int(raw.get("llm_calls") or out.get("llm_calls") or 1)
    tokens_total = raw.get("tokens_total") or out.get("tokens_total")
    
    # Extract prompt and completion tokens for cost calculation
    prompt_tokens = 0
    completion_tokens = 0
    
    # Try to get from raw directly first (LG puts them here)
    prompt_tokens = raw.get("prompt_tokens", 0) or 0
    completion_tokens = raw.get("completion_tokens", 0) or 0
    
    # If not found, try usage metadata (TB puts them here)
    if prompt_tokens == 0 and completion_tokens == 0:
        usage = raw.get("usage") or raw.get("token_usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
    
    # Calculate tokens_total if not provided
    if tokens_total is None and (prompt_tokens > 0 or completion_tokens > 0):
        tokens_total = prompt_tokens + completion_tokens
    
    # Calculate cost (if not already provided)
    cost_usd = raw.get("cost_usd")
    if cost_usd is None and (prompt_tokens > 0 or completion_tokens > 0):
        cost_usd = _calculate_cost("gpt-4o-mini", prompt_tokens, completion_tokens)
    
    latency_ms = raw.get("latency_ms") or dt

    return {
        "item_id": item.get("id") or idx,
        "system": which,
        "topic": ";".join(topics) if isinstance(topics, list) else "",
        "topics_n": len(topics) if isinstance(topics, list) else 0,
        "ok": True,
        "llm_calls": llm_calls,
        "tokens_total": tokens_total,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = load_jsonl(Path(args.dataset))
    if args.limit:
        data = data[: args.limit]

    ts = now_epoch()
    run_id = f"bench_finance_{ts}"
    out_jsonl = Path("evaluation/runs") / f"{run_id}.jsonl"
    out_csv   = Path("evaluation/runs") / f"{run_id}.csv"
    manifest  = Path("evaluation/runs") / f"{run_id}_summary.json"

    rows: List[Dict[str, Any]] = []
    thread_prefix = f"fin-{ts}"

    for i, it in enumerate(data):
        rows.append(_row("LG", i, it, thread_prefix))
        rows.append(_row("TB", i, it, thread_prefix))

    write_csv(out_csv, rows)
    write_jsonl(out_jsonl, rows)
    summary = summarize_rows(rows)
    Path(manifest).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("LG summary:", summary["LG summary"])
    print("TB summary:", summary["TB summary"])
    print("CSV:", out_csv)
    print("JSONL:", out_jsonl)
    print("Manifest:", manifest)

if __name__ == "__main__":
    main()
