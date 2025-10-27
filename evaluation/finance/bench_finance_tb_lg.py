from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, Optional, List
from evaluation.benchlib import load_jsonl, write_jsonl, write_csv, summarize_rows, now_epoch
from evaluation.observe_utils import tokens_for_thread

# Import the existing baselines present in your tree
from evaluation.lg_finance_baseline import process_item as lg_proc   # type: ignore
from evaluation.tb_finance_baseline import process_item as tb_proc   # type: ignore

def _row(which: str, idx: int, item: Dict[str, Any], thread_prefix: str) -> Dict[str, Any]:
    tid = f"{thread_prefix}-{which.lower()}-{idx:04d}"
    t0 = time.perf_counter()
    proc = lg_proc if which == "LG" else tb_proc
    out = proc(item, thread_id=tid)  # expect keys: topics_extracted (list), raw (usage?), latency_ms?
    dt = (time.perf_counter() - t0) * 1000.0

    topics = out.get("topics_extracted") or []
    raw = out.get("raw") or {}
    # Try to read counters; fall back gracefully
    llm_calls = int(raw.get("llm_calls") or 1)
    tokens_total = raw.get("tokens_total")
    if tokens_total is None:
        # best-effort from prompt/compl tokens if present
        pt = raw.get("prompt_tokens") or 0
        ct = raw.get("completion_tokens") or 0
        tokens_total = pt + ct if (pt or ct) else None
    cost_usd = raw.get("cost_usd")  # may be None
    latency_ms = raw.get("latency_ms") or dt

    return {
        "item_id": item.get("id") or idx,
        "system": which,
        "topic": ";".join(topics) if isinstance(topics, list) else "",
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
