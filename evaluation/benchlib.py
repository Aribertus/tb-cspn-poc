from __future__ import annotations
import json, time, csv
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # expects rows with keys: system, topics_n, llm_calls, tokens_total, latency_ms, cost_usd
    def agg(sysname: str) -> Dict[str, Any]:
        subset = [r for r in rows if r.get("system") == sysname]
        n = len(subset)
        if not n:
            return {"n": 0}
        def avg(key): 
            vals = [r.get(key) for r in subset if isinstance(r.get(key), (int, float))]
            return sum(vals)/len(vals) if vals else None
        def pctl(key, p):
            vals = sorted([r.get(key) for r in subset if isinstance(r.get(key), (int, float))])
            if not vals: return 0.0
            i = max(0, min(len(vals)-1, int(round((p/100.0)*(len(vals)-1)))))
            return vals[i]
        return {
            "n": n,
            "topics_n_avg": avg("topics_n"),
            "directive_rate": 1.0,   # kept for compatibility
            "action_rate": 1.0,
            "llm_calls_avg": avg("llm_calls"),
            "tokens_avg": avg("tokens_total"),
            "latency_avg_ms": avg("latency_ms"),
            "latency_p50_ms": pctl("latency_ms", 50),
            "latency_p90_ms": pctl("latency_ms", 90),
            "cost_avg_usd": avg("cost_usd"),
            "cost_total_usd": sum([r.get("cost_usd", 0.0) or 0.0 for r in subset]),
        }
    return {"LG summary": agg("LG"), "TB summary": agg("TB")}

def now_epoch() -> int:
    return int(time.time())
