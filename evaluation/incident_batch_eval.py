# evaluation/incident_batch_eval.py
# Batch-compare TB-CSPN vs LangGraph on synthetic incident patterns.
# Writes runs/incident_batch_results.csv and prints a short summary.

from __future__ import annotations
import importlib.util
import json
import time
import csv
import re
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)

def load_module_by_path(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load the two baseline scripts directly by file path to avoid PYTHONPATH issues.
LG = load_module_by_path("lg_incident_baseline", HERE / "lg_incident_baseline.py")
TB = load_module_by_path("tb_incident_baseline", HERE / "tb_incident_baseline.py")

# --- Call adapters ------------------------------------------------------------

def _find_callable(mod):
    """
    Try common entry points that return a dict when passed an 'incident' dict.
    Falls back to Processor.* if present. Raises if nothing reasonable is found.
    """
    # 1) Preference: module-level function taking an incident dict.
    candidates = ["process_incident", "run_incident", "process", "run"]
    for name in candidates:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                return ("function", fn)

    # 2) Look for a *Processor class with a process_incident method.
    for attr in dir(mod):
        if "Processor" in attr:
            cls = getattr(mod, attr)
            try:
                try:
                    inst = cls(use_real_llm=False)  # prefer cheap mode if supported
                except TypeError:
                    inst = cls()
                if hasattr(inst, "process_incident") and callable(inst.process_incident):
                    return ("method", inst.process_incident)
            except Exception:
                continue

    raise RuntimeError(
        f"Could not find a callable in {mod.__name__}. "
        f"Please expose `process_incident(incident: dict) -> dict`."
    )

CALL_KIND_LG, LG_CALL = _find_callable(LG)
CALL_KIND_TB, TB_CALL = _find_callable(TB)

# If your callables don't accept an incident param, we’ll detect and wrap.
def _call_with_incident(callable_obj, incident: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return callable_obj(incident)
    except TypeError:
        # Try no-arg call (module may use an internal default incident)
        return callable_obj()

# --- Synthetic batch of incidents --------------------------------------------

INCIDENTS: List[Dict[str, Any]] = [
    {
        "title": "CPU spike pods restart",
        "signals": ["cpu_spike", "kube_pod_restart"],
        "assets": ["cluster-A"],
        "timestamp": "2025-08-01T09:15:00Z",
    },
    {
        "title": "DB slow queries & pool exhaustion",
        "signals": ["db_slow_queries", "db_pool_exhaustion", "p95_latency_up"],
        "assets": ["db-prod-1"],
        "timestamp": "2025-08-01T10:40:00Z",
    },
    {
        "title": "Memory leak with OOMKills",
        "signals": ["rss_rising", "oom_kills", "pod_restarts"],
        "assets": ["svc-payments"],
        "timestamp": "2025-08-01T12:05:00Z",
    },
    {
        "title": "Network packet loss & errors",
        "signals": ["packet_loss", "rx_errors", "tcp_retransmits"],
        "assets": ["edge-router-3"],
        "timestamp": "2025-08-01T13:20:00Z",
    },
    {
        "title": "Web 5xx surge",
        "signals": ["http_5xx_surge", "p99_latency_up", "error_budget_burn"],
        "assets": ["web-frontend"],
        "timestamp": "2025-08-01T14:55:00Z",
    },
    {
        "title": "Kafka consumer lag spike",
        "signals": ["kafka_consumer_lag", "throughput_drop"],
        "assets": ["streaming-pipeline"],
        "timestamp": "2025-08-01T15:40:00Z",
    },
    {
        "title": "Disk IO saturation",
        "signals": ["disk_io_saturation", "queue_depth_up"],
        "assets": ["storage-1"],
        "timestamp": "2025-08-01T16:10:00Z",
    },
    {
        "title": "SSL cert nearing expiry",
        "signals": ["tls_cert_expiry_soon"],
        "assets": ["api-gateway"],
        "timestamp": "2025-08-01T18:00:00Z",
    },
    {
        "title": "Auth failures spike",
        "signals": ["auth_failed_logins_spike", "rate_limit_hits"],
        "assets": ["auth-service"],
        "timestamp": "2025-08-01T19:35:00Z",
    },
    {
        "title": "Cloud throttling",
        "signals": ["cloud_throttling", "quota_exceeded"],
        "assets": ["batch-workers"],
        "timestamp": "2025-08-01T20:20:00Z",
    },
]

# --- Metrics helpers ----------------------------------------------------------

ESCALATION_KEYWORDS = (
    "page", "escalate", "notify on-call", "notify oncall", "page service owner"
)
HIGH_BANDS = {"HIGH", "CRITICAL"}

def is_escalated(result: Dict[str, Any]) -> bool:
    # Band-based
    band = str(result.get("band", "")).upper()
    if band in HIGH_BANDS:
        return True
    # Actions-based
    actions = result.get("actions", []) or []
    text = " ".join([a.lower() for a in actions])
    return any(kw in text for kw in ESCALATION_KEYWORDS)

def get_processing_time(result: Dict[str, Any], measured: float) -> float:
    # Prefer result-provided processing_time; else measured wall time.
    rt = result.get("processing_time")
    try:
        return float(rt) if rt is not None else measured
    except Exception:
        return measured

def get_llm_calls(result: Dict[str, Any], default_calls: int) -> int:
    val = result.get("llm_calls")
    try:
        return int(val) if val is not None else default_calls
    except Exception:
        return default_calls

# --- Run batch ----------------------------------------------------------------

def run_one(callable_obj, incident: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    out = _call_with_incident(callable_obj, incident) or {}
    dt = time.time() - t0
    out.setdefault("title", incident.get("title"))
    out.setdefault("timestamp", incident.get("timestamp"))
    out["processing_time_measured"] = dt
    return out

def main():
    rows = []
    # Default LLM-call assumptions if the result doesn't report it:
    DEFAULT_TB_CALLS = 1
    DEFAULT_LG_CALLS = 3

    for idx, inc in enumerate(INCIDENTS, start=1):
        # TB-CSPN
        tb_res = run_one(TB_CALL, inc)
        tb_pt = get_processing_time(tb_res, tb_res["processing_time_measured"])
        tb_llm = get_llm_calls(tb_res, DEFAULT_TB_CALLS)
        tb_row = {
            "idx": idx,
            "arch": "TB-CSPN",
            "llm_calls": tb_llm,
            "processing_time": round(tb_pt, 3),
            "severity": tb_res.get("severity", ""),
            "band": tb_res.get("band", ""),
            "escalated": is_escalated(tb_res),
            "directive": tb_res.get("summary", tb_res.get("directive", "")),
            "action_taken": "; ".join(tb_res.get("actions", []) or []),
            "topics_extracted": json.dumps(tb_res.get("topics_extracted", {}), ensure_ascii=False),
            "title": tb_res.get("title", ""),
        }
        rows.append(tb_row)

        # LangGraph
        lg_res = run_one(LG_CALL, inc)
        lg_pt = get_processing_time(lg_res, lg_res["processing_time_measured"])
        lg_llm = get_llm_calls(lg_res, DEFAULT_LG_CALLS)
        lg_row = {
            "idx": idx,
            "arch": "LangGraph",
            "llm_calls": lg_llm,
            "processing_time": round(lg_pt, 3),
            "severity": lg_res.get("severity", ""),
            "band": lg_res.get("band", ""),
            "escalated": is_escalated(lg_res),
            "directive": lg_res.get("summary", lg_res.get("directive", "")),
            "action_taken": "; ".join(lg_res.get("actions", []) or []),
            "topics_extracted": json.dumps(lg_res.get("topics_extracted", {}), ensure_ascii=False),
            "title": lg_res.get("title", ""),
        }
        rows.append(lg_row)

    # Write CSV
    out_csv = RUNS / "incident_batch_results.csv"
    fieldnames = [
        "idx", "arch", "llm_calls", "processing_time", "severity", "band",
        "escalated", "directive", "action_taken", "topics_extracted", "title"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Console summary
    def _stats(arch: str):
        subset = [r for r in rows if r["arch"] == arch]
        n = len(subset)
        avg_rt = sum(float(r["processing_time"]) for r in subset) / n if n else 0.0
        sev_vals = [float(r["severity"]) for r in subset if isinstance(r["severity"], (int, float)) or re.match(r"^\d+(\.\d+)?$", str(r["severity"])) ]
        avg_sev = (sum(map(float, sev_vals)) / len(sev_vals)) if sev_vals else None
        esc = sum(1 for r in subset if r["escalated"])
        esc_rate = esc / n if n else 0.0
        avg_calls = sum(int(r["llm_calls"]) for r in subset) / n if n else 0.0
        return {
            "count": n,
            "avg_runtime_s": round(avg_rt, 3),
            "avg_severity": round(avg_sev, 3) if avg_sev is not None else "n/a",
            "escalation_rate": f"{round(esc_rate*100,1)}%",
            "avg_llm_calls": round(avg_calls, 2),
        }

    tb_stats = _stats("TB-CSPN")
    lg_stats = _stats("LangGraph")

    print("\nWrote", out_csv)
    print("\n--- Summary (10 incidents) ---")
    print("TB-CSPN  :", tb_stats)
    print("LangGraph:", lg_stats)
    print("\nTip: open the CSV in Excel or Pandas to slice by title/band/escalation.")

if __name__ == "__main__":
    main()
