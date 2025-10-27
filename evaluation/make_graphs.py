from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from glob import glob as _glob

def _resolve_jsonl(pathish: Path) -> Path:
    s = str(pathish)
    if any(ch in s for ch in "*?"):
        matches = sorted(_glob(s))
        if not matches:
            raise FileNotFoundError(f"No files match pattern: {s}")
        # pick the latest by mtime
        latest = max(matches, key=lambda p: Path(p).stat().st_mtime)
        return Path(latest)
    return pathish

def _load_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _annotate_bars(ax, values, money=False):
    for i, v in enumerate(values):
        text = f"${v:,.5f}" if (money and v is not None) else (f"{v:.2f}" if v is not None else "n/a")
        offset = 3
        ax.text(i, (v or 0.0), text, ha="center", va="bottom", fontsize=9)
        
def make_figures(jsonl_path: Path, outdir: Path, prefix: str = ""):
    rows = _load_rows(_resolve_jsonl(jsonl_path))
    lg = [r for r in rows if r.get("system") == "LG"]
    tb = [r for r in rows if r.get("system") == "TB"]

    def avg(lst, key):
        vals = [x.get(key) for x in lst if isinstance(x.get(key), (int, float))]
        return sum(vals)/len(vals) if vals else 0.0

    lg_cost_avg = avg(lg, "cost_usd"); tb_cost_avg = avg(tb, "cost_usd")
    lg_cost_tot = sum([x.get("cost_usd", 0.0) or 0.0 for x in lg]); tb_cost_tot = sum([x.get("cost_usd", 0.0) or 0.0 for x in tb])
    lg_lat = [x.get("latency_ms", 0.0) for x in lg]; tb_lat = [x.get("latency_ms", 0.0) for x in tb]
    lg_lat_avg = avg(lg, "latency_ms"); tb_lat_avg = avg(tb, "latency_ms")
    lg_tok_avg = avg(lg, "tokens_total"); tb_tok_avg = avg(tb, "tokens_total")

    outdir.mkdir(parents=True, exist_ok=True)

    # Cost avg
    fig, ax = plt.subplots()
    ax.bar(["LG","TB"], [lg_cost_avg, tb_cost_avg])
    ax.set_title("Average cost per item (USD)")
    _annotate_bars(ax, [lg_cost_avg, tb_cost_avg], money=True)
    p = outdir / f"{prefix}cost_avg.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")

    # Cost total
    fig, ax = plt.subplots()
    ax.bar(["LG","TB"], [lg_cost_tot, tb_cost_tot])
    ax.set_title("Total cost (USD)")
    _annotate_bars(ax, [lg_cost_tot, tb_cost_tot], money=True)
    p = outdir / f"{prefix}cost_total.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")

    # Latency avg
    fig, ax = plt.subplots()
    ax.bar(["LG","TB"], [lg_lat_avg, tb_lat_avg])
    ax.set_title("Average latency (ms)")
    _annotate_bars(ax, [lg_lat_avg, tb_lat_avg], money=False)
    p = outdir / f"{prefix}latency_avg.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")

    # Latency box
    fig, ax = plt.subplots()
    ax.boxplot([lg_lat, tb_lat], tick_labels=["LG", "TB"], showfliers=True)
    ax.set_title("Latency distribution (ms)")
    p = outdir / f"{prefix}latency_box.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")

    # Tokens avg
    fig, ax = plt.subplots()
    ax.bar(["LG","TB"], [lg_tok_avg, tb_tok_avg])
    ax.set_title("Average tokens per item")
    _annotate_bars(ax, [lg_tok_avg, tb_tok_avg], money=False)
    p = outdir / f"{prefix}tokens_avg.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print(f"Saved: {p}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    make_figures(Path(args.jsonl), Path(args.outdir), prefix=args.prefix)
