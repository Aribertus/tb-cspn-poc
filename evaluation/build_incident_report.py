# evaluation/build_incident_report.py
import os, io, base64, html
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("runs", "incident_batch_results.csv")
OUT_HTML = os.path.join("runs", "incident_batch_report_inline.html")

def _fig_to_data_uri(fig):
    import matplotlib
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return "data:image/png;base64," + b64

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names we rely on
    cols = {c.lower(): c for c in df.columns}
    # runtime_s vs processing_time
    if "runtime_s" not in cols and "processing_time" in cols:
        df["runtime_s"] = df[cols["processing_time"]]
    elif "runtime_s" not in cols:
        df["runtime_s"] = 0.0

    # escalation may be missing; derive from band if possible
    if "escalation" not in cols:
        if "band" in cols:
            df["escalation"] = df[cols["band"]].astype(str).str.upper().isin(["HIGH", "CRITICAL"])
        else:
            df["escalation"] = False

    # severity fallback
    if "severity" not in cols:
        df["severity"] = 0.0

    # llm_calls fallback
    if "llm_calls" not in cols:
        df["llm_calls"] = 1

    # title fallback
    if "title" not in cols:
        df["title"] = ""

    # arch must exist
    if "arch" not in cols:
        raise ValueError("CSV must contain an 'arch' column (e.g., TB-CSPN / LangGraph).")

    return df

def _summary_by_arch(df: pd.DataFrame):
    rows = []
    for arch, g in df.groupby("arch"):
        count = len(g)
        avg_runtime = float(g["runtime_s"].mean()) if count else 0.0
        avg_sev = float(g["severity"].mean()) if count else 0.0
        esc_rate = 100.0 * (g["escalation"].astype(bool).sum() / count) if count else 0.0
        avg_calls = float(g["llm_calls"].mean()) if count else 0.0
        rows.append({
            "arch": arch,
            "count": count,
            "avg_runtime_s": round(avg_runtime, 3),
            "avg_severity": round(avg_sev, 3),
            "escalation_rate_%": round(esc_rate, 1),
            "avg_llm_calls": round(avg_calls, 2),
        })
    return pd.DataFrame(rows)

def _plot_bar_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str):
    fig, ax = plt.subplots()
    ax.bar(df["arch"], df[metric])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(df[metric]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    return _fig_to_data_uri(fig)

def _plot_band_distribution(df: pd.DataFrame):
    if "band" not in df.columns:
        return None
    # Count per (arch, band)
    counts = df.pivot_table(index="band", columns="arch", values="title", aggfunc="count", fill_value=0)
    counts = counts.sort_index()
    fig, ax = plt.subplots()
    bottom = None
    for band in counts.index:
        vals = counts.loc[band].values
        if bottom is None:
            ax.bar(counts.columns, vals, label=band)
            bottom = vals
        else:
            ax.bar(counts.columns, vals, bottom=bottom, label=band)
            bottom = bottom + vals
    ax.set_title("Band distribution by architecture")
    ax.set_ylabel("Count")
    ax.legend(title="Band")
    return _fig_to_data_uri(fig)

def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = _ensure_columns(df)

    summary = _summary_by_arch(df)

    # Charts (inline as data URIs)
    dur_uri = _plot_bar_metric(summary, "avg_runtime_s", "seconds", "Average runtime (s)")
    calls_uri = _plot_bar_metric(summary, "avg_llm_calls", "calls", "Average LLM calls")
    band_uri = _plot_band_distribution(df)

    # Build HTML
    parts = []
    parts.append("<!doctype html><meta charset='utf-8'>")
    parts.append("""
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; margin:24px;}
      h1,h2{margin:0 0 8px 0}
      .section{margin:24px 0}
      table{border-collapse:collapse; width:100%}
      th,td{border:1px solid #ddd; padding:8px; font-size:14px}
      th{background:#f5f5f5; text-align:left}
      .charts img{max-width:100%; height:auto; display:block; margin:12px 0}
      .kpi{display:flex; gap:16px; flex-wrap:wrap; margin:12px 0}
      .kpi div{background:#fafafa; border:1px solid #eee; padding:8px 12px; border-radius:8px}
    </style>
    """)

    parts.append("<h1>Incident Batch Report</h1>")
    parts.append(f"<div>Source CSV: <code>{html.escape(CSV_PATH)}</code></div>")

    # KPIs (from summary if both arches present)
    parts.append("<div class='section'><h2>Summary</h2>")
    parts.append(summary.to_html(index=False))
    parts.append("</div>")

    parts.append("<div class='section charts'><h2>Charts</h2>")
    parts.append(f"<div><h3>Average runtime (s)</h3><img src='{dur_uri}' alt='avg runtime'></div>")
    parts.append(f"<div><h3>Average LLM calls</h3><img src='{calls_uri}' alt='avg calls'></div>")
    if band_uri:
        parts.append(f"<div><h3>Band distribution</h3><img src='{band_uri}' alt='band distribution'></div>")
    parts.append("</div>")

    # Detail rows table (safe subset of columns)
    safe_cols = [c for c in ["idx","title","arch","band","severity","runtime_s","llm_calls","escalation"] if c in df.columns]
    parts.append("<div class='section'><h2>Rows</h2>")
    parts.append(df[safe_cols].to_html(index=False))
    parts.append("</div>")

    html_out = "".join(parts)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Wrote {OUT_HTML}")

if __name__ == "__main__":
    main()
