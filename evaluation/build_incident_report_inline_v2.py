# evaluation/build_incident_report_inline_v2.py
import argparse, base64, io, os, sys
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

def _img_b64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _bar(df, x, y, title):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    df.plot(kind="bar", x=x, y=y, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xlabel("")
    return _img_b64(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("runs", "incident_batch_results.csv"))
    ap.add_argument("--out", default=os.path.join("runs", "incident_batch_report_inline.html"))
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    nrows, ncols = df.shape

    # Basic summaries (robust to whatever columns exist)
    grp = df.groupby("arch", dropna=False)
    summaries = []
    for arch, g in grp:
        rec = {"arch": arch, "count": len(g)}
        if "runtime_s" in g.columns:
            rec["avg_runtime_s"] = round(g["runtime_s"].mean(), 3)
        if "severity" in g.columns:
            rec["avg_severity"] = round(g["severity"].mean(), 3)
        if "llm_calls" in g.columns:
            rec["avg_llm_calls"] = round(g["llm_calls"].mean(), 3)
        # escalation rate if present as bool or 0/1
        if "escalate" in g.columns:
            try:
                rec["escalation_rate_%"] = round(100 * g["escalate"].astype(float).mean(), 1)
            except Exception:
                pass
        summaries.append(rec)
    df_sum = pd.DataFrame(summaries)

    # Charts only if the fields exist
    charts = []
    if "runtime_s" in df.columns:
        charts.append(("Average runtime (s)", _bar(df_sum.fillna(0), "arch", "avg_runtime_s", "Average runtime (s)")))
    if "severity" in df.columns:
        charts.append(("Average severity", _bar(df_sum.fillna(0), "arch", "avg_severity", "Average severity")))
    if "llm_calls" in df.columns:
        charts.append(("Average LLM calls", _bar(df_sum.fillna(0), "arch", "avg_llm_calls", "Average LLM calls")))

    # Order columns for the rows table (fall back to all)
    preferred = [c for c in [
        "title","arch","band","severity","escalate","action_taken","directive",
        "runtime_s","llm_calls","confidence","topics_extracted","timestamp","idx"
    ] if c in df.columns]
    if preferred:
        df_show = df[preferred].copy()
    else:
        df_show = df.copy()

    # Limit very long text fields a bit so the table is readable
    def _trim(x, maxlen=180):
        if isinstance(x, str) and len(x) > maxlen:
            return x[:maxlen-1] + "…"
        return x
    for col in df_show.columns:
        df_show[col] = df_show[col].map(_trim)

    # HTML pieces
    style = """
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; margin:24px;}
      h1,h2{margin:0 0 8px 0}
      .section{margin:24px 0}
      table{border-collapse:collapse; width:100%}
      th,td{border:1px solid #ddd; padding:8px; font-size:14px; vertical-align:top}
      th{background:#f5f5f5; text-align:left}
      .charts img{max-width:100%; height:auto; display:block; margin:12px 0}
      .meta{color:#666; font-size:12px; margin-bottom:8px}
      code{background:#f7f7f7; padding:1px 4px; border-radius:4px}
    </style>
    """

    summary_html = df_sum.to_html(index=False, border=1, classes="dataframe")
    rows_html = df_show.to_html(index=False, border=1, classes="dataframe")

    charts_html = ""
    if charts:
        charts_html = "<div class='section charts'><h2>Charts</h2>" + "".join(
            f"<div><h3>{title}</h3><img src='{src}'></div>" for title, src in charts
        ) + "</div>"

    html = f"""<!doctype html><meta charset='utf-8'>
    {style}
    <h1>Incident Batch Report</h1>
    <div class='meta'>Generated: {datetime.utcnow().isoformat()}Z — Source CSV: <code>{args.csv}</code> — Rows: {nrows}, Cols: {ncols}</div>
    <div class='section'>
      <h2>Summary</h2>
      {summary_html}
    </div>
    {charts_html}
    <div class='section'>
      <h2>Rows</h2>
      {rows_html}
    </div>
    """

    # Ensure runs/ exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote report to: {os.path.abspath(args.out)}")
    print(f"Rows rendered: {len(df_show)} (columns: {list(df_show.columns)})")

if __name__ == "__main__":
    main()
