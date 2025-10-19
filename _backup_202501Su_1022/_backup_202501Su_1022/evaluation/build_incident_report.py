# -*- coding: utf-8 -*-
"""
Build a plain HTML report from runs/incident_batch_results.csv (no images).
Outputs runs/incident_batch_report.html
"""
from __future__ import annotations

import csv
from pathlib import Path
from html import escape

IN_CSV = Path("runs/incident_batch_results.csv")
OUT_HTML = Path("runs/incident_batch_report.html")

def main() -> None:
    if not IN_CSV.exists():
        print(f"Input not found: {IN_CSV}")
        return

    rows = []
    with IN_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Simple summary
    tb = [x for x in rows if x["arch"] == "TB-CSPN"]
    lg = [x for x in rows if x["arch"] == "LangGraph"]
    def _summ(lst):
        n = len(lst)
        avg_t = sum(float(x["processing_time"]) for x in lst) / n if n else 0.0
        avg_s = sum(float(x["severity"]) for x in lst) / n if n else 0.0
        esc = sum(1 for x in lst if str(x["escalate"]).lower() in ("true","1")) / n if n else 0.0
        avg_calls = sum(float(x["llm_calls"]) for x in lst) / n if n else 0.0
        return n, avg_t, avg_s, esc, avg_calls
    n_tb, t_tb, s_tb, e_tb, c_tb = _summ(tb)
    n_lg, t_lg, s_lg, e_lg, c_lg = _summ(lg)

    head = """<!doctype html><html><head><meta charset="utf-8">
<title>Incident Batch Report</title>
<style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;}
table{border-collapse:collapse;width:100%;}
th,td{border:1px solid #ddd;padding:6px;font-size:14px;}
th{background:#f7f7f7;}
code{background:#f3f3f3;padding:2px 4px;border-radius:4px;}
.summary{margin-bottom:16px;}
</style></head><body>
<h1>Incident Batch Report</h1>
"""
    summary = f"""
<div class="summary">
  <p><strong>TB-CSPN</strong>: n={n_tb}, avg_runtime={t_tb:.3f}s, avg_severity={s_tb:.2f}, escalation_rate={e_tb*100:.1f}%, avg_llm_calls={c_tb:.2f}</p>
  <p><strong>LangGraph</strong>: n={n_lg}, avg_runtime={t_lg:.3f}s, avg_severity={s_lg:.2f}, escalation_rate={e_lg*100:.1f}%, avg_llm_calls={c_lg:.2f}</p>
</div>
"""

    thead = "<table><thead><tr>" + "".join(
        f"<th>{escape(col)}</th>" for col in ["idx","arch","llm_calls","processing_time","severity","band","escalate","directive"]
    ) + "</tr></thead><tbody>"

    body = []
    for r in rows:
        body.append("<tr>" + "".join(
            f"<td>{escape(str(r[col]))}</td>" for col in ["idx","arch","llm_calls","processing_time","severity","band","escalate","directive"]
        ) + "</tr>")
    tail = "</tbody></table></body></html>"

    OUT_HTML.write_text(head + summary + thead + "\n".join(body) + tail, encoding="utf-8")
    print(f"Wrote {OUT_HTML.resolve()}")

if __name__ == "__main__":
    main()
