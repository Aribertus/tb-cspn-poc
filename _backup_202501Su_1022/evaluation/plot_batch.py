# evaluation/plot_batch.py
import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "runs/batch_results.csv"
OUT1 = "runs/batch_processing_time.png"
OUT2 = "runs/batch_llm_calls.png"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Run batch_eval.py first.")

df = pd.read_csv(CSV_PATH)

# Ensure arch order is consistent in plots
if "arch" in df.columns:
    df["arch"] = pd.Categorical(df["arch"], ["TB-CSPN", "LangGraph"])

# ---- Summary table printed to console ----
summary = (
    df.groupby("arch")
      .agg(
          n=("idx", "count"),
          mean_time=("processing_time", "mean"),
          median_time=("processing_time", "median"),
          mean_calls=("llm_calls", "mean"),
          mean_conf=("confidence", "mean"),
      )
      .reset_index()
)

print("\n=== Summary by architecture ===")
with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
    print(summary.to_string(index=False))

# ---- Plot: processing time by scenario ----
pivot_time = df.pivot(index="idx", columns="arch", values="processing_time").sort_index()
ax = pivot_time.plot(kind="bar", figsize=(9, 5))
ax.set_xlabel("Scenario idx")
ax.set_ylabel("Processing time (s)")
ax.set_title("Processing time by architecture")
plt.tight_layout()
plt.savefig(OUT1, dpi=150)
plt.close()

# ---- Plot: LLM calls by scenario ----
pivot_calls = df.pivot(index="idx", columns="arch", values="llm_calls").sort_index()
ax = pivot_calls.plot(kind="bar", figsize=(9, 5))
ax.set_xlabel("Scenario idx")
ax.set_ylabel("LLM calls")
ax.set_title("LLM calls by architecture")
plt.tight_layout()
plt.savefig(OUT2, dpi=150)
plt.close()

print(f"\nSaved plots:\n  - {OUT1}\n  - {OUT2}")
