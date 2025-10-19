# evaluation/batch_eval.py
import json, csv, time
from pathlib import Path
from enhanced_fair_comparison import RealLLMTBCSPNProcessor, RealLLMLangGraphProcessor

HEADLINES = [
    "NVIDIA beats earnings expectations as demand for AI chips surges. Meanwhile, the Fed signals possible rate cuts amid cooling inflation.",
    "OPEC+ hints at extended production cuts as crude inventories decline.",
    "Retail sales jump unexpectedly; consumer sentiment improves to 18-month high.",
    "Major bank reports higher loan-loss provisions as delinquency rates tick up.",
    "European regulators propose new rules for big tech data portability."
]

USE_REAL_LLM = True
Path("runs").mkdir(exist_ok=True)

tb = RealLLMTBCSPNProcessor(use_real_llm=USE_REAL_LLM)
lg = RealLLMLangGraphProcessor(use_real_llm=USE_REAL_LLM)

jsonl_path = Path("runs/batch_results.jsonl")
csv_path   = Path("runs/batch_results.csv")

with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow([
        "idx", "arch", "llm_calls", "processing_time", "confidence",
        "directive", "action_taken", "topics_extracted"
    ])

    for i, text in enumerate(HEADLINES, 1):
        tb_res = tb.process_news_item(text)
        lg_res = lg.process_news_item(text)

        jf.write(json.dumps({"idx": i, "arch": "TB-CSPN", **tb_res}, ensure_ascii=False) + "\n")
        jf.write(json.dumps({"idx": i, "arch": "LangGraph", **lg_res}, ensure_ascii=False) + "\n")

        writer.writerow([i, "TB-CSPN", tb_res["llm_calls"], round(tb_res["processing_time"], 3),
                         round(tb_res["confidence"], 3), tb_res["directive"], tb_res["action_taken"],
                         json.dumps(tb_res["topics_extracted"], ensure_ascii=False)])
        writer.writerow([i, "LangGraph", lg_res["llm_calls"], round(lg_res["processing_time"], 3),
                         round(lg_res["confidence"], 3), lg_res["directive"], lg_res["action_taken"],
                         json.dumps(lg_res["topics_extracted"], ensure_ascii=False)])

print(f"Wrote {jsonl_path} and {csv_path}")
