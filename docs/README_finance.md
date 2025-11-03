# Finance Topic Classification - Evaluation Framework

This document describes the finance news topic classification benchmark, including taxonomy design, baseline implementations, and reproducible evaluation commands.

## Setup

### 1. Dependencies

Install required packages:

```bash
pip install -r evaluation/requirements.txt
```

Key dependencies:
- `langchain-openai`
- `openai`
- `matplotlib` (for graphs)

---


### 2. Environment Variables

Set OpenAI API key and enable real LLM calls:

```powershell
# PowerShell
$env:TB_USE_REAL_LLM = "true"
$env:OPENAI_API_KEY = "YOUR_KEY_HERE"

```cmd
# CMD (Command Prompt)
set OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
set TB_USE_REAL_LLM=true
```

```bash
# Bash (Linux/Mac)
export OPENAI_API_KEY="sk-proj-YOUR_KEY_HERE"
export TB_USE_REAL_LLM="true"
```



## Reproducible Commands

### Step 1: Run Benchmark

Execute both TB and LG baselines on the 50-item dataset:

```powershell
python -m evaluation.finance.bench_finance_tb_lg --dataset evaluation\finance\data\finance_seed_large.jsonl
```

***Alternative - Test only 3 item of dateset***

```powershell
python -m evaluation.finance.bench_finance_tb_lg --dataset evaluation\finance\data\finance_seed_large.jsonl --limit 3
```

**Output:**
- **CSV**: `evaluation/runs/bench_finance_TIMESTAMP.csv`
- **JSONL**: `evaluation/runs/bench_finance_TIMESTAMP.jsonl` 
- **Summary**: `evaluation/runs/bench_finance_TIMESTAMP_summary.json` 


### Step 2: Generate Visualizations

Create comparison graphs from benchmark results:

```powershell
python -m evaluation.make_graphs --jsonl evaluation\runs\bench_finance_*.jsonl --outdir docs\figures --prefix fin_
```

**Output Files** (in `docs/figures/`):
- `fin_cost_avg.png` - Average cost per item (USD)
- `fin_cost_total.png` - Total cost across dataset (USD)
- `fin_latency_avg.png` - Average latency per item (ms)
- `fin_latency_box.png` - Latency distribution box plot
- `fin_tokens_avg.png` - Average tokens per item


### Step 3(Optional): Compare topic TB (Precision/Recall)

Evaluate the topic classification quality of the TB system by comparing predictions against the gold-standard labels:

```powershell
python -m evaluation.compare_topics --pred evaluation/runs/bench_finance_*.jsonl --gold evaluation/finance/data/finance_gold.jsonl --out evaluation/runs/finance_topic_eval_tb.json
```

**Output File** (in `runs/finance_topic_eval_tb.json`):

---

## Benchmark Results

### Performance Comparison (50 items)

| Metric | TB (1-call) | LG (3-call) | TB Advantage |
|--------|-------------|-------------|--------------|
| **LLM Calls** | 1.0 | 3.0 | **3.0x fewer** |
| **Avg Tokens** | 337 | 931 | **2.8x fewer** |
| **Avg Latency** | 1,148 ms | 5,049 ms | **4.4x faster** |
| **Avg Cost** | $0.000060 | $0.000210 | **3.5x cheaper** |
| **Total Cost** | $0.003 | $0.011 | **3.7x cheaper** |



## Canonical Taxonomy

The framework uses **15 canonical finance topics** defined in `evaluation/finance_taxonomy.py`:

| Topic | Description | Examples |
|-------|-------------|----------|
| `ai_chip_demand` | AI/semiconductor chip demand | Nvidia earnings, TSMC production |
| `monetary_policy` | Central bank interest rate decisions | Fed rate cuts, ECB policy |
| `inflation` | Price level changes | CPI/PCE data, inflation reports |
| `retail_sales` | Consumer spending data | Commerce Dept retail sales |
| `consumer_sentiment` | Consumer confidence surveys | Confidence Index, sentiment data |
| `opec_production_cuts` | OPEC+ production decisions | Production cut extensions |
| `global_oil_prices` | Oil price movements | Brent/WTI crude prices |
| `loan_loss_provisions` | Bank credit risk reserves | Loan loss provision increases |
| `bank_financial_health` | Banking sector stability | Bank balance sheets, capital ratios |
| `earnings` | Corporate quarterly results | Q1/Q2/Q3/Q4 earnings reports |
| `guidance` | Forward-looking corporate statements | Earnings guidance, outlook |
| `regulation` | Regulatory actions/compliance | Antitrust, compliance costs |
| `data_portability_regulations` | Data portability rules | FTC/EU data transfer regulations |
| `big_tech_financial_impact` | Big Tech financial performance | FAANG earnings, revenue impact |
| `general_market` | Broad market conditions | Market trends, economic context |



---

## Dataset

**Location**: `evaluation/finance/data/`

- **`finance_seed_large.jsonl`**: 50 financial news items (F001-F050)
  - Fields: `id`, `title`, `summary`, `ticker`, `date`

  
- **`finance_gold.jsonl`**: Ground truth topic labels (F001-F050)
  - Fields: `id`, `topics`, `notes`





---

