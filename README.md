# TB-CSPN Comparative Evaluation

This directory contains the complete methodology and implementation for comparing TB-CSPN with LangGraph-style prompt chaining architectures.

## Overview

Our evaluation demonstrates TB-CSPN's architectural advantages through fair comparison where both systems use LLMs appropriately:
- **TB-CSPN**: Single LLM call for topic extraction + deterministic rule coordination
- **LangGraph**: Multiple LLM calls throughout the pipeline (consultant → supervisor → worker)

## Key Results

Based on our fair comparison evaluation:

| Metric | TB-CSPN | LangGraph | Improvement |
|--------|---------|-----------|-------------|
| Avg. Processing Time | 0.301s | 0.802s | **62.5% faster** |
| Peak Throughput | 199.5 items/min | 74.8 items/min | **166.8% higher** |
| LLM Calls per Item | 1.0 | 3.0 | **66.7% fewer** |
| Success Rate | 100.0% | 100.0% | Equal reliability |

## Files Structure

```
evaluation/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── tbcspn_poc.py                 # TB-CSPN implementation for benchmarking
├── langgraph_poc.py              # LangGraph baseline implementation
├── fair_comparison.py            # Main comparison script (VALIDATED RESULTS)
├── enhanced_fair_comparison.py   # Extended version with real LLM support
├── comparative_evaluation.py     # Original comprehensive benchmark
└── results/                      # Generated results from our evaluation
    ├── benchmark_analysis.json
    └── benchmark_report.txt
```

## Validated Implementation: fair_comparison.py

**This is the script that generated our published results.** It provides a controlled, fair comparison using:

- **Simulated LLM timing**: Realistic API latency (0.3-0.4s per call) without actual API costs
- **Deterministic topic extraction**: Consistent results for reproducible benchmarks  
- **Architectural isolation**: Compares coordination efficiency, not LLM variability

### Running the Validated Comparison

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn psutil

# Run the exact comparison that generated our paper results
python fair_comparison.py
```

Expected output:
```
Running Fair Comparison: Both Systems Using LLMs
============================================================
Average Processing Time:
  TB-CSPN:   0.301s
  LangGraph: 0.802s
  TB-CSPN is 62.5% faster

LLM Calls per Item:
  TB-CSPN:   1.0
  LangGraph: 3.0
  TB-CSPN uses 66.7% fewer LLM calls
...
```

## Extended Implementation: enhanced_fair_comparison.py

**This version supports real OpenAI API integration** for users who want to:
- Test with actual LLM variability
- Validate results with their own API keys
- Extend the evaluation to other LLM providers

### Using Real LLM Integration

```python
# Edit enhanced_fair_comparison.py and uncomment:
tb_results, lg_results = run_comprehensive_fair_comparison(
    use_real_llm=True, 
    api_key="your-openai-api-key-here"
)
```

**Note**: Real LLM usage will incur API costs (~$0.50-2.00 for full evaluation) and may show different absolute timing due to network latency, but should preserve the architectural efficiency ratios.

## Methodology: Why This Comparison is Fair

### Problem with Previous Evaluations
Many "agentic AI" comparisons are unfair because they compare:
- LLM-based systems vs. non-LLM systems (obviously LLMs are slower)
- Different semantic capabilities vs. architectural differences

### Our Fair Comparison Approach

**Both systems use LLMs for semantic understanding:**
- TB-CSPN: LLM for topic extraction (necessary semantic task)
- LangGraph: LLM for topic extraction + coordination (semantic + architectural tasks)

**Architectural difference isolated:**
- TB-CSPN: Formal coordination after semantic processing
- LangGraph: LLM-mediated coordination throughout pipeline

**Result**: The 62.5% efficiency gain demonstrates pure architectural advantage, not semantic capability differences.

## Reproducing Our Results

### Option 1: Exact Replication (Recommended)
```bash
python fair_comparison.py
```
Uses the same simulated LLM timing that generated our published results.

### Option 2: Real LLM Validation
```bash
# Edit enhanced_fair_comparison.py to add your API key
python enhanced_fair_comparison.py
```
May show different absolute times but should preserve efficiency ratios.

### Option 3: Custom Extension
Modify the test datasets, timing parameters, or evaluation metrics to explore different scenarios.

## Key Insights

### 1. Separation of Concerns
TB-CSPN's efficiency comes from architectural separation:
- **Information Acquisition**: LLMs where they excel (semantic understanding)  
- **Process Coordination**: Formal methods where they excel (deterministic logic)

### 2. Dedicated Multi-Agent Environment
LangGraph lacks a dedicated multi-agent substrate and uses LLMs for both semantic processing AND coordination management. TB-CSPN provides purpose-built coordination infrastructure.

### 3. Scalability Implications
At enterprise scale:
- TB-CSPN: ~287k items/day processing capacity
- LangGraph: ~108k items/day processing capacity

## Extending the Evaluation

Researchers can extend this evaluation by:

1. **Testing other frameworks**: AutoGen, Agentic RAG, etc.
2. **Different domains**: Healthcare, legal, technical documentation
3. **Real LLM providers**: Anthropic Claude, Google Gemini, etc.
4. **Scaling studies**: Performance under increasing load
5. **Cost analysis**: Detailed API cost comparisons

## Limitations and Future Work

### Current Limitations
- Simulated LLM timing (though calibrated to realistic latency)
- Financial news domain focus
- Limited test dataset size (30 items for validated results)

### Future Extensions
- Multi-domain evaluation datasets
- Real-world production load testing  
- Integration with enterprise knowledge bases
- Human-in-the-loop evaluation metrics

## Quickstart (Evaluation)

### Prerequisites
- Python 3.10–3.12
- Git
- (Optional) `OPENAI_API_KEY` in your environment if you run live LLM calls

### Setup
- Windows (CMD)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

- macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

- Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## Citation

If you use this evaluation methodology, please cite:

```bibtex
@article{borghoff2025agentic,
  title={Agentic AI: Debunking the Myth and Building the Concept},
  author={Borghoff, Uwe M. and Bottoni, Paolo and Pareschi, Remo},
  journal={Future Internet},
  year={2025}
}
```

## Contact

For questions about the evaluation methodology:
- Remo Pareschi: remo.pareschi@unimol.it
- GitHub Issues: [tb-cspn-poc/issues](https://github.com/Aribertus/tb-cspn-poc/issues)

