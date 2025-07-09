
# TB-CSPN: Topic-Based Communication Space Petri Net Framework

This repository contains the reference implementation of the **TB-CSPN (Topic-Based Communication-Space Petri Net)** framework, a formal architecture for coordinating multi-agent systems through topic-based communication and Petri net semantics.

## Overview

TB-CSPN demonstrates key principles for next-generation agentic AI by providing a dedicated multi-agent coordination substrate that separates semantic understanding from process orchestration. The framework models semantic coordination among modular agents via threshold-based token propagation across three communication layers: **Surface** (strategic), **Observation** (semantic), and **Computation** (operational).

## Key Features

- **Formal Coordination**: Colored Petri Net semantics ensure verifiable agent behavior
- **Topic-Based Communication**: Semantic interlingua enables heterogeneous agent coordination  
- **Human-AI Collaboration**: Native support for centaurian architectures and human oversight
- **Efficient LLM Integration**: Strategic LLM usage restricted to semantic understanding tasks
- **Multi-Engine Architecture**: Rule-based, CPN, and SNAKES engines for different deployment needs

## Architecture Components

### Core Agent Types
- **Consultant Agents**: LLM-powered semantic processors that transform unstructured input into topic-annotated tokens
- **Supervisor Agents**: Human decision-makers (optionally AI-augmented) who apply strategic rules and issue directives  
- **Worker Agents**: Specialized AI systems that execute domain-specific actions based on supervisor directives

### Multi-Engine Implementation
- **Rule Engine**: Declarative rule-based coordination emphasizing modular composition
- **CPN Engine**: Formal Colored Petri Net semantics with typed places and guarded transitions
- **SNAKES Engine**: Integration with established Petri net libraries for analysis and verification

## Empirical Validation

The `evaluation/` directory contains comprehensive benchmarks comparing TB-CSPN with LangGraph-style prompt chaining. Our validated results demonstrate significant architectural advantages:

### Key Results (Validated)
- **62.5% faster processing** than LangGraph pipelines
- **66.7% fewer LLM API calls** through architectural efficiency
- **Equal reliability** while maintaining semantic fidelity
- **166.8% higher throughput** for production deployments

### Quick Reproduction

```bash
# Install dependencies
pip install -r evaluation/requirements.txt

# Run validated comparison (no API key needed)
python evaluation/fair_comparison.py
```

**Published results based on**: `fair_comparison.py` with simulated LLM timing calibrated to realistic API latency.

### Advanced Usage

```bash
# Extended comparison with real OpenAI integration (requires API key)
python evaluation/enhanced_fair_comparison.py
```

**Note**: Real LLM integration requires OpenAI API key and incurs usage costs (~$0.50-2.00 for complete evaluation). Results may show different absolute timing due to network variability but preserve architectural efficiency ratios.

See `evaluation/README.md` for complete methodology, limitations, and extension guidelines.

## Repository Structure

```
tb-cspn-poc/
├── README.md                          # This file
├── core/                              # Core TB-CSPN framework
│   ├── token.py                       # Token data structures
│   └── agents/                        # Agent implementations
├── lo_engine/                         # Rule-based coordination engine
│   ├── token.py                       # Token definitions
│   ├── rules.py                       # Declarative rule system
│   └── lo_main.py                     # Main execution script
├── cpn_engine/                        # Colored Petri Net engine
│   ├── petriNet.py                    # CPN implementation
│   ├── place.py, transition.py        # Basic CPN components
│   └── petriNetTest.py                # CPN test example
├── snakes_engine/                     # SNAKES library integration
│   └── tb_cspn_snakes.py              # Petri net formalization
├── examples/                          # Example applications
│   └── financial_news/
│       └── consultant_1.py            # LLM-based financial consultant
├── evaluation/                        # Comparative evaluation framework
│   ├── README.md                      # Detailed methodology
│   ├── fair_comparison.py             # Validated benchmark (main results)
│   ├── enhanced_fair_comparison.py    # Real LLM integration option
│   └── results/                       # Generated benchmark results
└── docs/                              # Documentation
```

## Quick Start

### Basic Token Processing (Rule Engine)

```python
# Rule-based coordination example
from lo_engine.token import Token
from lo_engine.rules import rule

# Define a rule
@rule
def high_ai_relevance_guard(token):
    if token.topics.get("AI_stocks", 0) >= 0.8:
        return {"directive": ("AI_stocks", 1.0)}
    return None

# Process a token
token = Token(topics={"AI_stocks": 0.9}, content="AI breakthrough announced")
# Rule engine processes and fires appropriate directives
```

### CPN Engine Example

```python
# Colored Petri Net coordination
from cpn_engine.petriNet import PetriNet
from cpn_engine.place import Place
from cpn_engine.transition import Transition

# Create places and transitions
input_place = Place("input", capacity=10)
output_place = Place("output", capacity=10)
processing_transition = Transition("process", [input_place], [output_place])

# Execute Petri net workflow
net = PetriNet([input_place, output_place], [processing_transition])
# Run formal analysis and verification
```

### Financial News Processing

```python
# LLM-based financial analysis (examples/financial_news/)
from examples.financial_news.consultant_1 import generate_topic_from_csv

# Process financial news with LLM topic extraction
tokens = generate_topic_from_csv("financial_news.csv")
# Returns semantically annotated tokens with topic distributions
```

## Architecture Principles

TB-CSPN demonstrates key design principles for efficient agentic AI:

1. **Separation of Concerns**: LLMs for semantic understanding, formal methods for coordination
2. **Dedicated Multi-Agent Environment**: Purpose-built coordination substrate vs. LLM-mediated orchestration
3. **Human Strategic Control**: Humans at supervisor layer, AI at consultant/worker layers
4. **Formal Verification**: Petri net semantics enable mathematical guarantees
5. **Efficiency Through Architecture**: 2-3x performance gains through proper design

## Formal Properties and Verification

### Rule Engine (LO-Inspired)
The rule-based engine implements coordination through declarative rules inspired by Linear Objects formalism. Rules can be added independently without global rewrites, supporting modular agent architectures. This approach corresponds to the multiplicative fragment of Linear Logic and maintains equivalence with Colored Petri Nets.

### CPN Engine Guarantees
The Colored Petri Net implementation provides formal semantics with verifiable properties:

- **Typed Places**: Each place accepts tokens of specific semantic types
- **Guarded Transitions**: Threshold-based firing with relevance aggregation  
- **Structural Properties**: For layered V-model networks, guarantees include:
  - Monotonic enabling (enabled transitions stay enabled)
  - Eventual firing (all transitions fire when conditions hold)
  - Acyclic safety (each transition fires at most once per workflow)

### SNAKES Integration
The SNAKES-based implementation enables classical Petri net analysis including reachability analysis, deadlock detection, and liveness verification.

## Academic Papers

This implementation supports the following publications:

1. **"Agentic AI: Debunking the Myth and Building the Concept"** (Future Internet, 2025) - Main theoretical framework and empirical validation
2. **"TB-CSPN Framework"** ([Springer](https://link.springer.com/content/pdf/10.1007/s10791-025-09667-2.pdf)) - Formal foundations and multi-agent coordination
3. **"Centaurian Architectures"** ([Frontiers](https://www.frontiersin.org/journals/human-dynamics/articles/10.3389/fhumd.2025.1579166/full)) - Human-AI collaboration models

## Installation and Dependencies

```bash
git clone https://github.com/Aribertus/tb-cspn-poc.git
cd tb-cspn-poc

# Core dependencies
pip install snakes  # for Petri net analysis
pip install openai  # for LLM integration (examples)

# Evaluation dependencies  
pip install -r evaluation/requirements.txt
```

## Running Examples

### Test CPN Engine
```bash
python cpn_engine/petriNetTest.py
```
Expected output:
```
[TEST] Executing Petri Net Transition...
[RESULT] Final decision: topic = decision, value = buy
```

### Test Rule Engine
```bash
python lo_engine/lo_main.py
```

### Test SNAKES Formalization
```bash
python snakes_engine/tb_cspn_snakes.py
```

### Run Comparative Evaluation
```bash
cd evaluation
python fair_comparison.py
```

## Future Directions

### Enhanced Formal Analysis
- Token multisets and resource consumption semantics
- Tracing and visualization of Petri net execution
- Inhibitory arcs for complex coordination constraints

### Extended Integration
- Integration with external agents (LLMs, market simulators)
- Real-time data feeds and streaming processing
- Enterprise knowledge base integration

### Advanced Coordination
- Additive conjunction (`&`) for controlled environment forking
- Multi-scenario evaluation and divergent agent perspectives
- Hypothetical planning and simulation capabilities

## Contributing

We welcome contributions! Areas of particular interest:

- Additional benchmark frameworks (AutoGen, CrewAI, etc.)
- Domain-specific applications beyond financial news
- Enhanced visualization and debugging tools
- Performance optimizations and scalability improvements

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work, please cite our foundational paper:

```bibtex
@article{borghoff2025organizational,
  title={An organizational theory for multi-agent interactions integrating human agents, LLMs, and specialized AI},
  author={Borghoff, Uwe M. and Bottoni, Paolo and Pareschi, Remo},
  journal={Discovering Computing},
  volume={28},
  pages={138},
  year={2025},
  doi={10.1007/s10791-025-09667-2},
  url={https://doi.org/10.1007/s10791-025-09667-2}
}
```

## Contact

- Remo Pareschi: remo.pareschi@unimol.it
- Project Issues: [GitHub Issues](https://github.com/Aribertus/tb-cspn-poc/issues)

---

© 2025 — TB-CSPN Research Team
