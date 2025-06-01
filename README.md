
# TB-CSPN PoC: Financial Scenario

This repository contains a lightweight Proof of Concept (PoC) implementation of the **TB-CSPN (Topic-Based Communication-Space Petri Net)** framework, focused on the financial portfolio optimization scenario described in our forthcoming paper.

The prototype models semantic coordination among modular agents via threshold-based token propagation across communication layers (`observation`, `surface`, `computation`).

## 🧠 Core Components

- **Consultant Agent**: Parses incoming news items and produces semantically enriched tokens with topic distributions.
- **Supervisor Agent**: Evaluates topic activation thresholds and decides whether to issue a directive.
- **Worker Agent**: Executes portfolio reallocation actions based on valid directives (currently mocked).

## 🛠️ Features

- Python-based, modular design
- Token metadata tracking (`id`, `layer`)
- Semantic threshold logic
- Transparent trace logging
- Easily extendable to plug in real LLMs or optimization logic (e.g., G-learning)

## 📦 Installation

```bash
git clone https://github.com/remo-pareschi/tb-cspn-poc.git
cd tb-cspn-poc
pip install -r requirements.txt  # if using any external libraries

---

## 🧩 Formal Petri Net Model (SNAKES-based)

In addition to the agent-based simulation (`main.py`), the repository includes a **Petri Net formalization** of the TB-CSPN logic using the [SNAKES library](https://snakes.ibisc.univ-evry.fr/).

📄 File: `tb_cspn_snakes.py`

### 🔍 What it does

- Models the core flow shown in **Figure 3** of the paper (Consultant → Supervisor → Worker).
- Defines places and transitions explicitly using Petri Net primitives.
- Demonstrates token propagation, threshold filtering, and agent-equivalent logic.
- Can be used for formal analysis, visualization, or validation.

### 🧪 Run the Petri Net Model

```bash
python tb_cspn_snakes.py
# TB-CSPN PoC: Logical Rules Prototype (LO-inspired)

This repository hosts an experimental prototype for modeling the **TB-CSPN (Topic-Based Communication-Space Petri Net)** logic layer using **Linear Logic-style rules**, inspired by the **LO (Linear Objects)** formalism. This version focuses on **rule-based semantic coordination** and **relevance-driven agent execution**, illustrating how a logical substrate could govern multi-agent decision processes.

## 🧠 Overview
- **Tokens** represent semantically tagged news events or observations, annotated with topic weights.
- **Consultant** generates tokens from inputs, simulating a topic extractor (e.g., LLM or topic model).
- **Supervisor** applies a pool of declarative rules (guards) to decide whether action should be triggered.
- **Worker** agents execute domain-specific actions (e.g., financial optimization) when directives are fired.

## ⚙️ Features
- Declarative rule registration via decorators (simulating LO-style rules).
- Parameterized guard instantiation (simulating contextual unification).
- Extendable rule base without rewriting existing logic.
- Traceable token lifecycle.
- Modular architecture for integration with LLM APIs or optimization libraries (e.g., G-learning).

## 🧪 Example Rules
```python
@rule
def high_ai_relevance_guard(token):
    if token.topics.get("AI_stocks", 0) >= 0.8:
        return {"directive": ("AI_stocks", 1.0)}
    return None
```

## 🧩 LO Equivalence and Design Considerations
This implementation corresponds to the **multiplicative fragment** of LO, and by equivalence, to **Colored Petri Nets (CPNs)** — where topics act as colors. The advantage of using LO-style rules lies in their **locality**: each rule can be added independently, avoiding global rewrites. This aligns with modular and distributed agent architectures.

## 🔄 Future Directions

### Additive Conjunction (`&`) and Forking Semantics
The prototype currently focuses on linear resource consumption (`A ⊗ B`) but omits additive branching. Future versions may include the **additive conjunction operator** `&`, allowing controlled forking of environments. This would enable:
- Multi-scenario evaluation
- Divergent agent perspectives
- Hypothetical planning and simulation

### Worker Agents as Action Interfaces
While workers currently respond to directives using mock actions, future versions can instantiate **parameterized software components**, including:
- Portfolio optimizers (e.g., G-learning agent)
- LLM-assisted financial analysts
- External APIs for execution

### Data Context and Token Pool
The rule system operates over a pool of contextually available resources. These can be fed by:
- Financial data sources (Bloomberg, Reuters)
- Sentiment analyzers
- Semantic news aggregators
- Topic volume trend monitors

## 📂 Files
- `lo_main.py`: Main execution script for rule-driven token processing.
- `lo_engine/token.py`: Token dataclass definition.
- `lo_engine/rules.py`: Declarative guard definitions.
- `README.md`: You are here.

## 📚 Related Work
- Jean-Marc Andreoli, Remo Pareschi. "Linear Objects: Logical Processes with Built-in Inheritance." *New Generation Computing*, 9(3/4):445–474, 1991. [DOI: 10.1007/BF03037173](https://doi.org/10.1007/BF03037173)
- Kurt Jensen. *Coloured Petri Nets – Basic Concepts, Analysis Methods and Practical Use, Volume 1*. Springer, 1992. [DOI: 10.1007/978-3-662-06289-0](https://doi.org/10.1007/978-3-662-06289-0)
- [SNAKES: Petri Net Library in Python](https://snakes.ibisc.univ-evry.fr)

## 📜 License
MIT License.

---
© 2025 — TB-CSPN Research Team
