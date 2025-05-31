
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
