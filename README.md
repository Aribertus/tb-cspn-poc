
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
