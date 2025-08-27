# evaluation/finance_taxonomy.py
from __future__ import annotations
import re
from typing import Dict, Iterable

# Canonical finance topics you’ll see aggregated in results
CANON_TOPICS: Iterable[str] = [
    "ai_chip_demand",
    "monetary_policy",
    "inflation",
    "retail_sales",
    "consumer_sentiment",
    "opec_production_cuts",
    "global_oil_prices",
    "loan_loss_provisions",
    "bank_financial_health",
    "earnings",
    "guidance",
    "regulation",
    "data_portability_regulations",
    "big_tech_financial_impact",
    "general_market",
]

# Synonym patterns -> canonical topic
_PATTERNS = [
    (r"\b(ai|artificial intelligence).*(chip|demand)|nvidia|semiconductor", "ai_chip_demand"),
    (r"\b(fed|federal reserve|rate|monetary|policy)\b", "monetary_policy"),
    (r"\binflation|cpi|ppi\b", "inflation"),
    (r"\bretail sales?\b", "retail_sales"),
    (r"\bconsumer sentiment|confidence\b", "consumer_sentiment"),
    (r"\bopec\+?|production cuts?\b", "opec_production_cuts"),
    (r"\b(brent|wti|oil prices?)\b", "global_oil_prices"),
    (r"\bloan[-\s]?loss provisions?\b", "loan_loss_provisions"),
    (r"\b(bank|balance sheet|tier 1|delinquen(cy|cies))\b", "bank_financial_health"),
    (r"\bearnings|eps|q[1-4]\b", "earnings"),
    (r"\bguidance|outlook\b", "guidance"),
    (r"\b(regulation|regulatory|antitrust|privacy|compliance)\b", "regulation"),
    (r"\bdata portability\b", "data_portability_regulations"),
    (r"\bbig tech|faang|mega cap tech\b", "big_tech_financial_impact"),
]

_PATTERNS = [(re.compile(pat, re.I), canon) for pat, canon in _PATTERNS]

def _clip01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def normalize_topics(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Map raw topic names (from LLM or heuristics) to canonical finance topics
    and aggregate scores. Unmatched keys go to 'general_market'.
    """
    out: Dict[str, float] = {}
    for k, v in (raw or {}).items():
        key = (k or "").strip().lower()
        score = _clip01(v)

        matched = False
        for rx, canon in _PATTERNS:
            if rx.search(key):
                out[canon] = max(out.get(canon, 0.0), score)
                matched = True
                break

        if not matched:
            # heuristic fallback: short common terms
            if "inflation" in key:
                out["inflation"] = max(out.get("inflation", 0.0), score)
            elif "earnings" in key:
                out["earnings"] = max(out.get("earnings", 0.0), score)
            else:
                out["general_market"] = max(out.get("general_market", 0.0), score)

    # keep only canon topics + ensure clipping
    return {k: _clip01(out[k]) for k in out.keys() if k in CANON_TOPICS}
