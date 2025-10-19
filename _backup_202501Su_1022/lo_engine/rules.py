# lo_engine/rules.py

from typing import Callable, List

RULES: List[Callable] = []

def rule(func: Callable) -> Callable:
    """Decorator to register a rule (guard)."""
    RULES.append(func)
    return func

@rule
def high_ai_relevance_guard(token) -> dict | None:
    if token.topics.get("AI_stocks", 0) >= 0.8:
        return {"directive": ("AI_stocks", 1.0)}
    return None

@rule
def market_volatility_guard(token) -> dict | None:
    if token.topics.get("market_volatility", 0) > 0.85:
        return {"directive": ("hedge_position", 0.7)}
    return None

def apply_rules(token) -> dict | None:
    for r in RULES:
        result = r(token)
        if result:
            return result
    return None
