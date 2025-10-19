# agents/worker.py
from core.topic_utils import is_enabled
from core.token import Token


def optimize_portfolio(token: Token):
    token.metadata["layer"] = "computation"
    if is_enabled(token, "AI_stocks", 0.8):
        print("[Worker] Executing portfolio reallocation.")
        token.trace("Executed by Worker")
    else:
        print("[Worker] Token not relevant enough for portfolio optimization.")