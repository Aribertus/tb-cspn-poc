# agents/consultant.py
from core.token import Token
import uuid


def analyze_text(text: str) -> Token:
    if "AI" in text or "Tech" in text:
        topics = {"AI_stocks": 0.9, "market_volatility": 0.6}
    elif "Fed" in text or "uncertainty" in text:
        topics = {"AI_stocks": 0.3, "market_volatility": 0.9}
    elif "Retail" in text or "holiday" in text:
        topics = {"retail": 0.8, "AI_stocks": 0.2}
    else:
        topics = {"misc": 0.4}

    token = Token(
        payload=text,
        topic_distribution=topics,
        metadata={
            "id": str(uuid.uuid4()),
            "layer": "observation"
        }
    )
    print("[Consultant] Token created.")
    return token
