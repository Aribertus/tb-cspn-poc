# topic_utils.py
from core.token import Token

def is_enabled(token: Token, topic: str, threshold: float) -> bool:
    return token.topic_distribution.get(topic, 0.0) >= threshold



