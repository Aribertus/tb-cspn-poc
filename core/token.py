# core/token.py
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Token:
    payload: str
    topic_distribution: Dict[str, float]
    metadata: Dict[str, Any]

    def trace(self, step: str):
        token_id = self.metadata.get('id', 'undefined')
        layer = self.metadata.get('layer', 'undefined')
        print(f"[Trace] {step} | ID={token_id} | Layer={layer} | Topics={self.topic_distribution}")

# core/topic_utils.py
from core.token import Token


def is_enabled(token: Token, topic: str, threshold: float) -> bool:
    return token.topic_distribution.get(topic, 0.0) >= threshold