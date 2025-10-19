# lo_engine/token.py

from dataclasses import dataclass, field
import uuid

@dataclass
class Token:
    text: str
    topics: dict
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
