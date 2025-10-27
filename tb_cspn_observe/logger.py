# tb_cspn_observe/logger.py
from __future__ import annotations
import json
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Union

Pathish = Union[str, Path]

@contextmanager
def open_jsonl(path: Pathish) -> IO[str]:
    """Append-open a JSONL file, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("a", encoding="utf-8")
    try:
        yield f
    finally:
        f.close()

def write_json(f: IO[str], obj: Any) -> None:
    """Write one compact JSON object per line, then flush."""
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()

def log_jsonl(path: Pathish, obj: Any) -> None:
    with open_jsonl(path) as f:
        write_json(f, obj)

