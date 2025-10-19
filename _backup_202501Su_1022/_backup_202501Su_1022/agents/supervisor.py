# agents/supervisor.py
from core.token import Token


def issue_directive(token: Token) -> Token | None:
    print("[Supervisor] Evaluating token...")
    token.metadata["layer"] = "surface"
    token.trace("Supervisor Evaluation")

    if token.topic_distribution.get("AI_stocks", 0.0) > 0.8:
        print("[Supervisor] Directive triggered.")
        metadata = token.metadata.copy()
        metadata["layer"] = "surface"
        return Token(
            payload="Increase exposure to AI sector",
            topic_distribution={"AI_stocks": 1.0},
            metadata=metadata
        )
    print("[Supervisor] No action taken.")
    return None