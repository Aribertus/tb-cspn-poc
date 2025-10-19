from dataclasses import dataclass, field
import uuid

@dataclass
class Token:
    text: str
    topics: dict
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)

RULES = []

def rule(func):
    RULES.append(func)
    return func

@rule
def high_ai_relevance_guard(token):
    if token.topics.get("AI_stocks", 0) >= 0.8:
        return {"directive": ("AI_stocks", 1.0)}
    return None

def apply_rules(token):
    for r in RULES:
        result = r(token)
        if result:
            return result
    return None

def consultant_process(text):
    if "AI" in text:
        topics = {"AI_stocks": 0.9, "market_volatility": 0.5}
    elif "Fed" in text:
        topics = {"market_volatility": 0.9}
    else:
        topics = {"retail": 0.8}
    return Token(text=text, topics=topics)

def supervisor(token):
    directive = apply_rules(token)
    if directive:
        print(f"[Supervisor] Directive issued: {directive}")
        return directive
    else:
        print("[Supervisor] No directive issued.")
        return None

def execute_directive(directive):
    topic, value = directive["directive"]
    print(f"[Worker] Executing action on topic '{topic}' with weight {value}")

def main():
    news_inputs = [
        "Tech sector sees surge amid AI breakthroughs",
        "Uncertainty rises after unexpected Fed decision",
        "Retail stocks underperform despite holiday sales"
    ]

    for news in news_inputs:
        print(f"\n[Input] Processing: {news}")
        
        token = consultant_process(news)
        print(f"[Consultant] Token created | ID={token.id} | Topics={token.topics}")

        directive = supervisor(token)
        
        if directive:
            execute_directive(directive)
        else:
            print("[Computation] No task sent to Worker agent.")

if __name__ == "__main__":
    main()

