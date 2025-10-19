# lo_main.py — Main execution for LO-inspired TB-CSPN logic layer

from lo_engine.token import Token
from lo_engine.rules import apply_rules

def consultant_process(text: str) -> Token:
    if "AI" in text:
        topics = {"AI_stocks": 0.9, "market_volatility": 0.5}
    elif "Fed" in text:
        topics = {"market_volatility": 0.9}
    else:
        topics = {"retail": 0.8}
    token = Token(text=text, topics=topics)
    print(f"[Consultant] Token created | ID={token.id} | Topics={token.topics}")
    return token

def supervisor(token: Token):
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
        directive = supervisor(token)
        if directive:
            execute_directive(directive)
        else:
            print("[Computation] No task sent to Worker agent.")

if __name__ == "__main__":
    main()

