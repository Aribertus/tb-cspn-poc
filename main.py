# tb_cspn_poc/main.py
from agents import consultant, supervisor, worker
from core.token import Token


def run_pipeline():
    inputs = [
        "Tech sector sees surge amid AI breakthroughs",
        "Uncertainty rises after unexpected Fed decision",
        "Retail stocks underperform despite holiday sales"
    ]

    for news in inputs:
        print(f"\n[Input] Processing: {news}")

        # Consultant analyzes news
        analyzed_token = consultant.analyze_text(news)
        analyzed_token.trace("Received by Consultant")

        # Supervisor issues directive
        directive_token = supervisor.issue_directive(analyzed_token)
        if directive_token:
            directive_token.trace("Issued by Supervisor")
            worker.optimize_portfolio(directive_token)
        else:
            print("[Supervisor] No directive issued.")
            print("[Computation] No task sent to Worker agent.")


if __name__ == "__main__":
    run_pipeline()

