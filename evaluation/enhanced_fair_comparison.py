"""
Enhanced Fair Comparison using real OpenAI API (optional)
Based on your consultant_1.py approach
"""

from pathlib import Path
from tb_cspn_observe.logger import open_jsonl

# Observability setup
Path("runs").mkdir(exist_ok=True)
try:
    OBS_LOG
except NameError:
    OBS_LOG = open_jsonl("runs/obs.jsonl")
THREAD_ID = "run-001"

import openai
import time
import json
import random
import os
from typing import Dict, List, Any, Tuple
import re

# --------- Parse JSON even if wrapped in ```json fences ----------
def _json_from_maybe_markdown(text: str):
    """Remove ```...``` fences if present and parse JSON."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return json.loads(s)

# --------- Global model selector (edit via env OPENAI_MODEL if desired) ----------
LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# -------------------------------------------------------------------------------


# ---------- Helpers for robust logging & OpenAI call styles ----------

def _normalize_messages(messages) -> List[dict]:
    """Accepts list[{role,content}] or str; returns list[dict]."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, list):
        return messages
    return [{"role": "user", "content": str(messages)}]


def _extract_response_fields(response) -> Tuple[str | None, Any | None]:
    """
    Duck-typed extractor that works with:
      - openai.ChatCompletion.create (legacy) -> response.choices[0].message["content"]
      - openai.chat.completions.create (newer style) -> response.choices[0].message.content
      - anthropic.messages.create -> response.content[0].text (not used here, but supported)
      - otherwise falls back to str(response)
    """
    content = None
    tool_calls = None
    try:
        if hasattr(response, "choices"):
            ch0 = response.choices[0]
            msg = getattr(ch0, "message", None)
            if isinstance(msg, dict):
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")
            else:
                content = getattr(msg, "content", None) or getattr(ch0, "text", None)
                tool_calls = getattr(msg, "tool_calls", None)
        elif hasattr(response, "content"):  # anthropic style
            c = response.content
            if isinstance(c, list) and c:
                content = getattr(c[0], "text", None) or str(response)
        if content is None:
            content = str(response)
    except Exception:
        content = str(response)
        tool_calls = None
    return content, tool_calls


def with_llm_logging(*, node_name: str, messages, model: str = LLM_MODEL,
                     temperature: float = 0.2, call, **kwargs):
    """
    Log an LLM request/response around an arbitrary call.
    Usage:
        response = with_llm_logging(
            node_name="LLM_"+MODEL_NAME,
            messages=messages,
            model=MODEL_NAME,
            temperature=TEMP,
            call=lambda: <your llm call here>,
            max_tokens=512,  # any extra kwargs are logged in the request payload
        )
    """
    msgs = _normalize_messages(messages)
    req_payload = {"messages_rendered": msgs, "model": model, "temperature": temperature}
    for k, v in kwargs.items():
        if k not in ("system",):  # avoid logging secrets if you pass them
            req_payload[k] = v

    span_id = OBS_LOG.log(
        type="llm_request",
        thread_id=THREAD_ID,
        node=node_name,
        payload=req_payload,
    )

    response = call()

    content, tool_calls = _extract_response_fields(response)
    OBS_LOG.log(
        type="llm_response",
        thread_id=THREAD_ID,
        node=node_name,
        span_id=span_id,
        payload={"content": content, "tool_calls": tool_calls},
    )

    # (Nice-to-have) Log token usage after the response
    usage = getattr(response, "usage", None)
    if usage:
        try:
            if isinstance(usage, dict):
                usage_payload = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            else:
                usage_payload = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
        except Exception:
            usage_payload = str(usage)
        OBS_LOG.log(
            type="llm_usage",
            thread_id=THREAD_ID,
            node=node_name,
            span_id=span_id,
            payload=usage_payload,
        )

    return response


def _create_chat_completion(messages, *, model: str, temperature: float, max_tokens: int):
    """
    Works with either:
      - openai.chat.completions.create (if available)
      - openai.ChatCompletion.create (legacy)
    """
    # Newer style (requires openai.chat existing)
    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        return openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    # Legacy fallback
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------- TB-CSPN Processor (1 LLM call) ----------------

class RealLLMTBCSPNProcessor:
    """TB-CSPN with real LLM calls (matching your consultant_1.py)"""

    def __init__(self, use_real_llm: bool = False, api_key: str | None = None):
        self.use_real_llm = use_real_llm
        if use_real_llm:
            openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def extract_topics_with_real_llm(self, content: str) -> Dict[str, float]:
        """Real LLM-based topic extraction matching your approach"""
        if not self.use_real_llm:
            time.sleep(0.4)  # Simulate API latency
            return self._simulate_llm_extraction(content)

        system_prompt = (
            "You are a senior financial analyst. Identify the main topics of the article "
            "and assign each a score from 0 to 1 based on financial relevance. "
            "Respond only with valid JSON. "
            'Format: {"topic_distribution": {"topic_1": score, "topic_2": score}}'
        )

        user_prompt = (
            "Read the following news article and assess its financial impact.\n"
            "1. Identify 2-3 main topics\n"
            "2. Score each topic 0-1 for financial relevance\n"
            f"News: {content}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = with_llm_logging(
                node_name=f"LLM_{LLM_MODEL}_tb_consultant",
                messages=messages,
                model=LLM_MODEL,
                temperature=0.2,
                call=lambda: _create_chat_completion(
                    messages, model=LLM_MODEL, temperature=0.2, max_tokens=512
                ),
                max_tokens=512,
            )
            content_response = (
                response.choices[0].message["content"]
                if isinstance(response.choices[0].message, dict)
                else response.choices[0].message.content
            )
            parsed = _json_from_maybe_markdown(content_response)
            return parsed.get("topic_distribution", {})
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._simulate_llm_extraction(content)  # Fallback

    def _simulate_llm_extraction(self, content: str) -> Dict[str, float]:
        """Simulate LLM extraction with realistic variability"""
        content_lower = content.lower()
        topics: Dict[str, float] = {}
        base_variance = 0.1

        if any(w in content_lower for w in ["ai", "artificial intelligence", "machine learning"]):
            topics["AI_development"] = min(1.0, 0.80 + random.uniform(-base_variance, base_variance))
        if any(w in content_lower for w in ["fed", "federal reserve", "interest rate", "monetary"]):
            topics["monetary_policy"] = min(1.0, 0.75 + random.uniform(-base_variance, base_variance))
        if any(w in content_lower for w in ["volatile", "volatility", "uncertain", "risk"]):
            topics["market_volatility"] = min(1.0, 0.70 + random.uniform(-base_variance, base_variance))
        if any(w in content_lower for w in ["retail", "consumer", "spending", "sales"]):
            topics["consumer_sector"] = min(1.0, 0.65 + random.uniform(-base_variance, base_variance))
        if any(w in content_lower for w in ["tech", "technology", "innovation", "digital"]):
            topics["tech_sector"] = min(1.0, 0.60 + random.uniform(-base_variance, base_variance))

        if not topics:
            topics["general_market"] = 0.50 + random.uniform(-0.1, 0.1)

        return topics

    def apply_tbcspn_rules(self, topics: Dict[str, float]) -> Dict[str, Any]:
        """Apply TB-CSPN deterministic rules"""
        if not topics:
            return {
                "directive": "No actionable topics identified",
                "action": "NO_ACTION",
                "confidence": 0.0,
                "rule_fired": "default_rule",
            }

        # Rule 1: High AI relevance
        ai_score = topics.get("AI_development", 0)
        if ai_score >= 0.8:
            return {
                "directive": "Monitor AI sector for strategic repositioning",
                "action": "MONITOR_AI_SECTOR",
                "confidence": ai_score,
                "rule_fired": "high_ai_relevance_rule",
            }

        # Rule 2: Market volatility + monetary policy
        volatility = topics.get("market_volatility", 0)
        monetary = topics.get("monetary_policy", 0)
        if volatility >= 0.7 and monetary >= 0.6:
            confidence = (volatility + monetary) / 2
            return {
                "directive": "Implement defensive positioning",
                "action": "DEFENSIVE_POSITIONING",
                "confidence": confidence,
                "rule_fired": "volatility_monetary_rule",
            }

        # Rule 3: Any high-relevance topic
        max_topic, max_score = max(topics.items(), key=lambda x: x[1])
        if max_score >= 0.75:
            return {
                "directive": f"Analyze {max_topic} developments for investment impact",
                "action": "SECTOR_ANALYSIS",
                "confidence": max_score,
                "rule_fired": "high_relevance_rule",
            }

        # Rule 4: Standard monitoring
        return {
            "directive": "Continue standard monitoring",
            "action": "STANDARD_MONITORING",
            "confidence": max_score,
            "rule_fired": "standard_monitoring_rule",
        }

    def process_news_item(self, news_content: str) -> Dict[str, Any]:
        """TB-CSPN pipeline: LLM extraction + deterministic rules"""
        start_time = time.time()
        try:
            topics = self.extract_topics_with_real_llm(news_content)  # 1 call
            decision = self.apply_tbcspn_rules(topics)
            return {
                "input_text": (news_content[:100] + "...") if len(news_content) > 100 else news_content,
                "topics_extracted": topics,
                "directive": decision["directive"],
                "action_taken": decision["action"],
                "rule_fired": decision["rule_fired"],
                "confidence": decision["confidence"],
                "processing_time": time.time() - start_time,
                "success": True,
                "llm_calls": 1,
                "architecture": "TB-CSPN",
            }
        except Exception as e:
            return {
                "input_text": (news_content[:100] + "...") if len(news_content) > 100 else news_content,
                "topics_extracted": {},
                "directive": None,
                "action_taken": None,
                "rule_fired": None,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "success": False,
                "error_message": str(e),
                "llm_calls": 1,
                "architecture": "TB-CSPN",
            }


# ---------------- LangGraph-ish Processor (3 LLM calls) ----------------

class RealLLMLangGraphProcessor:
    """LangGraph with multiple real LLM calls"""

    def __init__(self, use_real_llm: bool = False, api_key: str | None = None):
        self.use_real_llm = use_real_llm
        if use_real_llm:
            openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
    def consultant_llm_call(self, content: str) -> Dict[str, float]:
        """Consultant node: LLM topic extraction"""
        if not self.use_real_llm:
            time.sleep(0.35)
            return self._simulate_consultant_response(content)

        messages = [
            {
                "role": "user",
                "content": (
                    "Extract 2-3 descriptive financial topics from this news with relevance scores 0-1. "
                    "Respond ONLY as JSON (no prose, no code fences), mapping topic names to scores. "
                    'Example: {"AI_chip_demand": 0.92, "monetary_policy": 0.80}. '
                    f"News: {content}"
                ),
            }
        ]

        try:
            response = with_llm_logging(
                node_name=f"LLM_{LLM_MODEL}_consultant",
                messages=messages,
                model=LLM_MODEL,
                temperature=0.3,
                call=lambda: _create_chat_completion(
                    messages, model=LLM_MODEL, temperature=0.3, max_tokens=256
                ),
                max_tokens=256,
            )
            msg = response.choices[0].message
            text = msg["content"] if isinstance(msg, dict) else msg.content
            return _json_from_maybe_markdown(text)
        except Exception:
            return self._simulate_consultant_response(content)


    def supervisor_llm_call(self, topics: Dict[str, float], content: str) -> str:
        """Supervisor node: LLM directive generation"""
        if not self.use_real_llm:
            time.sleep(0.30)
            return self._simulate_supervisor_response(topics)

        topics_str = ", ".join([f"{k}: {v:.2f}" for k, v in topics.items()])
        messages = [{
            "role": "user",
            "content": (
                f"Based on these financial topics {topics_str}, "
                "generate a brief investment directive. Keep it under 20 words."
            ),
        }]

        try:
            response = with_llm_logging(
                node_name=f"LLM_{LLM_MODEL}_supervisor",
                messages=messages,
                model=LLM_MODEL,
                temperature=0.2,
                call=lambda: _create_chat_completion(
                    messages, model=LLM_MODEL, temperature=0.2, max_tokens=128
                ),
                max_tokens=128,
            )
            msg = response.choices[0].message
            return (msg["content"] if isinstance(msg, dict) else msg.content).strip()
        except Exception:
            return self._simulate_supervisor_response(topics)

    def worker_llm_call(self, directive: str) -> str:
        """Worker node: LLM action determination"""
        if not self.use_real_llm:
            time.sleep(0.25)
            return self._simulate_worker_response(directive)

        messages = [{
            "role": "user",
            "content": (
                f"Based on this directive: '{directive}', choose ONE action: "
                "MONITOR_AI_SECTOR, DEFENSIVE_POSITIONING, SECTOR_ANALYSIS, "
                "STANDARD_MONITORING, or NO_ACTION"
            ),
        }]

        try:
            response = with_llm_logging(
                node_name=f"LLM_{LLM_MODEL}_worker",
                messages=messages,
                model=LLM_MODEL,
                temperature=0.1,
                call=lambda: _create_chat_completion(
                    messages, model=LLM_MODEL, temperature=0.1, max_tokens=64
                ),
                max_tokens=64,
            )
            msg = response.choices[0].message
            action_text = (msg["content"] if isinstance(msg, dict) else msg.content).strip()

            actions = [
                "MONITOR_AI_SECTOR", "DEFENSIVE_POSITIONING", "SECTOR_ANALYSIS",
                "STANDARD_MONITORING", "NO_ACTION",
            ]
            for action in actions:
                if action in action_text:
                    return action
            return "STANDARD_MONITORING"
        except Exception:
            return self._simulate_worker_response(directive)

    # ---- Simulators ----

    def _simulate_consultant_response(self, content: str) -> Dict[str, float]:
        """Simulate consultant LLM with more variability than TB-CSPN"""
        content_lower = content.lower()
        topics: Dict[str, float] = {}
        variance = 0.15

        if any(w in content_lower for w in ["ai", "artificial intelligence"]):
            topics["AI_development"] = max(0.1, min(1.0, 0.75 + random.uniform(-variance, variance)))
        if any(w in content_lower for w in ["fed", "federal reserve", "rate"]):
            topics["monetary_policy"] = max(0.1, min(1.0, 0.70 + random.uniform(-variance, variance)))
        if any(w in content_lower for w in ["volatile", "volatility"]):
            topics["market_volatility"] = max(0.1, min(1.0, 0.65 + random.uniform(-variance, variance)))
        if not topics:
            topics["general_market"] = 0.50 + random.uniform(-0.2, 0.2)
        return topics

    def _simulate_supervisor_response(self, topics: Dict[str, float]) -> str:
        """Simulate supervisor LLM response"""
        if not topics:
            return "Continue standard market monitoring"
        max_topic, max_score = max(topics.items(), key=lambda x: x[1])
        phrases = [
            f"Monitor {max_topic} developments closely",
            f"Focus attention on {max_topic} trends",
            f"Track {max_topic} for investment opportunities",
            f"Assess {max_topic} impact on portfolio",
        ]
        if max_score >= 0.7:
            return random.choice(phrases)
        else:
            return "Maintain current monitoring protocols"

    def _simulate_worker_response(self, directive: str) -> str:
        """Simulate worker LLM response"""
        dl = directive.lower()
        if any(w in dl for w in ["ai", "artificial"]):
            return "MONITOR_AI_SECTOR"
        elif any(w in dl for w in ["defensive", "risk"]):
            return "DEFENSIVE_POSITIONING"
        elif any(w in dl for w in ["monitor", "track", "focus"]):
            return "SECTOR_ANALYSIS"
        else:
            return "STANDARD_MONITORING"

    # ---- Orchestrator ----

    def process_news_item(self, news_content: str) -> Dict[str, Any]:
        """LangGraph pipeline: Multiple LLM calls"""
        start_time = time.time()
        llm_calls = 0
        try:
            topics = self.consultant_llm_call(news_content)
            llm_calls += 1

            directive = self.supervisor_llm_call(topics, news_content)
            llm_calls += 1

            action = self.worker_llm_call(directive)
            llm_calls += 1

            processing_time = time.time() - start_time
            confidence = max(topics.values()) if topics else 0.5
            return {
                "input_text": (news_content[:100] + "...") if len(news_content) > 100 else news_content,
                "topics_extracted": topics,
                "directive": directive,
                "action_taken": action,
                "confidence": confidence,
                "processing_time": processing_time,
                "success": True,
                "llm_calls": llm_calls,
                "architecture": "LangGraph",
            }
        except Exception as e:
            return {
                "input_text": (news_content[:100] + "...") if len(news_content) > 100 else news_content,
                "topics_extracted": {},
                "directive": None,
                "action_taken": None,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "success": False,
                "error_message": str(e),
                "llm_calls": llm_calls,
                "architecture": "LangGraph",
            }


# ----------------- quick driver -----------------

if __name__ == "__main__":
    import json

    # A short sample article
    sample_news = (
        "NVIDIA beats earnings expectations as demand for AI chips surges. "
        "Meanwhile, the Fed signals possible rate cuts amid cooling inflation."
    )

    # Toggle real API usage (costs tokens if True)
    USE_REAL_LLM = True

    # TB-CSPN (1 LLM call)
    tb = RealLLMTBCSPNProcessor(use_real_llm=USE_REAL_LLM)
    tb_result = tb.process_news_item(sample_news)

    # LangGraph-ish (3 LLM calls)
    lg = RealLLMLangGraphProcessor(use_real_llm=USE_REAL_LLM)
    lg_result = lg.process_news_item(sample_news)

    print("\n=== TB-CSPN result ===")
    print(json.dumps(tb_result, indent=2))
    print("\n=== LangGraph result ===")
    print(json.dumps(lg_result, indent=2))

    # (Optional) write results to runs/eval_results.jsonl
    with open("runs/eval_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"tb_cspn": tb_result}) + "\n")
        f.write(json.dumps({"langgraph": lg_result}) + "\n")

    print("\nSaved results to runs/eval_results.jsonl")
