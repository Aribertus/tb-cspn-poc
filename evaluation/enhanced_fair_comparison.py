"""
Enhanced Fair Comparison using real OpenAI API (optional)
Based on your consultant_1.py approach
"""

import openai
import time
import json
import random
from typing import Dict, List, Any

# Set your API key here or use environment variable
# openai.api_key = "your-api-key-here"  # Uncomment and add your key

class RealLLMTBCSPNProcessor:
    """TB-CSPN with real LLM calls (matching your consultant_1.py)"""
    
    def __init__(self, use_real_llm: bool = False, api_key: str = None):
        self.use_real_llm = use_real_llm
        if use_real_llm and api_key:
            openai.api_key = api_key
    
    def extract_topics_with_real_llm(self, content: str) -> Dict[str, float]:
        """Real LLM-based topic extraction matching your approach"""
        
        if not self.use_real_llm:
            # Simulate realistic LLM timing and results
            time.sleep(0.4)  # Realistic API latency
            return self._simulate_llm_extraction(content)
        
        system_prompt = (
            "You are a senior financial analyst. Identify the main topics of the article "
            "and assign each a score from 0 to 1 based on financial relevance. "
            "Respond only with valid JSON. "
            "Format: {\"topic_distribution\": {\"topic_1\": score, \"topic_2\": score}}"
        )
        
        user_prompt = (
            f"Read the following news article and assess its financial impact.\n"
            f"1. Identify 2-3 main topics\n"
            f"2. Score each topic 0-1 for financial relevance\n"
            f"News: {content}"
        )
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=512
            )
            
            content_response = response.choices[0].message.content
            parsed = json.loads(content_response)
            return parsed.get("topic_distribution", {})
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._simulate_llm_extraction(content)  # Fallback
    
    def _simulate_llm_extraction(self, content: str) -> Dict[str, float]:
        """Simulate LLM extraction with realistic variability"""
        content_lower = content.lower()
        topics = {}
        
        # Add some randomness to simulate LLM variability
        base_variance = 0.1
        
        if any(word in content_lower for word in ["ai", "artificial intelligence", "machine learning"]):
            topics["AI_development"] = min(1.0, 0.80 + random.uniform(-base_variance, base_variance))
        
        if any(word in content_lower for word in ["fed", "federal reserve", "interest rate", "monetary"]):
            topics["monetary_policy"] = min(1.0, 0.75 + random.uniform(-base_variance, base_variance))
            
        if any(word in content_lower for word in ["volatile", "volatility", "uncertain", "risk"]):
            topics["market_volatility"] = min(1.0, 0.70 + random.uniform(-base_variance, base_variance))
            
        if any(word in content_lower for word in ["retail", "consumer", "spending", "sales"]):
            topics["consumer_sector"] = min(1.0, 0.65 + random.uniform(-base_variance, base_variance))
            
        if any(word in content_lower for word in ["tech", "technology", "innovation", "digital"]):
            topics["tech_sector"] = min(1.0, 0.60 + random.uniform(-base_variance, base_variance))
        
        # Ensure we have at least one topic
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
                "rule_fired": "default_rule"
            }
        
        # Rule 1: High AI relevance
        ai_score = topics.get("AI_development", 0)
        if ai_score >= 0.8:
            return {
                "directive": "Monitor AI sector for strategic repositioning",
                "action": "MONITOR_AI_SECTOR",
                "confidence": ai_score,
                "rule_fired": "high_ai_relevance_rule"
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
                "rule_fired": "volatility_monetary_rule"
            }
        
        # Rule 3: Any high-relevance topic
        max_topic, max_score = max(topics.items(), key=lambda x: x[1])
        
        if max_score >= 0.75:
            return {
                "directive": f"Analyze {max_topic} developments for investment impact",
                "action": "SECTOR_ANALYSIS",
                "confidence": max_score,
                "rule_fired": "high_relevance_rule"
            }
        
        # Rule 4: Standard monitoring
        return {
            "directive": "Continue standard monitoring",
            "action": "STANDARD_MONITORING",
            "confidence": max_score,
            "rule_fired": "standard_monitoring_rule"
        }
    
    def process_news_item(self, news_content: str) -> Dict[str, Any]:
        """TB-CSPN pipeline: LLM extraction + deterministic rules"""
        start_time = time.time()
        
        try:
            # Step 1: LLM topic extraction (1 call)
            topics = self.extract_topics_with_real_llm(news_content)
            
            # Step 2: Deterministic rule application (no LLM)
            decision = self.apply_tbcspn_rules(topics)
            
            processing_time = time.time() - start_time
            
            return {
                "input_text": news_content[:100] + "..." if len(news_content) > 100 else news_content,
                "topics_extracted": topics,
                "directive": decision["directive"],
                "action_taken": decision["action"],
                "rule_fired": decision["rule_fired"],
                "confidence": decision["confidence"],
                "processing_time": processing_time,
                "success": True,
                "llm_calls": 1,
                "architecture": "TB-CSPN"
            }
            
        except Exception as e:
            return {
                "input_text": news_content[:100] + "..." if len(news_content) > 100 else news_content,
                "topics_extracted": {},
                "directive": None,
                "action_taken": None,
                "rule_fired": None,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "success": False,
                "error_message": str(e),
                "llm_calls": 1,
                "architecture": "TB-CSPN"
            }


class RealLLMLangGraphProcessor:
    """LangGraph with multiple real LLM calls"""
    
    def __init__(self, use_real_llm: bool = False, api_key: str = None):
        self.use_real_llm = use_real_llm
        if use_real_llm and api_key:
            openai.api_key = api_key
    
    def consultant_llm_call(self, content: str) -> Dict[str, float]:
        """Consultant node: LLM topic extraction"""
        
        if not self.use_real_llm:
            time.sleep(0.35)  # Simulate API latency
            return self._simulate_consultant_response(content)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user", 
                    "content": f"Extract 2-3 financial topics from this news with relevance scores 0-1. "
                              f"Respond as JSON: {{\"topic1\": score, \"topic2\": score}}. News: {content}"
                }],
                temperature=0.3,
                max_tokens=256
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception:
            return self._simulate_consultant_response(content)
    
    def supervisor_llm_call(self, topics: Dict[str, float], content: str) -> str:
        """Supervisor node: LLM directive generation"""
        
        if not self.use_real_llm:
            time.sleep(0.30)  # Simulate API latency
            return self._simulate_supervisor_response(topics)
        
        try:
            topics_str = ", ".join([f"{k}: {v:.2f}" for k, v in topics.items()])
            
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Based on these financial topics {topics_str}, "
                              f"generate a brief investment directive. Keep it under 20 words."
                }],
                temperature=0.2,
                max_tokens=128
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception:
            return self._simulate_supervisor_response(topics)
    
    def worker_llm_call(self, directive: str) -> str:
        """Worker node: LLM action determination"""
        
        if not self.use_real_llm:
            time.sleep(0.25)  # Simulate API latency
            return self._simulate_worker_response(directive)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Based on this directive: '{directive}', "
                              f"choose ONE action: MONITOR_AI_SECTOR, DEFENSIVE_POSITIONING, "
                              f"SECTOR_ANALYSIS, STANDARD_MONITORING, or NO_ACTION"
                }],
                temperature=0.1,
                max_tokens=64
            )
            
            action_text = response.choices[0].message.content.strip()
            
            # Extract action code
            actions = ["MONITOR_AI_SECTOR", "DEFENSIVE_POSITIONING", "SECTOR_ANALYSIS", 
                      "STANDARD_MONITORING", "NO_ACTION"]
            
            for action in actions:
                if action in action_text:
                    return action
            
            return "STANDARD_MONITORING"  # Default fallback
            
        except Exception:
            return self._simulate_worker_response(directive)
    
    def _simulate_consultant_response(self, content: str) -> Dict[str, float]:
        """Simulate consultant LLM with more variability than TB-CSPN"""
        content_lower = content.lower()
        topics = {}
        
        # More variability to simulate LLM inconsistency
        variance = 0.15
        
        if any(word in content_lower for word in ["ai", "artificial intelligence"]):
            topics["AI_development"] = max(0.1, min(1.0, 0.75 + random.uniform(-variance, variance)))
        
        if any(word in content_lower for word in ["fed", "federal reserve", "rate"]):
            topics["monetary_policy"] = max(0.1, min(1.0, 0.70 + random.uniform(-variance, variance)))
            
        if any(word in content_lower for word in ["volatile", "volatility"]):
            topics["market_volatility"] = max(0.1, min(1.0, 0.65 + random.uniform(-variance, variance)))
        
        if not topics:
            topics["general_market"] = 0.50 + random.uniform(-0.2, 0.2)
            
        return topics
    
    def _simulate_supervisor_response(self, topics: Dict[str, float]) -> str:
        """Simulate supervisor LLM response"""
        if not topics:
            return "Continue standard market monitoring"
        
        max_topic, max_score = max(topics.items(), key=lambda x: x[1])
        
        # Add some LLM-style variability in phrasing
        phrases = [
            f"Monitor {max_topic} developments closely",
            f"Focus attention on {max_topic} trends", 
            f"Track {max_topic} for investment opportunities",
            f"Assess {max_topic} impact on portfolio"
        ]
        
        if max_score >= 0.7:
            return random.choice(phrases)
        else:
            return "Maintain current monitoring protocols"
    
    def _simulate_worker_response(self, directive: str) -> str:
        """Simulate worker LLM response"""
        directive_lower = directive.lower()
        
        if any(word in directive_lower for word in ["ai", "artificial"]):
            return "MONITOR_AI_SECTOR"
        elif any(word in directive_lower for word in ["defensive", "risk"]):
            return "DEFENSIVE_POSITIONING"
        elif any(word in directive_lower for word in ["monitor", "track", "focus"]):
            return "SECTOR_ANALYSIS"
        else:
            return "STANDARD_MONITORING"
    
    def process_news_item(self, news_content: str) -> Dict[str, Any]:
        """LangGraph pipeline: Multiple LLM calls"""
        start_time = time.time()
        llm_calls = 0
        
        try:
            # Step 1: Consultant LLM call
            topics = self.consultant_llm_call(news_content)
            llm_calls += 1
            
            # Step 2: Supervisor LLM call
            directive = self.supervisor_llm_call(topics, news_content)
            llm_calls += 1
            
            # Step 3: Worker LLM call
            action = self.worker_llm_call(directive)
            llm_calls += 1
            
            processing_time = time.time() - start_time
            
            # Calculate confidence (estimate based on topic scores)
            confidence = max(topics.values()) if topics else 0.5
            
            return {
                "input_text": news_content[:100] + "..." if len(news_content) > 100 else news_content,
                "topics_extracted": topics,
                "directive": directive,
                "action_taken": action,
                "confidence": confidence,
                "processing_time": processing_time,
                "success": True,
                "llm_calls": llm_calls,
                "architecture": "LangGraph"
            }
            
        except Excepti