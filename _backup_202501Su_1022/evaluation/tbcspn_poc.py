"""
TB-CSPN POC Implementation for Financial News Processing
Enhanced version of your existing implementation for comparative evaluation
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class Token:
    """Enhanced Token class for TB-CSPN"""
    id: str
    layer: str
    topics: Dict[str, float]
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def exceeds_topic_threshold(self, topic: str, threshold: float) -> bool:
        return self.topics.get(topic, 0.0) >= threshold
    
    def get_dominant_topics(self, min_score: float = 0.5) -> List[tuple]:
        return sorted([(k, v) for k, v in self.topics.items() if v >= min_score], 
                     key=lambda x: x[1], reverse=True)


@dataclass
class Directive:
    """Directive issued by Supervisor agents"""
    id: str
    action: str
    confidence: float
    source_token_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class ProcessingResult:
    """Result structure matching LangGraph for comparison"""
    input_text: str
    topics_extracted: Dict[str, float]
    directive: Optional[str]
    action_taken: Optional[str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    # TB-CSPN specific fields
    token_id: Optional[str] = None
    rule_trace: List[str] = None


class TopicExtractor:
    """Topic extraction interface - can be LLM-based or rule-based"""
    
    def __init__(self, method: str = "keyword", llm_client=None):
        self.method = method
        self.llm_client = llm_client
    
    def extract_topics(self, content: str) -> Dict[str, float]:
        if self.method == "llm" and self.llm_client:
            return self._llm_extraction(content)
        else:
            return self._keyword_extraction(content)
    
    def _keyword_extraction(self, content: str) -> Dict[str, float]:
        """Enhanced keyword-based topic extraction"""
        keywords = {
            "market_volatility": {
                "patterns": [r"volatilit", r"uncertain", r"swing", r"fluctuat", r"unstable"],
                "weights": [1.0, 0.8, 0.7, 0.9, 0.8]
            },
            "fed_policy": {
                "patterns": [r"federal reserve", r"\bfed\b", r"interest rate", r"monetary policy", r"powell"],
                "weights": [1.0, 0.9, 0.95, 1.0, 0.8]
            },
            "AI_sector": {
                "patterns": [r"\bAI\b", r"artificial intelligence", r"machine learning", r"neural", r"algorithm"],
                "weights": [1.0, 1.0, 0.9, 0.8, 0.7]
            },
            "tech_momentum": {
                "patterns": [r"technolog", r"innovation", r"digital", r"software", r"chip"],
                "weights": [0.9, 0.8, 0.7, 0.8, 0.9]
            },
            "retail_sector": {
                "patterns": [r"retail", r"consumer goods", r"shopping", r"stores", r"e-commerce"],
                "weights": [1.0, 0.9, 0.8, 0.8, 0.9]
            },
            "consumer_spending": {
                "patterns": [r"spending", r"consumption", r"purchases", r"demand", r"sales"],
                "weights": [1.0, 0.9, 0.8, 0.8, 0.9]
            }
        }
        
        content_lower = content.lower()
        topics = {}
        
        for topic, config in keywords.items():
            total_score = 0
            max_weight = 0
            
            for pattern, weight in zip(config["patterns"], config["weights"]):
                matches = len(re.findall(pattern, content_lower))
                if matches > 0:
                    total_score += min(matches * weight * 0.3, weight)
                    max_weight = max(max_weight, weight)
            
            if total_score > 0:
                # Normalize and cap the score
                normalized_score = min(total_score, 1.0)
                topics[topic] = round(normalized_score, 3)
        
        return topics
    
    def _llm_extraction(self, content: str) -> Dict[str, float]:
        """LLM-based topic extraction (when available)"""
        # This would integrate with your LLM client
        # For now, fallback to keyword extraction
        return self._keyword_extraction(content)


class RuleEngine:
    """Enhanced rule engine with better tracing"""
    
    def __init__(self):
        self.rules = []
        self.execution_trace = []
    
    def add_rule(self, rule_func: Callable):
        """Add a rule function to the engine"""
        self.rules.append(rule_func)
    
    def process_token(self, token: Token) -> List[Dict]:
        """Process token through all rules"""
        directives = []
        rule_trace = []
        
        for rule in self.rules:
            try:
                result = rule(token)
                rule_name = rule.__name__
                
                if result:
                    directives.append(result)
                    rule_trace.append(f"FIRED: {rule_name}")
                    self.execution_trace.append({
                        'token_id': token.id,
                        'rule': rule_name,
                        'result': result,
                        'timestamp': datetime.now()
                    })
                else:
                    rule_trace.append(f"SKIPPED: {rule_name}")
                    
            except Exception as e:
                rule_trace.append(f"ERROR: {rule.__name__} - {str(e)}")
        
        return directives, rule_trace


# Rule definitions
def high_ai_relevance_rule(token: Token) -> Optional[Dict]:
    """Fire when AI-related topics exceed high relevance threshold"""
    ai_score = max(
        token.topics.get("AI_sector", 0),
        token.topics.get("tech_momentum", 0) * 0.8  # Tech momentum contributes but weighted
    )
    
    if ai_score >= 0.8:
        return {
            "directive": "Monitor AI sector for strategic repositioning",
            "confidence": ai_score,
            "action_code": "MONITOR_AI_SECTOR"
        }
    return None


def market_volatility_rule(token: Token) -> Optional[Dict]:
    """Respond to high market volatility combined with policy signals"""
    volatility = token.topics.get("market_volatility", 0)
    fed_policy = token.topics.get("fed_policy", 0)
    
    if volatility >= 0.8 and fed_policy >= 0.6:
        confidence = (volatility + fed_policy) / 2
        return {
            "directive": "Implement defensive positioning",
            "confidence": confidence,
            "action_code": "DEFENSIVE_POSITIONING"
        }
    return None


def high_relevance_general_rule(token: Token) -> Optional[Dict]:
    """General rule for any topic exceeding high threshold"""
    if not token.topics:
        return None
        
    max_topic, max_score = max(token.topics.items(), key=lambda x: x[1])
    
    if max_score >= 0.85:
        return {
            "directive": f"Analyze {max_topic} developments for investment impact",
            "confidence": max_score,
            "action_code": "SECTOR_ANALYSIS"
        }
    return None


def default_monitoring_rule(token: Token) -> Optional[Dict]:
    """Default action when no other rules fire"""
    if token.topics:  # Only if we have any topics
        return {
            "directive": "Continue standard monitoring",
            "confidence": 0.5,
            "action_code": "STANDARD_MONITORING"
        }
    return None


class ConsultantAgent:
    """Enhanced Consultant Agent"""
    
    def __init__(self, topic_extractor: TopicExtractor):
        self.topic_extractor = topic_extractor
        self.processing_log = []
    
    def process_input(self, content: str) -> Token:
        """Transform input into semantically annotated token"""
        start_time = time.time()
        
        topics = self.topic_extractor.extract_topics(content)
        
        token = Token(
            id=str(uuid.uuid4()),
            layer="observation",
            topics=topics,
            content=content,
            timestamp=datetime.now(),
            metadata={
                "agent": "consultant",
                "extraction_method": self.topic_extractor.method,
                "processing_time": time.time() - start_time
            }
        )
        
        self.processing_log.append({
            'token_id': token.id,
            'input_length': len(content),
            'topics_extracted': len(topics),
            'processing_time': token.metadata["processing_time"]
        })
        
        return token


class SupervisorAgent:
    """Enhanced Supervisor Agent"""
    
    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine
        self.decision_log = []
    
    def evaluate_token(self, token: Token) -> tuple[List[Directive], List[str]]:
        """Evaluate token and generate directives with trace"""
        start_time = time.time()
        
        rule_results, rule_trace = self.rule_engine.process_token(token)
        
        directives = []
        for result in rule_results:
            directive = Directive(
                id=str(uuid.uuid4()),
                action=result.get("action_code", "NO_ACTION"),
                confidence=result.get("confidence", 0.0),
                source_token_id=token.id,
                timestamp=datetime.now(),
                metadata={
                    "directive_text": result.get("directive", ""),
                    "processing_time": time.time() - start_time
                }
            )
            directives.append(directive)
        
        self.decision_log.append({
            'token_id': token.id,
            'directives_generated': len(directives),
            'rule_trace': rule_trace,
            'timestamp': datetime.now()
        })
        
        return directives, rule_trace


class WorkerAgent:
    """Enhanced Worker Agent"""
    
    def __init__(self):
        self.execution_log = []
    
    def execute_directive(self, directive: Directive) -> Dict[str, Any]:
        """Execute directive and return result"""
        start_time = time.time()
        
        # Simulate action execution
        action_map = {
            "MONITOR_AI_SECTOR": "Initiated AI sector monitoring dashboard",
            "DEFENSIVE_POSITIONING": "Activated defensive portfolio allocation",
            "SECTOR_ANALYSIS": "Launched detailed sector analysis report",
            "STANDARD_MONITORING": "Continuing regular market surveillance",
            "NO_ACTION": "No action required"
        }
        
        result = {
            "action_code": directive.action,
            "description": action_map.get(directive.action, "Unknown action"),
            "success": True,
            "execution_time": time.time() - start_time
        }
        
        self.execution_log.append({
            'directive_id': directive.id,
            'action': directive.action,
            'success': True,
            'timestamp': datetime.now()
        })
        
        return result


class TBCSPNProcessor:
    """Main TB-CSPN processor for comparative evaluation"""
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        # Initialize components
        self.topic_extractor = TopicExtractor("llm" if use_llm else "keyword", llm_client)
        
        self.rule_engine = RuleEngine()
        self.rule_engine.add_rule(high_ai_relevance_rule)
        self.rule_engine.add_rule(market_volatility_rule)
        self.rule_engine.add_rule(high_relevance_general_rule)
        self.rule_engine.add_rule(default_monitoring_rule)
        
        self.consultant = ConsultantAgent(self.topic_extractor)
        self.supervisor = SupervisorAgent(self.rule_engine)
        self.worker = WorkerAgent()
        
        self.execution_trace = []
    
    def process_news_item(self, news_content: str) -> ProcessingResult:
        """Process news item through TB-CSPN pipeline"""
        start_time = time.time()
        
        try:
            # 1. Consultant: Extract topics
            token = self.consultant.process_input(news_content)
            
            # 2. Supervisor: Evaluate and generate directives
            directives, rule_trace = self.supervisor.evaluate_token(token)
            
            # 3. Worker: Execute first directive (if any)
            action_result = None
            action_taken = None
            directive_text = None
            
            if directives:
                primary_directive = directives[0]  # Take first/highest priority
                action_result = self.worker.execute_directive(primary_directive)
                action_taken = action_result["action_code"]
                directive_text = primary_directive.metadata.get("directive_text")
            
            processing_time = time.time() - start_time
            
            # Log complete execution
            execution_record = {
                'token_id': token.id,
                'input_content': news_content,
                'topics_extracted': token.topics,
                'directives_count': len(directives),
                'action_executed': action_taken,
                'processing_time': processing_time,
                'success': True
            }
            self.execution_trace.append(execution_record)
            
            return ProcessingResult(
                input_text=news_content,
                topics_extracted=token.topics,
                directive=directive_text,
                action_taken=action_taken,
                processing_time=processing_time,
                success=True,
                token_id=token.id,
                rule_trace=rule_trace
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                input_text=news_content,
                topics_extracted={},
                directive=None,
                action_taken=None,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_news = [
        "Federal Reserve signals possible rate hike in July amid inflation concerns.",
        "Retail stocks underperform despite strong holiday sales figures.",
        "Tech sector surges on breakthrough AI chip development from major firms.",
        "Market volatility increases as geopolitical tensions affect trading.",
        "Consumer spending shows resilience in face of economic uncertainty."
    ]
    
    # Initialize TB-CSPN processor
    processor = TBCSPNProcessor(use_llm=False)
    
    # Process test items
    results = []
    for news in test_news:
        result = processor.process_news_item(news)
        results.append(result)
        print(f"Input: {news[:50]}...")
        print(f"Topics: {result.topics_extracted}")
        print(f"Directive: {result.directive}")
        print(f"Action: {result.action_taken}")
        print(f"Time: {result.processing_time:.3f}s")
        print(f"Rules: {result.rule_trace}")
        print(f"Success: {result.success}")
        print("-" * 80)
    
    # Basic statistics
    avg_time = sum(r.processing_time for r in results) / len(results)
    success_rate = sum(1 for r in results if r.success) / len(results)
    
    print(f"\nSummary:")
    print(f"Average processing time: {avg_time:.3f}s")
    print(f"Success rate: {success_rate:.1%}")
