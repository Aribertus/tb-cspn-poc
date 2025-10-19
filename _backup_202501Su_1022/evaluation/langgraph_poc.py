"""
LangGraph POC Implementation for Financial News Processing
Replicates TB-CSPN workflow using prompt-chained LLM agents
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


@dataclass
class ProcessingResult:
    """Result structure for comparison with TB-CSPN"""
    input_text: str
    topics_extracted: Dict[str, float]
    directive: Optional[str]
    action_taken: Optional[str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class LangGraphState(TypedDict):
    """State passed between LangGraph nodes"""
    messages: List[Any]
    input_text: str
    topics: Dict[str, float]
    directive: Optional[str]
    action: Optional[str]
    metadata: Dict[str, Any]


class LangGraphFinancialProcessor:
    """LangGraph implementation mimicking TB-CSPN architecture"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1
        )
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(LangGraphState)
        
        # Add nodes
        workflow.add_node("consultant", self._consultant_node)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("worker", self._worker_node)
        
        # Add edges
        workflow.set_entry_point("consultant")
        workflow.add_edge("consultant", "supervisor")
        workflow.add_edge("supervisor", "worker")
        workflow.add_edge("worker", END)
        
        return workflow.compile()
    
    def _consultant_node(self, state: LangGraphState) -> LangGraphState:
        """Consultant agent: Extract topics from financial news"""
        
        consultant_prompt = """
        You are a financial news analyst. Extract key topics from the following news text 
        and assign relevance scores from 0.0 to 1.0.
        
        Focus on these topic categories:
        - market_volatility
        - fed_policy  
        - AI_sector
        - tech_momentum
        - retail_sector
        - consumer_spending
        - earnings_outlook
        - regulatory_changes
        
        Text: {text}
        
        Respond with ONLY a JSON object mapping topic names to scores:
        {{"topic_name": 0.85, "another_topic": 0.62}}
        """
        
        try:
            message = HumanMessage(content=consultant_prompt.format(text=state["input_text"]))
            response = self.llm.invoke([message])
            
            # Parse LLM response to extract topics
            topics_text = response.content.strip()
            if topics_text.startswith('```json'):
                topics_text = topics_text.split('```json')[1].split('```')[0]
            elif topics_text.startswith('```'):
                topics_text = topics_text.split('```')[1].split('```')[0]
                
            topics = json.loads(topics_text)
            
            # Ensure scores are floats and within range
            topics = {k: max(0.0, min(1.0, float(v))) for k, v in topics.items()}
            
            state["topics"] = topics
            state["messages"] = add_messages(state["messages"], [message, response])
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to simple keyword extraction
            state["topics"] = self._fallback_topic_extraction(state["input_text"])
            state["metadata"]["consultant_error"] = str(e)
            
        return state
    
    def _supervisor_node(self, state: LangGraphState) -> LangGraphState:
        """Supervisor agent: Evaluate topics and generate directive"""
        
        supervisor_prompt = """
        You are a financial decision supervisor. Based on the following topic scores,
        decide if any action should be taken and generate a directive.
        
        Topics and scores: {topics}
        
        Rules:
        - If any AI or tech topic >= 0.8: "Monitor AI sector for strategic repositioning"
        - If market_volatility >= 0.8 AND fed_policy >= 0.6: "Implement defensive positioning"
        - If any single topic >= 0.85: "Analyze {topic} developments for investment impact"
        - Otherwise: "Continue standard monitoring"
        
        Respond with ONLY the directive text, no additional explanation.
        """
        
        try:
            # Find highest scoring topics
            if not state["topics"]:
                state["directive"] = "Continue standard monitoring"
                return state
                
            max_topic = max(state["topics"].items(), key=lambda x: x[1])
            max_topic_name, max_score = max_topic
            
            # Apply decision rules (hardcoded logic, mimicking TB-CSPN rules)
            ai_tech_score = max(
                state["topics"].get("AI_sector", 0),
                state["topics"].get("tech_momentum", 0)
            )
            
            if ai_tech_score >= 0.8:
                directive = "Monitor AI sector for strategic repositioning"
            elif (state["topics"].get("market_volatility", 0) >= 0.8 and 
                  state["topics"].get("fed_policy", 0) >= 0.6):
                directive = "Implement defensive positioning"
            elif max_score >= 0.85:
                directive = f"Analyze {max_topic_name} developments for investment impact"
            else:
                directive = "Continue standard monitoring"
                
            state["directive"] = directive
            
        except Exception as e:
            state["directive"] = "Continue standard monitoring"
            state["metadata"]["supervisor_error"] = str(e)
            
        return state
    
    def _worker_node(self, state: LangGraphState) -> LangGraphState:
        """Worker agent: Execute action based on directive"""
        
        worker_prompt = """
        You are a financial action executor. Based on the directive: "{directive}"
        
        Determine the appropriate action and respond with ONE of:
        - "MONITOR_AI_SECTOR"
        - "DEFENSIVE_POSITIONING" 
        - "SECTOR_ANALYSIS"
        - "STANDARD_MONITORING"
        - "NO_ACTION"
        
        Respond with ONLY the action code, no additional text.
        """
        
        try:
            directive = state.get("directive", "Continue standard monitoring")
            
            # Simple mapping (could be made more sophisticated)
            if "AI sector" in directive or "tech" in directive.lower():
                action = "MONITOR_AI_SECTOR"
            elif "defensive" in directive.lower():
                action = "DEFENSIVE_POSITIONING"
            elif "Analyze" in directive:
                action = "SECTOR_ANALYSIS"
            elif "standard monitoring" in directive.lower():
                action = "STANDARD_MONITORING"
            else:
                action = "NO_ACTION"
                
            state["action"] = action
            
        except Exception as e:
            state["action"] = "NO_ACTION"
            state["metadata"]["worker_error"] = str(e)
            
        return state
    
    def _fallback_topic_extraction(self, text: str) -> Dict[str, float]:
        """Simple keyword-based fallback when LLM parsing fails"""
        keywords = {
            "market_volatility": ["volatile", "volatility", "uncertain", "swing"],
            "fed_policy": ["fed", "federal reserve", "interest rate", "monetary"],
            "AI_sector": ["AI", "artificial intelligence", "machine learning", "tech"],
            "tech_momentum": ["technology", "tech", "innovation", "digital"],
            "retail_sector": ["retail", "consumer", "shopping", "sales"],
            "consumer_spending": ["spending", "consumption", "purchases", "demand"]
        }
        
        text_lower = text.lower()
        topics = {}
        
        for topic, words in keywords.items():
            score = sum(1 for word in words if word in text_lower) / len(words)
            if score > 0:
                topics[topic] = min(score * 0.7, 1.0)  # Cap at 0.7 for fallback
                
        return topics
    
    def process_news_item(self, news_text: str) -> ProcessingResult:
        """Process a single news item through the LangGraph pipeline"""
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = LangGraphState(
                messages=[],
                input_text=news_text,
                topics={},
                directive=None,
                action=None,
                metadata={"start_time": start_time}
            )
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                input_text=news_text,
                topics_extracted=result.get("topics", {}),
                directive=result.get("directive"),
                action_taken=result.get("action"),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                input_text=news_text,
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
    
    # Initialize processor (you'll need to provide your OpenAI API key)
    processor = LangGraphFinancialProcessor(
        openai_api_key="your-openai-api-key-here"
    )
    
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
        print(f"Success: {result.success}")
        print("-" * 80)
    
    # Basic statistics
    avg_time = sum(r.processing_time for r in results) / len(results)
    success_rate = sum(1 for r in results if r.success) / len(results)
    
    print(f"\nSummary:")
    print(f"Average processing time: {avg_time:.3f}s")
    print(f"Success rate: {success_rate:.1%}")
