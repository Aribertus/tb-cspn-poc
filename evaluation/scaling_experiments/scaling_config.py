
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium" 
    COMPLEX = "complex"

class CoordinationPattern(Enum):
    STAR = "star"
    MESH = "mesh"
    HIERARCHICAL = "hierarchical"

@dataclass
class ScalingConfig:
    """Configuration for scaling experiments"""
    agent_count: int
    task_complexity: TaskComplexity
    concurrency_level: int
    coordination_pattern: CoordinationPattern
    duration_seconds: int = 300
    task_batch_size: int = 50

@dataclass 
class ScalingMetrics:
    """Metrics collected during scaling experiments"""
    total_execution_time: float = 0.0
    coordination_overhead: float = 0.0
    memory_usage_peak: float = 0.0
    cpu_utilization_avg: float = 0.0
    task_success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_execution_time': self.total_execution_time,
            'coordination_overhead': self.coordination_overhead,
            'memory_usage_peak': self.memory_usage_peak,
            'cpu_utilization_avg': self.cpu_utilization_avg,
            'task_success_rate': self.task_success_rate
        }