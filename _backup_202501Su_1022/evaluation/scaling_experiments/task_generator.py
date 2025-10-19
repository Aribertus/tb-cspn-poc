import random
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass
from scaling_config import TaskComplexity

@dataclass
class ScalableTask:
    """Represents a task that can scale with agent count"""
    task_id: str
    complexity: TaskComplexity
    required_agents: int
    coordination_steps: int
    parallel_subtasks: int
    estimated_duration: float
    task_data: Dict[str, Any]

class ScalableTaskGenerator:
    """Generates tasks that scale appropriately with agent count"""
    
    def __init__(self, agent_count: int):
        self.agent_count = agent_count
        self.task_counter = 0
        
    def generate_task_batch(self, complexity: TaskComplexity, batch_size: int) -> List[ScalableTask]:
        """Generate a batch of tasks with specified complexity"""
        tasks = []
        
        for i in range(batch_size):
            if complexity == TaskComplexity.SIMPLE:
                task = self._generate_simple_task()
            elif complexity == TaskComplexity.MEDIUM:
                task = self._generate_medium_task()
            else:  # COMPLEX
                task = self._generate_complex_task()
                
            tasks.append(task)
            self.task_counter += 1
            
        return tasks
    
    def _generate_simple_task(self) -> ScalableTask:
        """Generate simple coordination task"""
        required_agents = min(random.randint(1, 3), self.agent_count)
        
        return ScalableTask(
            task_id=f"simple_{self.task_counter:04d}_{str(uuid.uuid4())[:8]}",
            complexity=TaskComplexity.SIMPLE,
            required_agents=required_agents,
            coordination_steps=random.randint(1, 3),
            parallel_subtasks=1,
            estimated_duration=random.uniform(5.0, 15.0),
            task_data={
                'operation_type': random.choice(['data_processing', 'analysis', 'computation']),
                'data_size': random.randint(100, 1000),
                'priority': random.choice(['low', 'medium'])
            }
        )
    
    def _generate_medium_task(self) -> ScalableTask:
        """Generate medium complexity coordination task"""
        required_agents = max(2, min(self.agent_count // 4, 8))
        
        return ScalableTask(
            task_id=f"medium_{self.task_counter:04d}_{str(uuid.uuid4())[:8]}",
            complexity=TaskComplexity.MEDIUM,
            required_agents=required_agents,
            coordination_steps=random.randint(3, 8),
            parallel_subtasks=random.randint(2, 4),
            estimated_duration=random.uniform(15.0, 45.0),
            task_data={
                'operation_type': random.choice(['multi_step_analysis', 'data_fusion', 'optimization']),
                'data_size': random.randint(1000, 10000),
                'dependencies': random.randint(1, 3),
                'priority': random.choice(['medium', 'high'])
            }
        )
    
    def _generate_complex_task(self) -> ScalableTask:
        """Generate complex coordination task"""
        required_agents = max(5, min(self.agent_count // 2, 20))
        
        return ScalableTask(
            task_id=f"complex_{self.task_counter:04d}_{str(uuid.uuid4())[:8]}",
            complexity=TaskComplexity.COMPLEX,
            required_agents=required_agents,
            coordination_steps=random.randint(8, 20),
            parallel_subtasks=random.randint(4, 8),
            estimated_duration=random.uniform(45.0, 120.0),
            task_data={
                'operation_type': random.choice(['distributed_computation', 'complex_workflow', 'simulation']),
                'data_size': random.randint(10000, 100000),
                'dependencies': random.randint(3, 8),
                'coordination_complexity': random.choice(['high', 'very_high']),
                'priority': 'high'
            }
        )
    
    def generate_mixed_workload(self, total_tasks: int) -> List[ScalableTask]:
        """Generate a mixed workload of different complexity tasks"""
        tasks = []
        
        # Distribution: 50% simple, 30% medium, 20% complex
        simple_count = int(total_tasks * 0.5)
        medium_count = int(total_tasks * 0.3)
        complex_count = total_tasks - simple_count - medium_count
        
        tasks.extend(self.generate_task_batch(TaskComplexity.SIMPLE, simple_count))
        tasks.extend(self.generate_task_batch(TaskComplexity.MEDIUM, medium_count))
        tasks.extend(self.generate_task_batch(TaskComplexity.COMPLEX, complex_count))
        
        # Shuffle to avoid predictable patterns
        random.shuffle(tasks)
        return tasks