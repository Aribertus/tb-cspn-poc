import time
import logging
import random
from typing import Dict, Any, List
from pathlib import Path

from scaling_config import ScalingConfig, ScalingMetrics
from metrics_collector import MetricsCollector
from task_generator import ScalableTaskGenerator, ScalableTask, TaskComplexity

class ScalingExperimentFramework:
    """Framework for conducting systematic scaling experiments"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.task_generator = ScalableTaskGenerator(config.agent_count)
        self.system = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the experiment"""
        logger = logging.getLogger(f"scaling_experiment_{self.config.agent_count}")
        logger.setLevel(logging.INFO)
        
        # Create handler if it doesn't exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def setup_experiment(self):
        """Initialize the TB-CSPN system for the experiment"""
        self.logger.info(f"Setting up experiment with {self.config.agent_count} agents")
        
        # For now, we'll use a mock system
        # TODO: Replace with actual TB-CSPN system initialization
        self.system = MockTBCSPNSystem(self.config.agent_count)
        self.logger.info(f"System initialized with {self.config.agent_count} agents")
        
    def run_scaling_experiment(self) -> Dict[str, Any]:
        """Execute the complete scaling experiment"""
        self.logger.info("Starting scaling experiment")
        
        try:
            # Start metrics collection
            self.metrics_collector.start_monitoring()
            
            # Generate tasks
            tasks = self.task_generator.generate_mixed_workload(self.config.task_batch_size)
            self.logger.info(f"Generated {len(tasks)} tasks")
            
            # Execute experiment
            experiment_results = self._execute_tasks(tasks)
            
            # Calculate coordination efficiency
            self._calculate_coordination_metrics()
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise
        finally:
            # Stop monitoring and finalize metrics
            self.metrics_collector.stop_monitoring()
            
        # Compile and return results
        results = self._compile_results(experiment_results)
        self.logger.info("Scaling experiment completed")
        
        return results
        
    def _execute_tasks(self, tasks: List[ScalableTask]) -> Dict[str, Any]:
        """Execute the generated tasks"""
        results = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_coordination_events': 0
        }
        
        for i, task in enumerate(tasks):
            task_start_time = time.time()
            
            # Record task start
            self.metrics_collector.record_task_start(
                task.task_id, 
                task.required_agents
            )
            
            try:
                # Simulate task execution based on complexity
                success = self._simulate_task_execution(task)
                
                task_duration = time.time() - task_start_time
                
                # Record task completion
                self.metrics_collector.record_task_completion(
                    task.task_id,
                    success,
                    task_duration
                )
                
                if success:
                    results['completed_tasks'] += 1
                else:
                    results['failed_tasks'] += 1
                    
                # Simulate coordination overhead based on agent count and complexity
                coordination_time = self._calculate_coordination_overhead(task)
                
                results['total_coordination_events'] += 1
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(tasks)} tasks")
                
            except Exception as e:
                self.logger.error(f"Task {task.task_id} failed: {e}")
                self.metrics_collector.record_task_completion(task.task_id, False, time.time() - task_start_time)
                results['failed_tasks'] += 1
                
        return results
        
    def _simulate_task_execution(self, task: ScalableTask) -> bool:
        """Simulate task execution - replace with actual TB-CSPN implementation"""
        
        # Simulate processing time based on task complexity
        base_time = {
            TaskComplexity.SIMPLE: 0.1,
            TaskComplexity.MEDIUM: 0.3,
            TaskComplexity.COMPLEX: 0.8
        }[task.complexity]
        
        # Add some randomness and scale with agent count
        agent_factor = 1.0 + (task.required_agents / self.config.agent_count) * 0.5
        processing_time = base_time * agent_factor * random.uniform(0.8, 1.2)
        
        time.sleep(processing_time)
        
        # Simulate failure rates based on complexity and agent count
        base_failure_rate = {
            TaskComplexity.SIMPLE: 0.02,    # 2% failure rate
            TaskComplexity.MEDIUM: 0.05,    # 5% failure rate  
            TaskComplexity.COMPLEX: 0.10    # 10% failure rate
        }[task.complexity]
        
        # Higher failure rate with more agents (coordination complexity)
        coordination_penalty = (task.required_agents / self.config.agent_count) * 0.03
        failure_rate = min(base_failure_rate + coordination_penalty, 0.15)  # Cap at 15%
        
        return random.random() > failure_rate
        
    def _calculate_coordination_overhead(self, task: ScalableTask) -> float:
        """Calculate coordination overhead for a task"""
        
        # Base coordination time depends on number of agents
        base_coordination = 0.01 * task.required_agents  # 10ms per agent
        
        # Complexity multiplier
        complexity_multiplier = {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MEDIUM: 2.0,
            TaskComplexity.COMPLEX: 4.0
        }[task.complexity]
        
        # Coordination pattern affects overhead
        pattern_multiplier = {
            'star': 1.0,
            'hierarchical': 1.5,
            'mesh': 2.0
        }.get(self.config.coordination_pattern.value, 1.0)
        
        coordination_time = base_coordination * complexity_multiplier * pattern_multiplier
        
        # Add some realistic variance
        coordination_time *= random.uniform(0.8, 1.2)
        
        return coordination_time
        
    def _calculate_coordination_metrics(self):
        """Calculate coordination-specific metrics"""
        coordination_events = self.metrics_collector.coordination_events
        
        if coordination_events:
            total_coordination_time = sum(e['processing_time'] for e in coordination_events)
            self.metrics_collector.metrics.coordination_overhead = total_coordination_time
            
    def _compile_results(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final experiment results"""
        return {
            'config': {
                'agent_count': self.config.agent_count,
                'task_complexity': self.config.task_complexity.value,
                'concurrency_level': self.config.concurrency_level,
                'coordination_pattern': self.config.coordination_pattern.value,
                'task_batch_size': self.config.task_batch_size
            },
            'metrics': self.metrics_collector.metrics.to_dict(),
            'experiment_results': experiment_results,
            'timestamp': time.time()
        }
        
    def save_results(self, output_dir: str = "results"):
        """Save experiment results to file"""
        results = self._compile_results({})
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate filename
        filename = f"scaling_{self.config.agent_count}agents_{int(time.time())}.json"
        filepath = output_path / filename
        
        # Save using metrics collector
        self.metrics_collector.save_results(str(filepath), results['config'])
        self.logger.info(f"Results saved to {filepath}")

class MockTBCSPNSystem:
    """Mock TB-CSPN system for testing - replace with actual implementation"""
    
    def __init__(self, agent_count: int):
        self.agent_count = agent_count
        self.agents = [f"agent_{i:03d}" for i in range(agent_count)]
        
    def execute_task(self, task):
        """Mock task execution"""
        # This will be replaced with actual TB-CSPN task execution
        return True