#!/usr/bin/env python3
"""
Simple test of the scaling experiments framework
"""

import sys
import time
import os
from pathlib import Path

# Add the project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the evaluation directory to the path as well
evaluation_dir = project_root / "evaluation"
sys.path.insert(0, str(evaluation_dir))

# Now import our modules
try:
    from scaling_experiments.scaling_config import (
        ScalingConfig, TaskComplexity, CoordinationPattern
    )
    from scaling_experiments.metrics_collector import MetricsCollector
except ImportError as e:
    print(f"Import error: {e}")
    print("Let's try a different import approach...")
    
    # Alternative import method
    sys.path.insert(0, str(current_dir.parent))
    from scaling_config import ScalingConfig, TaskComplexity, CoordinationPattern
    from metrics_collector import MetricsCollector

def run_simple_test():
    """Run a simple test of the framework"""
    
    print("Testing TB-CSPN Scaling Experiments Framework")
    print("=" * 50)
    
    # Create a simple config
    config = ScalingConfig(
        agent_count=5,
        task_complexity=TaskComplexity.SIMPLE,
        concurrency_level=1,
        coordination_pattern=CoordinationPattern.STAR,
        duration_seconds=10,
        task_batch_size=5
    )
    
    print(f"Config: {config.agent_count} agents, {config.task_complexity.value} tasks")
    
    # Test metrics collector
    collector = MetricsCollector()
    collector.start_monitoring()
    
    # Simulate some work
    print("Simulating work...")
    for i in range(3):
        collector.record_task_start(f"task_{i}", 2)
        time.sleep(1)  # Simulate task execution
        collector.record_task_completion(f"task_{i}", True, 1.0)
    
    collector.stop_monitoring()
    
    # Print results
    print("\nResults:")
    print(f"Execution time: {collector.metrics.total_execution_time:.2f}s")
    print(f"Success rate: {collector.metrics.task_success_rate:.2%}")
    print(f"Peak memory: {collector.metrics.memory_usage_peak:.1f} MB")
    print(f"Avg CPU: {collector.metrics.cpu_utilization_avg:.1f}%")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    run_simple_test()