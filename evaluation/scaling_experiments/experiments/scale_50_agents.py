#!/usr/bin/env python3
"""
Scaling experiment with 50 agents - Large Scale Test
This addresses reviewer concerns about large-scale coordination
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the evaluation directory to the path
evaluation_dir = project_root / "evaluation"
sys.path.insert(0, str(evaluation_dir))

# Import our scaling framework
sys.path.insert(0, str(current_dir.parent))
from scaling_framework import ScalingExperimentFramework
from scaling_config import ScalingConfig, TaskComplexity, CoordinationPattern

def run_50_agent_experiment():
    """Run large-scale scaling experiment with 50 agents"""
    
    print("TB-CSPN Scaling Experiment: 50 Agents - Large Scale")
    print("=" * 60)
    print("This experiment addresses reviewer concerns about:")
    print("- Large-scale coordination (50 agents)")
    print("- State-space management at scale")
    print("- Performance degradation analysis")
    print("=" * 60)
    
    # Configuration for 50-agent experiment
    config = ScalingConfig(
        agent_count=50,
        task_complexity=TaskComplexity.MEDIUM,
        concurrency_level=8,                    # Higher concurrency for more agents
        coordination_pattern=CoordinationPattern.HIERARCHICAL,
        duration_seconds=240,                   # More time for larger scale
        task_batch_size=75                      # More tasks to test scaling
    )
    
    print(f"\nConfiguration:")
    print(f"- Agent Count: {config.agent_count}")
    print(f"- Task Complexity: {config.task_complexity.value}")
    print(f"- Concurrency Level: {config.concurrency_level}")
    print(f"- Coordination Pattern: {config.coordination_pattern.value}")
    print(f"- Task Batch Size: {config.task_batch_size}")
    print(f"- Max Duration: {config.duration_seconds}s")
    
    # Initialize experiment framework
    framework = ScalingExperimentFramework(config)
    framework.setup_experiment()
    
    print(f"\nStarting experiment...")
    start_time = time.time()
    
    try:
        # Run the scaling experiment
        results = framework.run_scaling_experiment()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Save results
        framework.save_results()
        
        # Display comprehensive results
        print_detailed_results(results, total_time)
        
        return results
        
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        raise

def print_detailed_results(results, total_time):
    """Print detailed experimental results"""
    
    metrics = results['metrics']
    experiment_results = results['experiment_results']
    
    print(f"\n" + "=" * 60)
    print("EXPERIMENT RESULTS - 50 AGENTS")
    print("=" * 60)
    
    # Performance Metrics
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"Total Execution Time: {metrics['total_execution_time']:.2f}s")
    print(f"Average Task Time: {metrics.get('average_task_time', 0):.3f}s")
    print(f"Task Success Rate: {metrics['task_success_rate']:.1%}")
    print(f"Coordination Overhead: {metrics['coordination_overhead']:.3f}s")
    
    # Resource Metrics
    print(f"\n💾 RESOURCE METRICS")
    print(f"Peak Memory Usage: {metrics['memory_usage_peak']:.1f} MB")
    print(f"Average CPU Usage: {metrics['cpu_utilization_avg']:.1f}%")
    print(f"Peak Thread Count: {metrics.get('thread_count_peak', 0)}")
    
    # Task Execution Metrics
    print(f"\n🎯 TASK EXECUTION METRICS")
    completed = experiment_results.get('completed_tasks', 0)
    failed = experiment_results.get('failed_tasks', 0)
    total_tasks = completed + failed
    print(f"Total Tasks: {total_tasks}")
    print(f"Completed Tasks: {completed}")
    print(f"Failed Tasks: {failed}")
    print(f"Coordination Events: {experiment_results.get('total_coordination_events', 0)}")
    
    # Efficiency Metrics
    print(f"\n⚡ EFFICIENCY METRICS")
    if metrics['total_execution_time'] > 0 and total_tasks > 0:
        throughput = total_tasks / metrics['total_execution_time']
        throughput_per_agent = throughput / 50  # 50 agents
        print(f"Overall Throughput: {throughput:.2f} tasks/second")
        print(f"Throughput per Agent: {throughput_per_agent:.3f} tasks/second/agent")
        
        if metrics['coordination_overhead'] > 0:
            coordination_percentage = (metrics['coordination_overhead'] / metrics['total_execution_time']) * 100
            print(f"Coordination Overhead: {coordination_percentage:.1f}% of total time")
    
    # Scalability Analysis
    print(f"\n📈 SCALABILITY ANALYSIS")
    print(f"Memory per Agent: {metrics['memory_usage_peak'] / 50:.1f} MB/agent")
    if metrics['coordination_overhead'] > 0 and total_tasks > 0:
        coord_per_task = metrics['coordination_overhead'] / total_tasks
        print(f"Coordination Time per Task: {coord_per_task:.3f}s")
    
    # Scaling Comparison (vs previous tests)
    print(f"\n🔍 SCALING INSIGHTS (50 vs previous tests)")
    print(f"1. Agent Scale Factor: 5x vs 10-agent baseline")
    print(f"2. Task Load: {total_tasks} tasks")
    print(f"3. Memory Usage: {metrics['memory_usage_peak']:.1f} MB")
    print(f"4. Success Rate: {metrics['task_success_rate']:.1%}")
    
    # Performance thresholds
    if metrics['task_success_rate'] >= 0.95:
        print("✅ SUCCESS: Meets 95% success rate threshold")
    else:
        print("❌ WARNING: Below 95% success rate threshold")
        
    if metrics['memory_usage_peak'] <= 4096:  # 4GB threshold for 50 agents
        print("✅ SUCCESS: Memory usage within reasonable bounds")
    else:
        print("⚠️  WARNING: High memory usage detected")
    
    print("=" * 60)

if __name__ == "__main__":
    run_50_agent_experiment()