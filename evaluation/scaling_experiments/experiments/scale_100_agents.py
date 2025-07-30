#!/usr/bin/env python3
"""
Scaling experiment with 100 agents - Ultimate Stress Test
This addresses reviewer concerns about extreme-scale coordination
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

def run_100_agent_stress_test():
    """Run ultimate stress test with 100 agents"""
    
    print("TB-CSPN Scaling Experiment: 100 Agents - ULTIMATE STRESS TEST")
    print("=" * 70)
    print("This experiment addresses reviewer concerns about:")
    print("- Extreme-scale coordination (100 agents)")
    print("- State-space explosion at maximum scale")
    print("- System stability under heavy load")
    print("- Memory and performance limits")
    print("=" * 70)
    
    # Configuration for 100-agent stress test
    config = ScalingConfig(
        agent_count=100,
        task_complexity=TaskComplexity.MEDIUM,
        concurrency_level=12,                   # Maximum concurrency
        coordination_pattern=CoordinationPattern.HIERARCHICAL,
        duration_seconds=300,                   # 5 minutes max
        task_batch_size=100                     # 100 tasks for ultimate test
    )
    
    print(f"\nConfiguration:")
    print(f"- Agent Count: {config.agent_count} (10x baseline)")
    print(f"- Task Complexity: {config.task_complexity.value}")
    print(f"- Concurrency Level: {config.concurrency_level}")
    print(f"- Coordination Pattern: {config.coordination_pattern.value}")
    print(f"- Task Batch Size: {config.task_batch_size}")
    print(f"- Max Duration: {config.duration_seconds}s")
    
    # Initialize experiment framework
    framework = ScalingExperimentFramework(config)
    framework.setup_experiment()
    
    print(f"\nStarting ULTIMATE STRESS TEST...")
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
        print(f"\nStress test failed: {e}")
        raise

def print_detailed_results(results, total_time):
    """Print detailed experimental results"""
    
    metrics = results['metrics']
    experiment_results = results['experiment_results']
    
    print(f"\n" + "=" * 70)
    print("ULTIMATE STRESS TEST RESULTS - 100 AGENTS")
    print("=" * 70)
    
    # Performance Metrics
    print(f"\n🚀 PERFORMANCE METRICS")
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
        throughput_per_agent = throughput / 100  # 100 agents
        print(f"Overall Throughput: {throughput:.2f} tasks/second")
        print(f"Throughput per Agent: {throughput_per_agent:.3f} tasks/second/agent")
        
        if metrics['coordination_overhead'] > 0:
            coordination_percentage = (metrics['coordination_overhead'] / metrics['total_execution_time']) * 100
            print(f"Coordination Overhead: {coordination_percentage:.1f}% of total time")
    
    # Scalability Analysis
    print(f"\n📈 SCALABILITY ANALYSIS")
    print(f"Memory per Agent: {metrics['memory_usage_peak'] / 100:.2f} MB/agent")
    if metrics['coordination_overhead'] > 0 and total_tasks > 0:
        coord_per_task = metrics['coordination_overhead'] / total_tasks
        print(f"Coordination Time per Task: {coord_per_task:.3f}s")
    
    # Ultimate Scaling Insights
    print(f"\n🏆 ULTIMATE SCALING INSIGHTS")
    print(f"1. Agent Scale Factor: 10x vs baseline (100 vs 10 agents)")
    print(f"2. Task Load: {total_tasks} tasks")
    print(f"3. Memory Usage: {metrics['memory_usage_peak']:.1f} MB")
    print(f"4. Success Rate: {metrics['task_success_rate']:.1%}")
    print(f"5. System Stability: {'STABLE' if metrics['task_success_rate'] > 0.85 else 'UNSTABLE'}")
    
    # Stress Test Evaluation
    print(f"\n🔥 STRESS TEST EVALUATION")
    if metrics['task_success_rate'] >= 0.95:
        print("✅ OUTSTANDING: Exceeds 95% success rate threshold")
    elif metrics['task_success_rate'] >= 0.90:
        print("✅ EXCELLENT: Meets high-performance threshold (90%+)")
    elif metrics['task_success_rate'] >= 0.85:
        print("✅ GOOD: Acceptable performance under stress")
    else:
        print("❌ WARNING: Performance degradation detected")
        
    if metrics['memory_usage_peak'] <= 8192:  # 8GB threshold for 100 agents
        print("✅ OUTSTANDING: Memory usage within enterprise bounds")
    else:
        print("⚠️  WARNING: High memory usage - monitor for production")
    
    print("=" * 70)

if __name__ == "__main__":
    run_100_agent_stress_test()