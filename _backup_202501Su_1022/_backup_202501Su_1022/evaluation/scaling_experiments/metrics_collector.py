import time
import psutil
import threading
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from scaling_config import ScalingMetrics

@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    memory_mb: float
    thread_count: int

class MetricsCollector:
    """Collects performance and resource metrics during scaling experiments"""
    
    def __init__(self):
        self.metrics = ScalingMetrics()
        self.resource_snapshots: List[ResourceSnapshot] = []
        self.coordination_events: List[Dict[str, Any]] = []
        self.task_events: List[Dict[str, Any]] = []
        self.monitoring = False
        self.start_time = None
        
    def start_monitoring(self):
        """Start background resource monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and finalize metrics"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        self._calculate_final_metrics()
        
    def _monitor_resources(self):
        """Background thread for resource monitoring"""
        process = psutil.Process()
        while self.monitoring:
            try:
                snapshot = ResourceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(interval=None),
                    memory_mb=process.memory_info().rss / 1024 / 1024,
                    thread_count=threading.active_count()
                )
                self.resource_snapshots.append(snapshot)
            except Exception as e:
                print(f"Warning: Resource monitoring error: {e}")
            time.sleep(1)
            
    def record_task_start(self, task_id: str, agent_count: int):
        """Record when a task starts"""
        self.task_events.append({
            'event_type': 'task_start',
            'task_id': task_id,
            'timestamp': time.time(),
            'agent_count': agent_count
        })
        
    def record_task_completion(self, task_id: str, success: bool, duration: float):
        """Record task completion"""
        self.task_events.append({
            'event_type': 'task_completion',
            'task_id': task_id,
            'timestamp': time.time(),
            'success': success,
            'duration': duration
        })
        
    def _calculate_final_metrics(self):
        """Calculate final metrics from collected data"""
        if self.start_time:
            self.metrics.total_execution_time = time.time() - self.start_time
            
        if self.resource_snapshots:
            self.metrics.memory_usage_peak = max(s.memory_mb for s in self.resource_snapshots)
            self.metrics.cpu_utilization_avg = sum(s.cpu_percent for s in self.resource_snapshots) / len(self.resource_snapshots)
            
        completed_tasks = [e for e in self.task_events if e['event_type'] == 'task_completion']
        if completed_tasks:
            successful_tasks = [t for t in completed_tasks if t['success']]
            self.metrics.task_success_rate = len(successful_tasks) / len(completed_tasks)

    def save_results(self, filepath: str, config: Dict[str, Any] = None):
        """Save metrics and raw data to file"""
        import json
        results = {
            'config': config or {},
            'metrics': self.metrics.to_dict(),
            'raw_data': {
                'resource_snapshots': [
                    {
                        'timestamp': s.timestamp,
                        'cpu_percent': s.cpu_percent,
                        'memory_mb': s.memory_mb,
                        'thread_count': s.thread_count
                    }
                    for s in self.resource_snapshots
                ],
                'coordination_events': self.coordination_events,
                'task_events': self.task_events
            }
        }
        
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)