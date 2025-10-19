"""
TB-CSPN Scaling Experiments Package

This package provides tools for conducting systematic scaling experiments
to evaluate TB-CSPN performance across different agent counts and scenarios.
"""

from scaling_config import ScalingConfig, ScalingMetrics, TaskComplexity, CoordinationPattern
from metrics_collector import MetricsCollector

__all__ = [
    'ScalingConfig',
    'ScalingMetrics', 
    'TaskComplexity',
    'CoordinationPattern',
    'MetricsCollector'
]

__version__ = "1.0.0"