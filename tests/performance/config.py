"""
Performance test configuration and utilities.

Provides configuration settings and utilities for performance testing.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os


@dataclass
class PerformanceTestConfig:
    """Configuration for performance tests."""
    
    # Concurrent request test settings
    max_concurrent_requests: int = 50
    concurrent_test_timeout: int = 60
    concurrent_failure_threshold: float = 30.0
    
    # Memory test settings
    memory_sampling_interval: float = 0.5
    max_memory_growth_mb: float = 1000.0
    memory_leak_threshold_mb_per_hour: float = 500.0
    
    # Scalability test settings
    max_agent_instances: int = 10
    scalability_test_duration: int = 30
    load_distribution_variance_threshold: float = 5.0
    
    # System limits test settings
    system_limit_timeout: int = 120
    degradation_threshold_percent: float = 50.0
    recovery_timeout: int = 30
    
    # General test settings
    test_data_size: int = 100
    default_test_timeout: int = 300
    resource_monitoring_enabled: bool = True
    
    @classmethod
    def from_environment(cls) -> 'PerformanceTestConfig':
        """Create configuration from environment variables."""
        return cls(
            max_concurrent_requests=int(os.getenv('PERF_MAX_CONCURRENT', '50')),
            concurrent_test_timeout=int(os.getenv('PERF_CONCURRENT_TIMEOUT', '60')),
            concurrent_failure_threshold=float(os.getenv('PERF_FAILURE_THRESHOLD', '30.0')),
            
            memory_sampling_interval=float(os.getenv('PERF_MEMORY_INTERVAL', '0.5')),
            max_memory_growth_mb=float(os.getenv('PERF_MAX_MEMORY_GROWTH', '1000.0')),
            memory_leak_threshold_mb_per_hour=float(os.getenv('PERF_LEAK_THRESHOLD', '500.0')),
            
            max_agent_instances=int(os.getenv('PERF_MAX_INSTANCES', '10')),
            scalability_test_duration=int(os.getenv('PERF_SCALABILITY_DURATION', '30')),
            load_distribution_variance_threshold=float(os.getenv('PERF_LOAD_VARIANCE', '5.0')),
            
            system_limit_timeout=int(os.getenv('PERF_LIMIT_TIMEOUT', '120')),
            degradation_threshold_percent=float(os.getenv('PERF_DEGRADATION_THRESHOLD', '50.0')),
            recovery_timeout=int(os.getenv('PERF_RECOVERY_TIMEOUT', '30')),
            
            test_data_size=int(os.getenv('PERF_TEST_DATA_SIZE', '100')),
            default_test_timeout=int(os.getenv('PERF_DEFAULT_TIMEOUT', '300')),
            resource_monitoring_enabled=os.getenv('PERF_RESOURCE_MONITORING', 'true').lower() == 'true'
        )


@dataclass
class PerformanceBenchmarks:
    """Performance benchmarks and thresholds."""
    
    # Throughput benchmarks (requests per second)
    min_throughput_rps: float = 1.0
    target_throughput_rps: float = 5.0
    excellent_throughput_rps: float = 10.0
    
    # Response time benchmarks (milliseconds)
    max_response_time_ms: float = 30000.0  # 30 seconds
    target_response_time_ms: float = 10000.0  # 10 seconds
    excellent_response_time_ms: float = 5000.0  # 5 seconds
    
    # Memory usage benchmarks (MB)
    max_memory_per_request_mb: float = 100.0
    target_memory_per_request_mb: float = 50.0
    excellent_memory_per_request_mb: float = 25.0
    
    # Error rate benchmarks (percentage)
    max_error_rate_percent: float = 20.0
    target_error_rate_percent: float = 10.0
    excellent_error_rate_percent: float = 5.0
    
    # Scalability benchmarks
    min_scaling_efficiency_percent: float = 50.0
    target_scaling_efficiency_percent: float = 70.0
    excellent_scaling_efficiency_percent: float = 85.0
    
    def evaluate_performance(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Evaluate performance metrics against benchmarks."""
        evaluation = {}
        
        # Evaluate throughput
        throughput = metrics.get('throughput_rps', 0)
        if throughput >= self.excellent_throughput_rps:
            evaluation['throughput'] = 'excellent'
        elif throughput >= self.target_throughput_rps:
            evaluation['throughput'] = 'good'
        elif throughput >= self.min_throughput_rps:
            evaluation['throughput'] = 'acceptable'
        else:
            evaluation['throughput'] = 'poor'
        
        # Evaluate response time
        response_time = metrics.get('average_response_time_ms', float('inf'))
        if response_time <= self.excellent_response_time_ms:
            evaluation['response_time'] = 'excellent'
        elif response_time <= self.target_response_time_ms:
            evaluation['response_time'] = 'good'
        elif response_time <= self.max_response_time_ms:
            evaluation['response_time'] = 'acceptable'
        else:
            evaluation['response_time'] = 'poor'
        
        # Evaluate memory usage
        memory_per_request = metrics.get('memory_per_request_mb', float('inf'))
        if memory_per_request <= self.excellent_memory_per_request_mb:
            evaluation['memory'] = 'excellent'
        elif memory_per_request <= self.target_memory_per_request_mb:
            evaluation['memory'] = 'good'
        elif memory_per_request <= self.max_memory_per_request_mb:
            evaluation['memory'] = 'acceptable'
        else:
            evaluation['memory'] = 'poor'
        
        # Evaluate error rate
        error_rate = metrics.get('error_rate_percent', 100)
        if error_rate <= self.excellent_error_rate_percent:
            evaluation['error_rate'] = 'excellent'
        elif error_rate <= self.target_error_rate_percent:
            evaluation['error_rate'] = 'good'
        elif error_rate <= self.max_error_rate_percent:
            evaluation['error_rate'] = 'acceptable'
        else:
            evaluation['error_rate'] = 'poor'
        
        return evaluation


class PerformanceTestUtils:
    """Utilities for performance testing."""
    
    @staticmethod
    def calculate_percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile value from a list of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a list of values."""
        if not values:
            return {
                'count': 0,
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'std_dev': 0.0
            }
        
        import statistics
        
        sorted_values = sorted(values)
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': PerformanceTestUtils.calculate_percentile(values, 95),
            'p99': PerformanceTestUtils.calculate_percentile(values, 99),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"
    
    @staticmethod
    def format_memory(mb: float) -> str:
        """Format memory size in human-readable format."""
        if mb < 1:
            return f"{mb * 1024:.1f}KB"
        elif mb < 1024:
            return f"{mb:.1f}MB"
        else:
            gb = mb / 1024
            return f"{gb:.2f}GB"
    
    @staticmethod
    def format_rate(rate: float, unit: str = "RPS") -> str:
        """Format rate in human-readable format."""
        if rate < 1:
            return f"{rate:.3f} {unit}"
        elif rate < 10:
            return f"{rate:.2f} {unit}"
        else:
            return f"{rate:.1f} {unit}"


# Global configuration instance
PERF_CONFIG = PerformanceTestConfig.from_environment()
PERF_BENCHMARKS = PerformanceBenchmarks()