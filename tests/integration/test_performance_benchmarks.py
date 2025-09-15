"""
Performance benchmarking tests for Agent Scrivener.

Tests system performance under various load conditions, measures throughput,
latency, and resource usage.
"""

import pytest
import asyncio
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tests.integration.test_framework import (
    IntegrationTestFramework, MockServiceConfig, TestDataGenerator
)
from agent_scrivener.models.core import TaskStatus


@dataclass
class PerformanceMetrics:
    """Performance metrics for benchmarking."""
    throughput_requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate_percent: float
    concurrent_sessions: int
    total_requests: int


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage."""
        return {
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "cpu_percent": self.process.cpu_percent(),
            "memory_percent": self.process.memory_percent()
        }
    
    async def measure_latency(self, operation, *args, **kwargs) -> Tuple[Any, float]:
        """Measure operation latency in milliseconds."""
        start_time = time.perf_counter()
        result = await operation(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return result, latency_ms
    
    def calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentile statistics."""
        if not values:
            return {"p50": 0, "p95": 0, "p99": 0}
        
        sorted_values = sorted(values)
        return {
            "p50": statistics.median(sorted_values),
            "p95": sorted_values[int(len(sorted_values) * 0.95)],
            "p99": sorted_values[int(len(sorted_values) * 0.99)]
        }


class TestPerformanceBenchmarks:
    """Performance benchmark test cases."""
    
    @pytest.mark.asyncio
    async def test_single_session_baseline_performance(self):
        """Establish baseline performance for single research session."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        benchmark = PerformanceBenchmark()
        
        async with framework.test_environment():
            # Measure baseline system metrics
            baseline_metrics = benchmark.get_system_metrics()
            
            # Execute single research session
            query = "Machine learning performance optimization"
            start_time = time.perf_counter()
            
            results = await framework.run_end_to_end_test(query)
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Measure final system metrics
            final_metrics = benchmark.get_system_metrics()
            
            session = results["session"]
            metrics = results["metrics"]
            
            # Verify successful completion
            assert session.status == TaskStatus.COMPLETED
            assert metrics.success_rate == 100.0
            
            # Performance assertions
            assert execution_time_ms < 5000  # Should complete within 5 seconds
            assert final_metrics["memory_mb"] - baseline_metrics["memory_mb"] < 100  # Memory increase < 100MB
            assert metrics.error_count == 0
            
            # Log baseline metrics for comparison
            print(f"\nBaseline Performance Metrics:")
            print(f"Execution time: {execution_time_ms:.2f}ms")
            print(f"Memory usage: {final_metrics['memory_mb']:.2f}MB")
            print(f"CPU usage: {final_metrics['cpu_percent']:.2f}%")
            print(f"Success rate: {metrics.success_rate:.2f}%")
    
    @pytest.mark.asyncio
    async def test_concurrent_session_performance(self):
        """Test performance with multiple concurrent research sessions."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.0,
            rate_limit_enabled=False
        )
        
        framework = IntegrationTestFramework(config)
        benchmark = PerformanceBenchmark()
        
        async with framework.test_environment():
            # Test with increasing concurrency levels
            concurrency_levels = [1, 3, 5, 10]
            performance_results = []
            
            for concurrency in concurrency_levels:
                print(f"\nTesting concurrency level: {concurrency}")
                
                # Generate test queries
                queries = TestDataGenerator.generate_research_queries()[:concurrency]
                
                # Measure system metrics before test
                start_metrics = benchmark.get_system_metrics()
                start_time = time.perf_counter()
                
                # Execute concurrent sessions
                tasks = []
                for query in queries:
                    task = asyncio.create_task(framework.run_end_to_end_test(query))
                    tasks.append(task)
                
                # Wait for all sessions to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.perf_counter()
                end_metrics = benchmark.get_system_metrics()
                
                # Calculate performance metrics
                total_time_seconds = end_time - start_time
                successful_sessions = sum(1 for r in results if not isinstance(r, Exception) and r["session"].status == TaskStatus.COMPLETED)
                failed_sessions = concurrency - successful_sessions
                
                throughput = successful_sessions / total_time_seconds if total_time_seconds > 0 else 0
                error_rate = (failed_sessions / concurrency) * 100
                
                # Collect latency data
                latencies = []
                for result in results:
                    if not isinstance(result, Exception):
                        latencies.append(result["metrics"].execution_time_ms)
                
                percentiles = benchmark.calculate_percentiles(latencies)
                
                metrics = PerformanceMetrics(
                    throughput_requests_per_second=throughput,
                    average_latency_ms=statistics.mean(latencies) if latencies else 0,
                    p95_latency_ms=percentiles["p95"],
                    p99_latency_ms=percentiles["p99"],
                    memory_usage_mb=end_metrics["memory_mb"] - start_metrics["memory_mb"],
                    cpu_usage_percent=end_metrics["cpu_percent"],
                    error_rate_percent=error_rate,
                    concurrent_sessions=concurrency,
                    total_requests=concurrency
                )
                
                performance_results.append(metrics)
                
                # Performance assertions
                assert error_rate <= 10.0  # Error rate should be <= 10%
                assert throughput > 0  # Should have some throughput
                assert metrics.memory_usage_mb < 500  # Memory usage should be reasonable
                
                print(f"Throughput: {throughput:.2f} sessions/sec")
                print(f"Average latency: {metrics.average_latency_ms:.2f}ms")
                print(f"P95 latency: {metrics.p95_latency_ms:.2f}ms")
                print(f"Error rate: {error_rate:.2f}%")
                print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")
            
            # Verify performance scaling
            # Throughput should not degrade significantly with increased concurrency
            baseline_throughput = performance_results[0].throughput_requests_per_second
            for metrics in performance_results[1:]:
                # Allow some degradation but not more than 50%
                assert metrics.throughput_requests_per_second >= baseline_throughput * 0.5
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self):
        """Test performance under sustained load over time."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.05,  # 5% failure rate for realistic testing
            rate_limit_enabled=True,
            rate_limit_requests_per_second=20
        )
        
        framework = IntegrationTestFramework(config)
        benchmark = PerformanceBenchmark()
        
        async with framework.test_environment():
            # Run sustained load for 30 seconds
            test_duration_seconds = 30
            concurrent_sessions = 3
            
            queries = TestDataGenerator.generate_research_queries()
            query_index = 0
            
            start_time = time.perf_counter()
            active_tasks = []
            completed_sessions = []
            performance_snapshots = []
            
            while time.perf_counter() - start_time < test_duration_seconds:
                # Maintain target concurrency
                while len(active_tasks) < concurrent_sessions:
                    query = queries[query_index % len(queries)]
                    query_index += 1
                    
                    task = asyncio.create_task(framework.run_end_to_end_test(query))
                    active_tasks.append(task)
                
                # Check for completed tasks
                done_tasks = [task for task in active_tasks if task.done()]
                for task in done_tasks:
                    active_tasks.remove(task)
                    try:
                        result = await task
                        completed_sessions.append(result)
                    except Exception as e:
                        print(f"Task failed: {e}")
                
                # Take performance snapshot every 5 seconds
                current_time = time.perf_counter() - start_time
                if len(performance_snapshots) == 0 or current_time - performance_snapshots[-1]["timestamp"] >= 5:
                    metrics = benchmark.get_system_metrics()
                    performance_snapshots.append({
                        "timestamp": current_time,
                        "memory_mb": metrics["memory_mb"],
                        "cpu_percent": metrics["cpu_percent"],
                        "completed_sessions": len(completed_sessions),
                        "active_tasks": len(active_tasks)
                    })
                
                await asyncio.sleep(0.5)  # Brief pause
            
            # Wait for remaining tasks to complete
            if active_tasks:
                remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)
                for result in remaining_results:
                    if not isinstance(result, Exception):
                        completed_sessions.append(result)
            
            # Analyze sustained load performance
            total_time = time.perf_counter() - start_time
            successful_sessions = sum(1 for r in completed_sessions if r["session"].status == TaskStatus.COMPLETED)
            failed_sessions = len(completed_sessions) - successful_sessions
            
            throughput = successful_sessions / total_time
            error_rate = (failed_sessions / len(completed_sessions)) * 100 if completed_sessions else 0
            
            # Analyze performance stability over time
            memory_values = [snapshot["memory_mb"] for snapshot in performance_snapshots]
            cpu_values = [snapshot["cpu_percent"] for snapshot in performance_snapshots]
            
            memory_stability = statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            cpu_stability = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            
            # Performance assertions
            assert throughput > 0.1  # Should maintain reasonable throughput
            assert error_rate <= 20.0  # Error rate should be acceptable
            assert memory_stability < 50  # Memory usage should be stable (< 50MB std dev)
            assert max(memory_values) < 1000  # Memory should not exceed 1GB
            
            print(f"\nSustained Load Performance:")
            print(f"Duration: {total_time:.2f}s")
            print(f"Completed sessions: {len(completed_sessions)}")
            print(f"Successful sessions: {successful_sessions}")
            print(f"Throughput: {throughput:.2f} sessions/sec")
            print(f"Error rate: {error_rate:.2f}%")
            print(f"Memory stability (std dev): {memory_stability:.2f}MB")
            print(f"CPU stability (std dev): {cpu_stability:.2f}%")
    
    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self):
        """Test memory usage patterns and detect memory leaks."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        benchmark = PerformanceBenchmark()
        
        async with framework.test_environment():
            # Measure memory usage over multiple sessions
            memory_measurements = []
            queries = TestDataGenerator.generate_research_queries()
            
            # Initial memory measurement
            initial_memory = benchmark.get_system_metrics()["memory_mb"]
            memory_measurements.append(initial_memory)
            
            # Run multiple sessions and measure memory after each
            for i in range(10):
                query = queries[i % len(queries)]
                results = await framework.run_end_to_end_test(query)
                
                # Verify session completed successfully
                assert results["session"].status == TaskStatus.COMPLETED
                
                # Measure memory after session
                current_memory = benchmark.get_system_metrics()["memory_mb"]
                memory_measurements.append(current_memory)
                
                # Brief pause to allow garbage collection
                await asyncio.sleep(0.1)
            
            # Analyze memory usage patterns
            memory_increases = [
                memory_measurements[i] - memory_measurements[i-1] 
                for i in range(1, len(memory_measurements))
            ]
            
            final_memory = memory_measurements[-1]
            total_memory_increase = final_memory - initial_memory
            average_increase_per_session = statistics.mean(memory_increases)
            
            # Memory leak detection
            # If memory consistently increases, it might indicate a leak
            increasing_trend = sum(1 for increase in memory_increases if increase > 5)  # > 5MB increase
            leak_indicator = increasing_trend / len(memory_increases)
            
            # Performance assertions
            assert total_memory_increase < 200  # Total increase should be < 200MB
            assert average_increase_per_session < 20  # Average increase should be < 20MB per session
            assert leak_indicator < 0.7  # Less than 70% of sessions should show significant memory increase
            
            print(f"\nMemory Usage Analysis:")
            print(f"Initial memory: {initial_memory:.2f}MB")
            print(f"Final memory: {final_memory:.2f}MB")
            print(f"Total increase: {total_memory_increase:.2f}MB")
            print(f"Average increase per session: {average_increase_per_session:.2f}MB")
            print(f"Potential leak indicator: {leak_indicator:.2f}")
    
    @pytest.mark.asyncio
    async def test_latency_distribution(self):
        """Test latency distribution and identify performance outliers."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        benchmark = PerformanceBenchmark()
        
        async with framework.test_environment():
            # Collect latency data from multiple sessions
            latencies = []
            queries = TestDataGenerator.generate_research_queries()
            
            for i in range(20):  # Run 20 sessions for statistical significance
                query = queries[i % len(queries)]
                
                start_time = time.perf_counter()
                results = await framework.run_end_to_end_test(query)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                # Verify session completed
                assert results["session"].status == TaskStatus.COMPLETED
            
            # Calculate latency statistics
            percentiles = benchmark.calculate_percentiles(latencies)
            mean_latency = statistics.mean(latencies)
            std_dev = statistics.stdev(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # Identify outliers (values > 2 standard deviations from mean)
            outliers = [lat for lat in latencies if abs(lat - mean_latency) > 2 * std_dev]
            outlier_rate = len(outliers) / len(latencies)
            
            # Performance assertions
            assert mean_latency < 3000  # Average latency should be < 3 seconds
            assert percentiles["p95"] < 5000  # 95th percentile should be < 5 seconds
            assert percentiles["p99"] < 8000  # 99th percentile should be < 8 seconds
            assert outlier_rate < 0.1  # Less than 10% outliers
            assert max_latency < 10000  # No request should take more than 10 seconds
            
            print(f"\nLatency Distribution Analysis:")
            print(f"Mean latency: {mean_latency:.2f}ms")
            print(f"Standard deviation: {std_dev:.2f}ms")
            print(f"Min latency: {min_latency:.2f}ms")
            print(f"Max latency: {max_latency:.2f}ms")
            print(f"P50 latency: {percentiles['p50']:.2f}ms")
            print(f"P95 latency: {percentiles['p95']:.2f}ms")
            print(f"P99 latency: {percentiles['p99']:.2f}ms")
            print(f"Outlier rate: {outlier_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self):
        """Test performance under rate limiting conditions."""
        config = MockServiceConfig(
            web_search_delay=0.001,
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.0,
            rate_limit_enabled=True,
            rate_limit_requests_per_second=5  # Aggressive rate limiting
        )
        
        framework = IntegrationTestFramework(config)
        benchmark = PerformanceBenchmark()
        
        async with framework.test_environment():
            # Test with rate limiting
            queries = ["Rate limit test query"] * 10
            
            start_time = time.perf_counter()
            
            # Execute sessions that will hit rate limits
            tasks = []
            for query in queries:
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            successful_sessions = sum(1 for r in results if r["session"].status == TaskStatus.COMPLETED)
            
            # With rate limiting, execution should take longer
            expected_min_time = len(queries) / config.rate_limit_requests_per_second
            
            # Performance assertions
            assert successful_sessions == len(queries)  # All should complete successfully
            assert total_time >= expected_min_time * 0.8  # Should respect rate limits (allow some variance)
            
            # Verify rate limiting was effective
            actual_rate = successful_sessions / total_time
            assert actual_rate <= config.rate_limit_requests_per_second * 1.5  # Allow some variance
            
            print(f"\nRate Limiting Performance:")
            print(f"Configured rate limit: {config.rate_limit_requests_per_second} req/sec")
            print(f"Actual rate: {actual_rate:.2f} req/sec")
            print(f"Total time: {total_time:.2f}s")
            print(f"Expected min time: {expected_min_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])