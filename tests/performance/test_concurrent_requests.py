"""
Concurrent request handling performance tests.

Tests system performance with multiple simultaneous requests,
measuring throughput, latency, and resource utilization.
"""

import pytest
import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from tests.integration.test_framework import (
    IntegrationTestFramework, MockServiceConfig, TestDataGenerator
)
from agent_scrivener.models.core import TaskStatus, SessionState


@dataclass
class ConcurrencyMetrics:
    """Metrics for concurrent request performance."""
    concurrent_requests: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_threads: int
    peak_memory_mb: float
    memory_growth_mb: float


@dataclass
class ResourceMonitor:
    """Monitor system resources during testing."""
    process: psutil.Process = field(default_factory=lambda: psutil.Process())
    initial_memory_mb: float = 0
    peak_memory_mb: float = 0
    memory_samples: List[float] = field(default_factory=list)
    cpu_samples: List[float] = field(default_factory=list)
    monitoring: bool = False
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = self.initial_memory_mb
        self.monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Monitor resources in background thread."""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                
                time.sleep(0.5)  # Sample every 500ms
            except Exception:
                break
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        if not self.memory_samples:
            return {
                "current_memory_mb": self.initial_memory_mb,
                "peak_memory_mb": self.initial_memory_mb,
                "memory_growth_mb": 0,
                "average_cpu_percent": 0,
                "peak_cpu_percent": 0
            }
        
        return {
            "current_memory_mb": self.memory_samples[-1],
            "peak_memory_mb": self.peak_memory_mb,
            "memory_growth_mb": self.peak_memory_mb - self.initial_memory_mb,
            "average_cpu_percent": sum(self.cpu_samples) / len(self.cpu_samples),
            "peak_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0
        }


class ConcurrentRequestTester:
    """Test concurrent request handling capabilities."""
    
    def __init__(self, framework: IntegrationTestFramework):
        self.framework = framework
        self.resource_monitor = ResourceMonitor()
    
    async def test_concurrent_requests(
        self,
        concurrent_count: int,
        total_requests: int = None
    ) -> ConcurrencyMetrics:
        """Test handling of concurrent requests."""
        
        if total_requests is None:
            total_requests = concurrent_count
        
        queries = TestDataGenerator.generate_research_queries()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Execute concurrent requests
            start_time = time.perf_counter()
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrent_count)
            
            async def execute_request(query_index: int) -> Tuple[bool, float, Any]:
                """Execute a single request with timing."""
                async with semaphore:
                    query = queries[query_index % len(queries)]
                    request_start = time.perf_counter()
                    
                    try:
                        result = await self.framework.run_end_to_end_test(query)
                        request_end = time.perf_counter()
                        
                        success = result["session"].status == TaskStatus.COMPLETED
                        response_time = (request_end - request_start) * 1000  # ms
                        
                        return success, response_time, result
                    except Exception as e:
                        request_end = time.perf_counter()
                        response_time = (request_end - request_start) * 1000
                        return False, response_time, e
            
            # Create and execute all requests
            tasks = [
                asyncio.create_task(execute_request(i))
                for i in range(total_requests)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            total_duration = end_time - start_time
            
            # Analyze results
            successful_requests = sum(1 for success, _, _ in results if success)
            failed_requests = total_requests - successful_requests
            response_times = [response_time for _, response_time, _ in results]
            
            # Calculate statistics
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            sorted_times = sorted(response_times)
            p50_response_time = sorted_times[int(len(sorted_times) * 0.5)]
            p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
            p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
            
            throughput_rps = total_requests / total_duration
            error_rate = (failed_requests / total_requests) * 100
            
            # Get resource metrics
            resource_metrics = self.resource_monitor.get_metrics()
            
            return ConcurrencyMetrics(
                concurrent_requests=concurrent_count,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time_ms=avg_response_time,
                min_response_time_ms=min_response_time,
                max_response_time_ms=max_response_time,
                p50_response_time_ms=p50_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                throughput_rps=throughput_rps,
                error_rate_percent=error_rate,
                memory_usage_mb=resource_metrics["current_memory_mb"],
                cpu_usage_percent=resource_metrics["average_cpu_percent"],
                active_threads=threading.active_count(),
                peak_memory_mb=resource_metrics["peak_memory_mb"],
                memory_growth_mb=resource_metrics["memory_growth_mb"]
            )
        
        finally:
            self.resource_monitor.stop_monitoring()


class TestConcurrentRequests:
    """Test cases for concurrent request handling."""
    
    @pytest.mark.asyncio
    async def test_low_concurrency_performance(self):
        """Test performance with low concurrency (1-5 concurrent requests)."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        tester = ConcurrentRequestTester(framework)
        
        async with framework.test_environment():
            # Test different concurrency levels
            concurrency_levels = [1, 2, 3, 5]
            results = []
            
            for concurrency in concurrency_levels:
                metrics = await tester.test_concurrent_requests(
                    concurrent_count=concurrency,
                    total_requests=concurrency * 2  # 2 requests per concurrent slot
                )
                results.append(metrics)
                
                # Performance assertions
                assert metrics.error_rate_percent <= 5.0
                assert metrics.throughput_rps > 0
                assert metrics.average_response_time_ms < 5000  # 5 seconds
                assert metrics.memory_growth_mb < 100  # Memory growth < 100MB
                
                print(f"\nConcurrency {concurrency} Results:")
                print(f"Throughput: {metrics.throughput_rps:.2f} RPS")
                print(f"Avg Response Time: {metrics.average_response_time_ms:.2f}ms")
                print(f"P95 Response Time: {metrics.p95_response_time_ms:.2f}ms")
                print(f"Error Rate: {metrics.error_rate_percent:.2f}%")
                print(f"Memory Growth: {metrics.memory_growth_mb:.2f}MB")
            
            # Verify scaling characteristics
            baseline_throughput = results[0].throughput_rps
            for i, metrics in enumerate(results[1:], 1):
                # Throughput should scale reasonably with concurrency
                expected_min_throughput = baseline_throughput * 0.8  # Allow some overhead
                assert metrics.throughput_rps >= expected_min_throughput
    
    @pytest.mark.asyncio
    async def test_medium_concurrency_performance(self):
        """Test performance with medium concurrency (10-20 concurrent requests)."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.02  # Allow small failure rate
        )
        
        framework = IntegrationTestFramework(config)
        tester = ConcurrentRequestTester(framework)
        
        async with framework.test_environment():
            # Test medium concurrency levels
            concurrency_levels = [10, 15, 20]
            
            for concurrency in concurrency_levels:
                metrics = await tester.test_concurrent_requests(
                    concurrent_count=concurrency,
                    total_requests=concurrency * 2
                )
                
                # More lenient assertions for higher concurrency
                assert metrics.error_rate_percent <= 15.0
                assert metrics.throughput_rps > 0
                assert metrics.average_response_time_ms < 10000  # 10 seconds
                assert metrics.p95_response_time_ms < 15000  # 15 seconds
                assert metrics.memory_growth_mb < 300  # Memory growth < 300MB
                
                print(f"\nMedium Concurrency {concurrency} Results:")
                print(f"Throughput: {metrics.throughput_rps:.2f} RPS")
                print(f"Avg Response Time: {metrics.average_response_time_ms:.2f}ms")
                print(f"P95 Response Time: {metrics.p95_response_time_ms:.2f}ms")
                print(f"P99 Response Time: {metrics.p99_response_time_ms:.2f}ms")
                print(f"Error Rate: {metrics.error_rate_percent:.2f}%")
                print(f"Memory Usage: {metrics.memory_usage_mb:.2f}MB")
                print(f"CPU Usage: {metrics.cpu_usage_percent:.2f}%")
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self):
        """Test system under high concurrency stress (30-50 concurrent requests)."""
        config = MockServiceConfig(
            web_search_delay=0.002,
            api_query_delay=0.002,
            analysis_delay=0.005,
            failure_rate=0.05  # Allow higher failure rate under stress
        )
        
        framework = IntegrationTestFramework(config)
        tester = ConcurrentRequestTester(framework)
        
        async with framework.test_environment():
            # Test high concurrency stress levels
            concurrency_levels = [30, 40, 50]
            
            for concurrency in concurrency_levels:
                metrics = await tester.test_concurrent_requests(
                    concurrent_count=concurrency,
                    total_requests=min(concurrency * 2, 80)  # Cap total requests
                )
                
                # Stress test assertions - more lenient
                assert metrics.error_rate_percent <= 30.0  # Allow higher error rate
                assert metrics.successful_requests > 0  # Some requests should succeed
                assert metrics.throughput_rps > 0
                assert metrics.memory_growth_mb < 500  # Memory growth < 500MB
                
                # Calculate success rate
                success_rate = (metrics.successful_requests / metrics.total_requests) * 100
                
                print(f"\nHigh Concurrency Stress {concurrency} Results:")
                print(f"Success Rate: {success_rate:.2f}%")
                print(f"Throughput: {metrics.throughput_rps:.2f} RPS")
                print(f"Avg Response Time: {metrics.average_response_time_ms:.2f}ms")
                print(f"P95 Response Time: {metrics.p95_response_time_ms:.2f}ms")
                print(f"Error Rate: {metrics.error_rate_percent:.2f}%")
                print(f"Peak Memory: {metrics.peak_memory_mb:.2f}MB")
                print(f"Memory Growth: {metrics.memory_growth_mb:.2f}MB")
                print(f"Active Threads: {metrics.active_threads}")
                
                # System should maintain some level of functionality
                assert success_rate >= 50.0  # At least 50% success rate
    
    @pytest.mark.asyncio
    async def test_concurrent_request_isolation(self):
        """Test that concurrent requests don't interfere with each other."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.1  # Some requests will fail
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Create requests with different queries to test isolation
            queries = TestDataGenerator.generate_research_queries()[:10]
            
            # Execute all requests concurrently
            tasks = []
            for i, query in enumerate(queries):
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                tasks.append((i, query, task))
            
            # Wait for all to complete
            results = []
            for i, query, task in tasks:
                try:
                    result = await task
                    results.append((i, query, result, None))
                except Exception as e:
                    results.append((i, query, None, e))
            
            # Analyze isolation
            successful_results = [r for r in results if r[2] is not None and r[2]["session"].status == TaskStatus.COMPLETED]
            failed_results = [r for r in results if r[2] is None or r[2]["session"].status != TaskStatus.COMPLETED]
            
            # Verify that successful requests have correct data
            for i, query, result, error in successful_results:
                session = result["session"]
                assert session.original_query == query
                assert session.session_id is not None
                assert len(session.plan.tasks) > 0
                
                # Verify session isolation - each should have unique session ID
                session_ids = [r[2]["session"].session_id for r in successful_results]
                assert len(set(session_ids)) == len(successful_results)  # All unique
            
            # Some requests should succeed even if others fail
            assert len(successful_results) > 0
            
            print(f"\nConcurrent Request Isolation Results:")
            print(f"Total Requests: {len(results)}")
            print(f"Successful Requests: {len(successful_results)}")
            print(f"Failed Requests: {len(failed_results)}")
            print(f"Success Rate: {(len(successful_results)/len(results))*100:.2f}%")
    
    @pytest.mark.asyncio
    async def test_concurrent_request_resource_limits(self):
        """Test system behavior when approaching resource limits."""
        config = MockServiceConfig(
            web_search_delay=0.001,  # Very fast to stress system
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        tester = ConcurrentRequestTester(framework)
        
        async with framework.test_environment():
            # Gradually increase load until we hit limits
            max_concurrency = 100
            step_size = 10
            
            for concurrency in range(step_size, max_concurrency + 1, step_size):
                print(f"\nTesting concurrency level: {concurrency}")
                
                try:
                    metrics = await tester.test_concurrent_requests(
                        concurrent_count=concurrency,
                        total_requests=min(concurrency, 50)  # Cap total requests
                    )
                    
                    print(f"Throughput: {metrics.throughput_rps:.2f} RPS")
                    print(f"Error Rate: {metrics.error_rate_percent:.2f}%")
                    print(f"Memory Usage: {metrics.memory_usage_mb:.2f}MB")
                    print(f"Memory Growth: {metrics.memory_growth_mb:.2f}MB")
                    
                    # Check if we're hitting resource limits
                    if (metrics.error_rate_percent > 50.0 or 
                        metrics.memory_growth_mb > 1000 or 
                        metrics.average_response_time_ms > 30000):
                        print(f"Resource limits reached at concurrency {concurrency}")
                        break
                    
                    # Basic functionality should be maintained
                    assert metrics.successful_requests > 0
                    assert metrics.throughput_rps > 0
                    
                except Exception as e:
                    print(f"System failure at concurrency {concurrency}: {e}")
                    break
    
    @pytest.mark.asyncio
    async def test_concurrent_request_fairness(self):
        """Test fairness of request handling under concurrent load."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Create requests with timing tracking
            num_requests = 20
            queries = TestDataGenerator.generate_research_queries()
            
            request_times = []
            
            async def timed_request(request_id: int, query: str):
                """Execute request with timing."""
                start_time = time.perf_counter()
                try:
                    result = await framework.run_end_to_end_test(query)
                    end_time = time.perf_counter()
                    
                    return {
                        "request_id": request_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "success": result["session"].status == TaskStatus.COMPLETED,
                        "result": result
                    }
                except Exception as e:
                    end_time = time.perf_counter()
                    return {
                        "request_id": request_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "success": False,
                        "error": str(e)
                    }
            
            # Execute all requests concurrently
            tasks = [
                asyncio.create_task(timed_request(i, queries[i % len(queries)]))
                for i in range(num_requests)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Analyze fairness
            successful_results = [r for r in results if r["success"]]
            durations = [r["duration"] for r in successful_results]
            
            if durations:
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
                
                # Calculate fairness metric (coefficient of variation)
                import statistics
                std_dev = statistics.stdev(durations)
                fairness_coefficient = std_dev / avg_duration
                
                print(f"\nConcurrent Request Fairness Results:")
                print(f"Successful Requests: {len(successful_results)}/{num_requests}")
                print(f"Average Duration: {avg_duration:.3f}s")
                print(f"Min Duration: {min_duration:.3f}s")
                print(f"Max Duration: {max_duration:.3f}s")
                print(f"Fairness Coefficient: {fairness_coefficient:.3f}")
                
                # Fairness assertions
                assert len(successful_results) > 0
                assert fairness_coefficient < 1.0  # Reasonable fairness
                assert max_duration / min_duration < 5.0  # No request should take 5x longer than fastest


    @pytest.mark.asyncio
    async def test_concurrent_request_throughput_benchmarks(self):
        """Benchmark throughput under various concurrent request scenarios."""
        config = MockServiceConfig(
            web_search_delay=0.001,
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        tester = ConcurrentRequestTester(framework)
        
        async with framework.test_environment():
            # Benchmark different throughput scenarios
            benchmark_scenarios = [
                {"name": "Low Load", "concurrent": 2, "total": 10, "target_rps": 5.0},
                {"name": "Medium Load", "concurrent": 5, "total": 25, "target_rps": 10.0},
                {"name": "High Load", "concurrent": 10, "total": 50, "target_rps": 15.0},
                {"name": "Burst Load", "concurrent": 20, "total": 40, "target_rps": 20.0}
            ]
            
            benchmark_results = []
            
            for scenario in benchmark_scenarios:
                print(f"\nBenchmarking {scenario['name']}...")
                
                metrics = await tester.test_concurrent_requests(
                    concurrent_count=scenario["concurrent"],
                    total_requests=scenario["total"]
                )
                
                benchmark_results.append({
                    "scenario": scenario["name"],
                    "metrics": metrics,
                    "target_rps": scenario["target_rps"],
                    "achieved_rps": metrics.throughput_rps,
                    "performance_ratio": metrics.throughput_rps / scenario["target_rps"]
                })
                
                print(f"{scenario['name']} Results:")
                print(f"  Target RPS: {scenario['target_rps']:.1f}")
                print(f"  Achieved RPS: {metrics.throughput_rps:.2f}")
                print(f"  Performance Ratio: {metrics.throughput_rps / scenario['target_rps']:.2f}")
                print(f"  Error Rate: {metrics.error_rate_percent:.2f}%")
                
                # Performance assertions
                assert metrics.error_rate_percent <= 20.0
                assert metrics.throughput_rps > 0
                
                # For low and medium load, should meet target performance
                if scenario["name"] in ["Low Load", "Medium Load"]:
                    assert metrics.throughput_rps >= scenario["target_rps"] * 0.8
            
            # Analyze scaling characteristics across scenarios
            throughputs = [r["achieved_rps"] for r in benchmark_results]
            concurrencies = [s["concurrent"] for s in benchmark_scenarios]
            
            # Throughput should generally increase with concurrency (allowing for overhead)
            for i in range(1, len(throughputs)):
                if concurrencies[i] > concurrencies[i-1]:
                    # Allow some degradation at high concurrency
                    min_expected = throughputs[i-1] * 0.7
                    assert throughputs[i] >= min_expected or throughputs[i] >= 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])