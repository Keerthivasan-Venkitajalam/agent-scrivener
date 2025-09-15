"""
Load testing for Agent Scrivener system.

Tests system behavior under various load conditions including
concurrent requests, sustained load, and stress testing.
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from tests.integration.test_framework import (
    IntegrationTestFramework, MockServiceConfig, TestDataGenerator
)
from tests.data.test_datasets import TestDatasets
from agent_scrivener.models.core import TaskStatus, SessionState


@dataclass
class LoadTestResult:
    """Results from a load test execution."""
    test_name: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate_percent: float
    concurrent_sessions: int
    memory_usage_mb: float
    cpu_usage_percent: float


class LoadTestRunner:
    """Runner for executing load tests."""
    
    def __init__(self, framework: IntegrationTestFramework):
        self.framework = framework
        self.results = []
    
    async def run_concurrent_load_test(
        self, 
        concurrent_sessions: int, 
        requests_per_session: int = 1,
        test_duration_seconds: int = 60
    ) -> LoadTestResult:
        """Run concurrent load test with specified parameters."""
        
        queries = TestDataGenerator.generate_research_queries()
        start_time = time.perf_counter()
        
        # Track all tasks and results
        all_tasks = []
        results = []
        
        # Generate load for specified duration
        end_time = start_time + test_duration_seconds
        
        while time.perf_counter() < end_time:
            # Create batch of concurrent sessions
            batch_tasks = []
            for i in range(concurrent_sessions):
                query = queries[i % len(queries)]
                task = asyncio.create_task(self.framework.run_end_to_end_test(query))
                batch_tasks.append(task)
                all_tasks.append(task)
            
            # Wait for batch to complete or timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=30.0
                )
                results.extend(batch_results)
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for task in batch_tasks:
                    if not task.done():
                        task.cancel()
                break
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        # Calculate metrics
        actual_duration = time.perf_counter() - start_time
        
        successful_results = [
            r for r in results 
            if not isinstance(r, Exception) and r["session"].status == TaskStatus.COMPLETED
        ]
        failed_results = len(results) - len(successful_results)
        
        # Calculate latency statistics
        latencies = [r["metrics"].execution_time_ms for r in successful_results]
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            sorted_latencies = sorted(latencies)
            p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        else:
            avg_latency = p95_latency = p99_latency = 0
        
        return LoadTestResult(
            test_name=f"concurrent_load_{concurrent_sessions}",
            duration_seconds=actual_duration,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=failed_results,
            requests_per_second=len(results) / actual_duration if actual_duration > 0 else 0,
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            error_rate_percent=(failed_results / len(results)) * 100 if results else 0,
            concurrent_sessions=concurrent_sessions,
            memory_usage_mb=0,  # Would be measured by framework
            cpu_usage_percent=0  # Would be measured by framework
        )
    
    async def run_sustained_load_test(
        self,
        target_rps: float,
        duration_seconds: int = 120
    ) -> LoadTestResult:
        """Run sustained load test at target requests per second."""
        
        queries = TestDataGenerator.generate_research_queries()
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        request_interval = 1.0 / target_rps
        active_tasks = []
        completed_results = []
        query_index = 0
        
        last_request_time = start_time
        
        while time.perf_counter() < end_time:
            current_time = time.perf_counter()
            
            # Send new request if interval has passed
            if current_time - last_request_time >= request_interval:
                query = queries[query_index % len(queries)]
                query_index += 1
                
                task = asyncio.create_task(self.framework.run_end_to_end_test(query))
                active_tasks.append(task)
                last_request_time = current_time
            
            # Check for completed tasks
            done_tasks = [task for task in active_tasks if task.done()]
            for task in done_tasks:
                active_tasks.remove(task)
                try:
                    result = await task
                    completed_results.append(result)
                except Exception as e:
                    completed_results.append(e)
            
            # Brief pause to prevent busy waiting
            await asyncio.sleep(0.01)
        
        # Wait for remaining tasks to complete
        if active_tasks:
            remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)
            completed_results.extend(remaining_results)
        
        # Calculate metrics
        actual_duration = time.perf_counter() - start_time
        
        successful_results = [
            r for r in completed_results 
            if not isinstance(r, Exception) and r["session"].status == TaskStatus.COMPLETED
        ]
        failed_results = len(completed_results) - len(successful_results)
        
        # Calculate latency statistics
        latencies = [r["metrics"].execution_time_ms for r in successful_results]
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            sorted_latencies = sorted(latencies)
            p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        else:
            avg_latency = p95_latency = p99_latency = 0
        
        return LoadTestResult(
            test_name=f"sustained_load_{target_rps}rps",
            duration_seconds=actual_duration,
            total_requests=len(completed_results),
            successful_requests=len(successful_results),
            failed_requests=failed_results,
            requests_per_second=len(completed_results) / actual_duration if actual_duration > 0 else 0,
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            error_rate_percent=(failed_results / len(completed_results)) * 100 if completed_results else 0,
            concurrent_sessions=len(active_tasks),
            memory_usage_mb=0,
            cpu_usage_percent=0
        )


class TestLoadTesting:
    """Load testing test cases."""
    
    @pytest.mark.asyncio
    async def test_low_concurrency_load(self):
        """Test system performance with low concurrency (1-3 concurrent sessions)."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        load_runner = LoadTestRunner(framework)
        
        async with framework.test_environment():
            # Test with 1, 2, and 3 concurrent sessions
            for concurrency in [1, 2, 3]:
                result = await load_runner.run_concurrent_load_test(
                    concurrent_sessions=concurrency,
                    test_duration_seconds=15
                )
                
                # Performance assertions
                assert result.error_rate_percent <= 5.0
                assert result.requests_per_second > 0
                assert result.average_latency_ms < 5000  # 5 seconds
                assert result.successful_requests > 0
                
                print(f"\nConcurrency {concurrency} Results:")
                print(f"RPS: {result.requests_per_second:.2f}")
                print(f"Avg Latency: {result.average_latency_ms:.2f}ms")
                print(f"Error Rate: {result.error_rate_percent:.2f}%")
                print(f"Total Requests: {result.total_requests}")
    
    @pytest.mark.asyncio
    async def test_medium_concurrency_load(self):
        """Test system performance with medium concurrency (5-10 concurrent sessions)."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        load_runner = LoadTestRunner(framework)
        
        async with framework.test_environment():
            # Test with 5 and 10 concurrent sessions
            for concurrency in [5, 10]:
                result = await load_runner.run_concurrent_load_test(
                    concurrent_sessions=concurrency,
                    test_duration_seconds=20
                )
                
                # Performance assertions
                assert result.error_rate_percent <= 10.0
                assert result.requests_per_second > 0
                assert result.average_latency_ms < 8000  # 8 seconds
                assert result.p95_latency_ms < 12000  # 12 seconds
                
                print(f"\nConcurrency {concurrency} Results:")
                print(f"RPS: {result.requests_per_second:.2f}")
                print(f"Avg Latency: {result.average_latency_ms:.2f}ms")
                print(f"P95 Latency: {result.p95_latency_ms:.2f}ms")
                print(f"Error Rate: {result.error_rate_percent:.2f}%")
    
    @pytest.mark.asyncio
    async def test_high_concurrency_load(self):
        """Test system performance with high concurrency (15-25 concurrent sessions)."""
        config = MockServiceConfig(
            web_search_delay=0.002,
            api_query_delay=0.002,
            analysis_delay=0.005,
            failure_rate=0.05  # Allow some failures under high load
        )
        
        framework = IntegrationTestFramework(config)
        load_runner = LoadTestRunner(framework)
        
        async with framework.test_environment():
            # Test with 15 and 25 concurrent sessions
            for concurrency in [15, 25]:
                result = await load_runner.run_concurrent_load_test(
                    concurrent_sessions=concurrency,
                    test_duration_seconds=25
                )
                
                # More lenient assertions for high concurrency
                assert result.error_rate_percent <= 25.0  # Allow higher error rate
                assert result.requests_per_second > 0
                assert result.average_latency_ms < 15000  # 15 seconds
                assert result.successful_requests > 0
                
                print(f"\nHigh Concurrency {concurrency} Results:")
                print(f"RPS: {result.requests_per_second:.2f}")
                print(f"Avg Latency: {result.average_latency_ms:.2f}ms")
                print(f"P95 Latency: {result.p95_latency_ms:.2f}ms")
                print(f"P99 Latency: {result.p99_latency_ms:.2f}ms")
                print(f"Error Rate: {result.error_rate_percent:.2f}%")
                print(f"Success Rate: {(result.successful_requests/result.total_requests)*100:.2f}%")
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self):
        """Test sustained load at various request rates."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.02,
            rate_limit_enabled=True,
            rate_limit_requests_per_second=20
        )
        
        framework = IntegrationTestFramework(config)
        load_runner = LoadTestRunner(framework)
        
        async with framework.test_environment():
            # Test different sustained load rates
            target_rates = [0.5, 1.0, 2.0]  # requests per second
            
            for target_rps in target_rates:
                result = await load_runner.run_sustained_load_test(
                    target_rps=target_rps,
                    duration_seconds=30
                )
                
                # Verify sustained load performance
                actual_rps = result.requests_per_second
                rps_variance = abs(actual_rps - target_rps) / target_rps
                
                assert result.error_rate_percent <= 15.0
                assert rps_variance <= 0.5  # Within 50% of target rate
                assert result.successful_requests > 0
                
                print(f"\nSustained Load {target_rps} RPS Results:")
                print(f"Target RPS: {target_rps:.2f}")
                print(f"Actual RPS: {actual_rps:.2f}")
                print(f"RPS Variance: {rps_variance:.2%}")
                print(f"Avg Latency: {result.average_latency_ms:.2f}ms")
                print(f"Error Rate: {result.error_rate_percent:.2f}%")
    
    @pytest.mark.asyncio
    async def test_stress_testing_with_failures(self):
        """Test system behavior under stress with induced failures."""
        config = MockServiceConfig(
            web_search_delay=0.001,
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.3,  # High failure rate for stress testing
            rate_limit_enabled=True,
            rate_limit_requests_per_second=10
        )
        
        framework = IntegrationTestFramework(config)
        load_runner = LoadTestRunner(framework)
        
        async with framework.test_environment():
            # Stress test with high concurrency and failures
            result = await load_runner.run_concurrent_load_test(
                concurrent_sessions=20,
                test_duration_seconds=30
            )
            
            # Verify system resilience under stress
            assert result.total_requests > 0
            assert result.successful_requests > 0  # Some requests should succeed
            assert result.error_rate_percent <= 70.0  # Not all requests should fail
            assert result.requests_per_second > 0
            
            # System should maintain some level of functionality
            success_rate = (result.successful_requests / result.total_requests) * 100
            assert success_rate >= 20.0  # At least 20% success rate
            
            print(f"\nStress Test Results:")
            print(f"Total Requests: {result.total_requests}")
            print(f"Successful Requests: {result.successful_requests}")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"RPS: {result.requests_per_second:.2f}")
            print(f"Avg Latency: {result.average_latency_ms:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_gradual_load_increase(self):
        """Test system behavior with gradually increasing load."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.05
        )
        
        framework = IntegrationTestFramework(config)
        load_runner = LoadTestRunner(framework)
        
        async with framework.test_environment():
            # Gradually increase load and measure performance
            load_levels = [1, 3, 5, 8, 12, 15]
            results = []
            
            for load_level in load_levels:
                result = await load_runner.run_concurrent_load_test(
                    concurrent_sessions=load_level,
                    test_duration_seconds=15
                )
                results.append(result)
                
                print(f"\nLoad Level {load_level}:")
                print(f"RPS: {result.requests_per_second:.2f}")
                print(f"Avg Latency: {result.average_latency_ms:.2f}ms")
                print(f"Error Rate: {result.error_rate_percent:.2f}%")
                
                # Brief pause between load levels
                await asyncio.sleep(2)
            
            # Analyze performance degradation
            baseline_rps = results[0].requests_per_second
            baseline_latency = results[0].average_latency_ms
            
            for i, result in enumerate(results[1:], 1):
                # Performance should not degrade too severely
                rps_degradation = (baseline_rps - result.requests_per_second) / baseline_rps
                latency_increase = (result.average_latency_ms - baseline_latency) / baseline_latency
                
                # Allow some degradation but not complete failure
                assert rps_degradation <= 0.8  # RPS should not drop by more than 80%
                assert latency_increase <= 5.0  # Latency should not increase by more than 500%
                assert result.error_rate_percent <= 30.0  # Error rate should stay reasonable
    
    @pytest.mark.asyncio
    async def test_burst_load_handling(self):
        """Test system handling of sudden load bursts."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.1
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Simulate burst load scenario
            queries = TestDataGenerator.generate_research_queries()
            
            # Normal load phase
            normal_tasks = []
            for i in range(3):
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                normal_tasks.append(task)
            
            # Wait a bit for normal load to establish
            await asyncio.sleep(1)
            
            # Sudden burst of requests
            burst_tasks = []
            for i in range(15):  # Sudden burst of 15 requests
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                burst_tasks.append(task)
            
            # Wait for all tasks to complete
            all_tasks = normal_tasks + burst_tasks
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Analyze burst handling
            successful_results = [
                r for r in results 
                if not isinstance(r, Exception) and r["session"].status == TaskStatus.COMPLETED
            ]
            failed_results = len(results) - len(successful_results)
            
            success_rate = (len(successful_results) / len(results)) * 100
            error_rate = (failed_results / len(results)) * 100
            
            # System should handle burst reasonably well
            assert success_rate >= 50.0  # At least 50% success rate
            assert error_rate <= 50.0  # Error rate should not exceed 50%
            assert len(successful_results) > 0  # Some requests should succeed
            
            print(f"\nBurst Load Test Results:")
            print(f"Total Requests: {len(results)}")
            print(f"Successful Requests: {len(successful_results)}")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Error Rate: {error_rate:.2f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])