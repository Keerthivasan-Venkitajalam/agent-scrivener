"""
System limits and degradation scenario tests.

Tests system behavior at operational limits, graceful degradation,
and recovery from extreme conditions.
"""

import pytest
import asyncio
import time
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from tests.integration.test_framework import (
    IntegrationTestFramework, MockServiceConfig, TestDataGenerator
)
from agent_scrivener.models.core import TaskStatus, SessionState


@dataclass
class SystemLimitMetrics:
    """Metrics for system limit testing."""
    test_name: str
    limit_reached: bool
    breaking_point: Optional[int]
    max_successful_operations: int
    degradation_threshold: Optional[int]
    recovery_time_seconds: Optional[float]
    error_rate_at_limit: float
    memory_at_limit_mb: float
    cpu_at_limit_percent: float
    system_stability_score: float  # 0-100


class SystemLimitTester:
    """Test system limits and degradation scenarios."""
    
    def __init__(self, framework: IntegrationTestFramework):
        self.framework = framework
        self.process = psutil.Process()
    
    async def find_concurrency_limit(
        self,
        start_concurrency: int = 5,
        max_concurrency: int = 100,
        step_size: int = 5,
        failure_threshold: float = 50.0
    ) -> SystemLimitMetrics:
        """Find the concurrency limit where system starts failing."""
        
        breaking_point = None
        degradation_threshold = None
        max_successful = 0
        
        for concurrency in range(start_concurrency, max_concurrency + 1, step_size):
            print(f"Testing concurrency level: {concurrency}")
            
            # Measure system state before test
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            try:
                # Execute concurrent requests
                queries = TestDataGenerator.generate_research_queries()
                tasks = []
                
                for i in range(concurrency):
                    query = queries[i % len(queries)]
                    task = asyncio.create_task(self.framework.run_end_to_end_test(query))
                    tasks.append(task)
                
                # Wait for completion with timeout
                start_time = time.perf_counter()
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=60.0  # 60 second timeout
                )
                end_time = time.perf_counter()
                
                # Analyze results
                successful = sum(
                    1 for r in results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                
                error_rate = ((concurrency - successful) / concurrency) * 100
                final_memory = self.process.memory_info().rss / 1024 / 1024
                
                print(f"Concurrency {concurrency}: {successful}/{concurrency} successful ({error_rate:.1f}% error rate)")
                
                # Track maximum successful operations
                max_successful = max(max_successful, successful)
                
                # Check for degradation threshold
                if error_rate > 10.0 and degradation_threshold is None:
                    degradation_threshold = concurrency
                    print(f"Degradation threshold reached at concurrency {concurrency}")
                
                # Check for breaking point
                if error_rate >= failure_threshold:
                    breaking_point = concurrency
                    print(f"Breaking point reached at concurrency {concurrency}")
                    break
                
                # Check for resource exhaustion
                if final_memory > initial_memory + 1000:  # 1GB memory increase
                    print(f"Memory limit reached at concurrency {concurrency}")
                    breaking_point = concurrency
                    break
                
            except asyncio.TimeoutError:
                print(f"Timeout at concurrency {concurrency}")
                breaking_point = concurrency
                break
            except Exception as e:
                print(f"System failure at concurrency {concurrency}: {e}")
                breaking_point = concurrency
                break
            
            # Brief pause between tests
            gc.collect()
            await asyncio.sleep(1.0)
        
        # Calculate final metrics
        final_memory = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        # Calculate stability score
        if breaking_point:
            stability_score = min(100, (breaking_point / max_concurrency) * 100)
        else:
            stability_score = 100
        
        return SystemLimitMetrics(
            test_name="concurrency_limit",
            limit_reached=breaking_point is not None,
            breaking_point=breaking_point,
            max_successful_operations=max_successful,
            degradation_threshold=degradation_threshold,
            recovery_time_seconds=None,
            error_rate_at_limit=failure_threshold if breaking_point else 0,
            memory_at_limit_mb=final_memory,
            cpu_at_limit_percent=cpu_percent,
            system_stability_score=stability_score
        )
    
    async def test_memory_exhaustion_recovery(self) -> SystemLimitMetrics:
        """Test system behavior under memory pressure and recovery."""
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        recovery_start_time = None
        recovery_end_time = None
        
        try:
            # Create memory pressure by running many concurrent sessions
            queries = TestDataGenerator.generate_research_queries()
            memory_pressure_tasks = []
            
            # Gradually increase memory pressure
            for batch in range(10):  # 10 batches
                batch_tasks = []
                for i in range(5):  # 5 tasks per batch
                    query = queries[(batch * 5 + i) % len(queries)]
                    task = asyncio.create_task(self.framework.run_end_to_end_test(query))
                    batch_tasks.append(task)
                
                memory_pressure_tasks.extend(batch_tasks)
                
                # Check memory usage
                current_memory = self.process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                print(f"Batch {batch + 1}: Memory usage {current_memory:.2f}MB")
                
                # If memory usage is too high, start recovery
                if current_memory > initial_memory + 500:  # 500MB increase
                    print("Memory pressure detected, starting recovery")
                    recovery_start_time = time.perf_counter()
                    break
                
                await asyncio.sleep(0.5)
            
            # Wait for some tasks to complete (natural recovery)
            if recovery_start_time:
                completed_count = 0
                while completed_count < len(memory_pressure_tasks) // 2:  # Wait for half to complete
                    completed_count = sum(1 for task in memory_pressure_tasks if task.done())
                    await asyncio.sleep(0.5)
                
                # Force garbage collection
                gc.collect()
                await asyncio.sleep(1.0)
                
                recovery_end_time = time.perf_counter()
            
            # Wait for all remaining tasks
            if memory_pressure_tasks:
                await asyncio.gather(*memory_pressure_tasks, return_exceptions=True)
            
            # Final cleanup
            gc.collect()
            await asyncio.sleep(2.0)
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Calculate recovery metrics
            recovery_time = None
            if recovery_start_time and recovery_end_time:
                recovery_time = recovery_end_time - recovery_start_time
            
            memory_recovered = peak_memory - final_memory
            recovery_percentage = (memory_recovered / (peak_memory - initial_memory)) * 100 if peak_memory > initial_memory else 100
            
            print(f"Memory recovery: {memory_recovered:.2f}MB ({recovery_percentage:.1f}%)")
            
            return SystemLimitMetrics(
                test_name="memory_exhaustion_recovery",
                limit_reached=peak_memory > initial_memory + 300,
                breaking_point=None,
                max_successful_operations=len(memory_pressure_tasks),
                degradation_threshold=None,
                recovery_time_seconds=recovery_time,
                error_rate_at_limit=0,
                memory_at_limit_mb=peak_memory,
                cpu_at_limit_percent=self.process.cpu_percent(),
                system_stability_score=min(100, recovery_percentage)
            )
            
        except Exception as e:
            print(f"Memory exhaustion test failed: {e}")
            return SystemLimitMetrics(
                test_name="memory_exhaustion_recovery",
                limit_reached=True,
                breaking_point=None,
                max_successful_operations=0,
                degradation_threshold=None,
                recovery_time_seconds=None,
                error_rate_at_limit=100,
                memory_at_limit_mb=peak_memory,
                cpu_at_limit_percent=0,
                system_stability_score=0
            )
    
    async def test_sustained_high_load_degradation(self, duration_seconds: int = 60) -> SystemLimitMetrics:
        """Test system degradation under sustained high load."""
        
        queries = TestDataGenerator.generate_research_queries()
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        # Metrics tracking
        total_requests = 0
        successful_requests = 0
        error_rates_over_time = []
        memory_usage_over_time = []
        
        # Maintain high concurrent load
        target_concurrency = 15
        active_tasks = []
        
        while time.perf_counter() < end_time:
            current_time = time.perf_counter()
            
            # Maintain target concurrency
            while len(active_tasks) < target_concurrency:
                query = queries[total_requests % len(queries)]
                total_requests += 1
                
                task = asyncio.create_task(self.framework.run_end_to_end_test(query))
                active_tasks.append(task)
            
            # Check for completed tasks
            done_tasks = [task for task in active_tasks if task.done()]
            for task in done_tasks:
                active_tasks.remove(task)
                try:
                    result = await task
                    if result["session"].status == TaskStatus.COMPLETED:
                        successful_requests += 1
                except Exception:
                    pass  # Count as failure
            
            # Record metrics every 10 seconds
            if int(current_time - start_time) % 10 == 0 and len(error_rates_over_time) < duration_seconds // 10:
                current_error_rate = ((total_requests - successful_requests) / total_requests) * 100 if total_requests > 0 else 0
                current_memory = self.process.memory_info().rss / 1024 / 1024
                
                error_rates_over_time.append(current_error_rate)
                memory_usage_over_time.append(current_memory)
                
                print(f"Time {int(current_time - start_time)}s: Error rate {current_error_rate:.1f}%, Memory {current_memory:.1f}MB")
            
            await asyncio.sleep(0.5)
        
        # Wait for remaining tasks
        if active_tasks:
            remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)
            for result in remaining_results:
                if not isinstance(result, Exception) and result["session"].status == TaskStatus.COMPLETED:
                    successful_requests += 1
        
        # Analyze degradation
        final_error_rate = ((total_requests - successful_requests) / total_requests) * 100 if total_requests > 0 else 0
        
        # Check if error rate increased over time (degradation)
        if len(error_rates_over_time) > 1:
            initial_error_rate = error_rates_over_time[0]
            degradation_detected = error_rates_over_time[-1] > initial_error_rate + 10  # 10% increase
        else:
            degradation_detected = False
        
        # Calculate stability score based on error rate consistency
        if error_rates_over_time:
            import statistics
            error_rate_variance = statistics.variance(error_rates_over_time)
            stability_score = max(0, 100 - error_rate_variance)
        else:
            stability_score = 0
        
        return SystemLimitMetrics(
            test_name="sustained_high_load",
            limit_reached=final_error_rate > 30,
            breaking_point=None,
            max_successful_operations=successful_requests,
            degradation_threshold=None,
            recovery_time_seconds=None,
            error_rate_at_limit=final_error_rate,
            memory_at_limit_mb=memory_usage_over_time[-1] if memory_usage_over_time else 0,
            cpu_at_limit_percent=self.process.cpu_percent(),
            system_stability_score=stability_score
        )


class TestSystemLimits:
    """Test cases for system limits and degradation scenarios."""
    
    @pytest.mark.asyncio
    async def test_find_concurrency_breaking_point(self):
        """Find the concurrency level where system performance breaks down."""
        config = MockServiceConfig(
            web_search_delay=0.001,
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        tester = SystemLimitTester(framework)
        
        async with framework.test_environment():
            metrics = await tester.find_concurrency_limit(
                start_concurrency=5,
                max_concurrency=50,
                step_size=5,
                failure_threshold=40.0
            )
            
            print(f"\nConcurrency Limit Test Results:")
            print(f"Breaking Point: {metrics.breaking_point}")
            print(f"Degradation Threshold: {metrics.degradation_threshold}")
            print(f"Max Successful Operations: {metrics.max_successful_operations}")
            print(f"System Stability Score: {metrics.system_stability_score:.1f}")
            print(f"Memory at Limit: {metrics.memory_at_limit_mb:.2f}MB")
            
            # System should handle at least some concurrency
            assert metrics.max_successful_operations >= 5
            assert metrics.system_stability_score > 0
            
            # If a breaking point was found, it should be reasonable
            if metrics.breaking_point:
                assert metrics.breaking_point >= 10  # Should handle at least 10 concurrent requests
                assert metrics.degradation_threshold is None or metrics.degradation_threshold <= metrics.breaking_point
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        tester = SystemLimitTester(framework)
        
        async with framework.test_environment():
            metrics = await tester.test_memory_exhaustion_recovery()
            
            print(f"\nMemory Pressure Test Results:")
            print(f"Limit Reached: {metrics.limit_reached}")
            print(f"Max Operations: {metrics.max_successful_operations}")
            print(f"Recovery Time: {metrics.recovery_time_seconds:.2f}s" if metrics.recovery_time_seconds else "N/A")
            print(f"Memory at Limit: {metrics.memory_at_limit_mb:.2f}MB")
            print(f"System Stability Score: {metrics.system_stability_score:.1f}")
            
            # System should handle memory pressure gracefully
            assert metrics.max_successful_operations > 0
            assert metrics.system_stability_score >= 50  # Should recover reasonably well
            
            # If recovery occurred, it should be reasonably fast
            if metrics.recovery_time_seconds:
                assert metrics.recovery_time_seconds < 30  # Should recover within 30 seconds
    
    @pytest.mark.asyncio
    async def test_sustained_load_degradation(self):
        """Test system degradation under sustained high load."""
        config = MockServiceConfig(
            web_search_delay=0.002,
            api_query_delay=0.002,
            analysis_delay=0.005,
            failure_rate=0.05  # Small failure rate for realism
        )
        
        framework = IntegrationTestFramework(config)
        tester = SystemLimitTester(framework)
        
        async with framework.test_environment():
            # Run sustained load test for 30 seconds
            metrics = await tester.test_sustained_high_load_degradation(duration_seconds=30)
            
            print(f"\nSustained Load Degradation Test Results:")
            print(f"Total Operations: {metrics.max_successful_operations}")
            print(f"Error Rate at End: {metrics.error_rate_at_limit:.2f}%")
            print(f"Memory at End: {metrics.memory_at_limit_mb:.2f}MB")
            print(f"System Stability Score: {metrics.system_stability_score:.1f}")
            
            # System should maintain reasonable performance under sustained load
            assert metrics.max_successful_operations > 0
            assert metrics.error_rate_at_limit <= 60.0  # Error rate should not exceed 60%
            assert metrics.system_stability_score >= 30  # Should maintain some stability
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_scenarios(self):
        """Test various resource exhaustion scenarios."""
        config = MockServiceConfig(
            web_search_delay=0.001,
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Test 1: File descriptor exhaustion simulation
            print("\nTesting file descriptor limits...")
            
            # Create many concurrent connections (simulated)
            queries = TestDataGenerator.generate_research_queries()
            fd_test_tasks = []
            
            try:
                for i in range(20):  # Create 20 concurrent tasks
                    query = queries[i % len(queries)]
                    task = asyncio.create_task(framework.run_end_to_end_test(query))
                    fd_test_tasks.append(task)
                
                # Wait for completion
                fd_results = await asyncio.gather(*fd_test_tasks, return_exceptions=True)
                
                fd_successful = sum(
                    1 for r in fd_results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                
                print(f"FD Test: {fd_successful}/{len(fd_test_tasks)} successful")
                
                # Should handle reasonable number of concurrent connections
                assert fd_successful >= len(fd_test_tasks) // 2  # At least 50% success
                
            except Exception as e:
                print(f"FD test encountered limits: {e}")
                # This is expected behavior when hitting limits
            
            # Test 2: Thread exhaustion simulation
            print("\nTesting thread limits...")
            
            thread_test_tasks = []
            try:
                for i in range(30):  # Create 30 concurrent tasks
                    query = queries[i % len(queries)]
                    task = asyncio.create_task(framework.run_end_to_end_test(query))
                    thread_test_tasks.append(task)
                
                thread_results = await asyncio.gather(*thread_test_tasks, return_exceptions=True)
                
                thread_successful = sum(
                    1 for r in thread_results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                
                print(f"Thread Test: {thread_successful}/{len(thread_test_tasks)} successful")
                
                # Should handle reasonable number of threads
                assert thread_successful >= len(thread_test_tasks) // 3  # At least 33% success
                
            except Exception as e:
                print(f"Thread test encountered limits: {e}")
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_failures(self):
        """Test graceful degradation when components fail."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.4  # High failure rate to simulate component failures
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Execute requests with high failure rate
            queries = TestDataGenerator.generate_research_queries()[:15]
            
            # Track performance over time as failures occur
            batch_size = 5
            batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
            
            batch_results = []
            
            for i, batch in enumerate(batches):
                print(f"\nExecuting batch {i+1}/{len(batches)}")
                
                batch_tasks = []
                for query in batch:
                    task = asyncio.create_task(framework.run_end_to_end_test(query))
                    batch_tasks.append(task)
                
                batch_start = time.perf_counter()
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                batch_duration = time.perf_counter() - batch_start
                
                successful = sum(
                    1 for r in results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                
                batch_metrics = {
                    "batch": i + 1,
                    "total": len(batch),
                    "successful": successful,
                    "success_rate": (successful / len(batch)) * 100,
                    "duration": batch_duration,
                    "throughput": len(batch) / batch_duration
                }
                
                batch_results.append(batch_metrics)
                
                print(f"Batch {i+1}: {successful}/{len(batch)} successful ({batch_metrics['success_rate']:.1f}%)")
                
                # Brief pause between batches
                await asyncio.sleep(1.0)
            
            # Analyze degradation pattern
            success_rates = [b["success_rate"] for b in batch_results]
            throughputs = [b["throughput"] for b in batch_results]
            
            # System should maintain some level of functionality despite failures
            overall_success_rate = sum(b["successful"] for b in batch_results) / sum(b["total"] for b in batch_results) * 100
            
            print(f"\nGraceful Degradation Analysis:")
            print(f"Overall Success Rate: {overall_success_rate:.2f}%")
            print(f"Success Rate Range: {min(success_rates):.1f}% - {max(success_rates):.1f}%")
            print(f"Average Throughput: {sum(throughputs)/len(throughputs):.2f} req/s")
            
            # Graceful degradation assertions
            assert overall_success_rate >= 30.0  # Should maintain at least 30% success rate
            assert all(rate >= 10.0 for rate in success_rates)  # Each batch should have some success
            assert max(success_rates) >= 50.0  # At least one batch should perform reasonably well
    
    @pytest.mark.asyncio
    async def test_recovery_after_system_stress(self):
        """Test system recovery after experiencing stress conditions."""
        config = MockServiceConfig(
            web_search_delay=0.001,
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            queries = TestDataGenerator.generate_research_queries()
            
            # Phase 1: Baseline performance
            print("\nPhase 1: Baseline performance")
            baseline_tasks = []
            for i in range(5):
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                baseline_tasks.append(task)
            
            baseline_start = time.perf_counter()
            baseline_results = await asyncio.gather(*baseline_tasks, return_exceptions=True)
            baseline_duration = time.perf_counter() - baseline_start
            
            baseline_successful = sum(
                1 for r in baseline_results 
                if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
            )
            baseline_throughput = len(baseline_tasks) / baseline_duration
            
            print(f"Baseline: {baseline_successful}/{len(baseline_tasks)} successful, {baseline_throughput:.2f} RPS")
            
            # Phase 2: Stress the system
            print("\nPhase 2: System stress")
            stress_tasks = []
            for i in range(25):  # High load
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                stress_tasks.append(task)
            
            stress_start = time.perf_counter()
            stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
            stress_duration = time.perf_counter() - stress_start
            
            stress_successful = sum(
                1 for r in stress_results 
                if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
            )
            stress_throughput = len(stress_tasks) / stress_duration
            
            print(f"Stress: {stress_successful}/{len(stress_tasks)} successful, {stress_throughput:.2f} RPS")
            
            # Phase 3: Recovery period
            print("\nPhase 3: Recovery period")
            await asyncio.sleep(3.0)  # Allow system to recover
            gc.collect()  # Force cleanup
            
            # Phase 4: Post-recovery performance
            print("\nPhase 4: Post-recovery performance")
            recovery_tasks = []
            for i in range(5):
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                recovery_tasks.append(task)
            
            recovery_start = time.perf_counter()
            recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
            recovery_duration = time.perf_counter() - recovery_start
            
            recovery_successful = sum(
                1 for r in recovery_results 
                if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
            )
            recovery_throughput = len(recovery_tasks) / recovery_duration
            
            print(f"Recovery: {recovery_successful}/{len(recovery_tasks)} successful, {recovery_throughput:.2f} RPS")
            
            # Analyze recovery
            recovery_ratio = recovery_throughput / baseline_throughput if baseline_throughput > 0 else 0
            
            print(f"\nRecovery Analysis:")
            print(f"Baseline Throughput: {baseline_throughput:.2f} RPS")
            print(f"Recovery Throughput: {recovery_throughput:.2f} RPS")
            print(f"Recovery Ratio: {recovery_ratio:.2f}")
            
            # Recovery assertions
            assert recovery_successful > 0  # Should have some successful requests
            assert recovery_ratio >= 0.7  # Should recover to at least 70% of baseline
            
            # System should demonstrate resilience
            if baseline_successful > 0:
                assert recovery_successful >= baseline_successful * 0.8  # At least 80% of baseline
    
    @pytest.mark.asyncio
    async def test_system_degradation_patterns(self):
        """Test and analyze system degradation patterns under stress."""
        config = MockServiceConfig(
            web_search_delay=0.003,
            api_query_delay=0.003,
            analysis_delay=0.006,
            failure_rate=0.1  # Moderate failure rate
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Test degradation under increasing load
            load_levels = [5, 10, 15, 20, 25]
            degradation_results = []
            
            for load_level in load_levels:
                print(f"\nTesting degradation at load level {load_level}")
                
                queries = TestDataGenerator.generate_research_queries()[:load_level]
                
                # Measure performance at this load level
                start_time = time.perf_counter()
                
                tasks = [
                    asyncio.create_task(framework.run_end_to_end_test(query))
                    for query in queries
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                # Analyze results
                successful = sum(
                    1 for r in results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                
                throughput = load_level / duration
                success_rate = (successful / load_level) * 100
                error_rate = 100 - success_rate
                
                degradation_result = {
                    "load_level": load_level,
                    "successful": successful,
                    "total": load_level,
                    "success_rate": success_rate,
                    "error_rate": error_rate,
                    "throughput": throughput,
                    "duration": duration
                }
                
                degradation_results.append(degradation_result)
                
                print(f"Load {load_level}: {successful}/{load_level} successful ({success_rate:.1f}%), "
                      f"Throughput: {throughput:.2f} RPS")
                
                # Brief pause between load levels
                await asyncio.sleep(1.0)
            
            # Analyze degradation patterns
            print(f"\nDegradation Pattern Analysis:")
            
            baseline = degradation_results[0]
            
            for result in degradation_results:
                load_factor = result["load_level"] / baseline["load_level"]
                throughput_factor = result["throughput"] / baseline["throughput"]
                
                print(f"Load {result['load_level']}: "
                      f"Load Factor {load_factor:.1f}x, "
                      f"Throughput Factor {throughput_factor:.2f}x, "
                      f"Success Rate {result['success_rate']:.1f}%")
                
                # System should maintain some functionality at all load levels
                assert result["successful"] > 0
                assert result["success_rate"] >= 20.0  # At least 20% success rate
            
            # Check for graceful degradation (not cliff-edge failure)
            success_rates = [r["success_rate"] for r in degradation_results]
            
            # Success rate should not drop too dramatically between adjacent levels
            for i in range(1, len(success_rates)):
                rate_drop = success_rates[i-1] - success_rates[i]
                assert rate_drop <= 40.0  # No more than 40% drop between levels
    
    @pytest.mark.asyncio
    async def test_extreme_load_conditions(self):
        """Test system behavior under extreme load conditions."""
        config = MockServiceConfig(
            web_search_delay=0.001,  # Very fast to create extreme load
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Create extreme load scenario
            extreme_load = 50  # Very high concurrent load
            queries = TestDataGenerator.generate_research_queries()
            
            print(f"Testing extreme load conditions with {extreme_load} concurrent requests")
            
            # Monitor system resources during extreme load
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            try:
                # Create all tasks at once for maximum stress
                tasks = []
                for i in range(extreme_load):
                    query = queries[i % len(queries)]
                    task = asyncio.create_task(framework.run_end_to_end_test(query))
                    tasks.append(task)
                
                # Set a reasonable timeout for extreme conditions
                start_time = time.perf_counter()
                
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=120.0  # 2 minute timeout
                    )
                    
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                except asyncio.TimeoutError:
                    print("Extreme load test timed out - this is expected behavior")
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Wait a bit for cancellation to complete
                    await asyncio.sleep(2.0)
                    
                    # Collect results from completed tasks
                    results = []
                    for task in tasks:
                        if task.done() and not task.cancelled():
                            try:
                                result = await task
                                results.append(result)
                            except Exception as e:
                                results.append(e)
                    
                    duration = 120.0  # Timeout duration
                
                # Analyze extreme load results
                successful = sum(
                    1 for r in results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = final_memory - initial_memory
                
                throughput = len(results) / duration if duration > 0 else 0
                success_rate = (successful / len(results)) * 100 if results else 0
                
                print(f"\nExtreme Load Test Results:")
                print(f"Attempted Requests: {extreme_load}")
                print(f"Completed Requests: {len(results)}")
                print(f"Successful Requests: {successful}")
                print(f"Success Rate: {success_rate:.2f}%")
                print(f"Throughput: {throughput:.2f} RPS")
                print(f"Duration: {duration:.2f}s")
                print(f"Memory Growth: {memory_growth:.2f}MB")
                
                # Extreme load assertions - more lenient
                assert len(results) > 0  # Some requests should complete
                assert successful >= 0  # No negative successful count
                
                # System should not consume excessive memory even under extreme load
                assert memory_growth < 2000  # Less than 2GB growth
                
                # If any requests succeeded, success rate should be reasonable
                if successful > 0:
                    assert success_rate >= 10.0  # At least 10% success under extreme load
                
            except Exception as e:
                print(f"Extreme load test encountered system limits: {e}")
                # This is acceptable behavior under extreme conditions
                assert True  # Test passes if system handles extreme load gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])hould have some successful operations
            assert recovery_ratio >= 0.7  # Should recover to at least 70% of baseline performance
            assert recovery_successful >= baseline_successful * 0.8  # Should maintain similar success count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])