"""
Memory usage and resource consumption tests.

Tests system memory usage patterns, detects memory leaks,
and monitors resource consumption under various conditions.
"""

import pytest
import asyncio
import time
import psutil
import gc
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from tests.integration.test_framework import (
    IntegrationTestFramework, MockServiceConfig, TestDataGenerator
)
from agent_scrivener.models.core import TaskStatus


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available system memory
    num_fds: int  # Number of file descriptors
    num_threads: int  # Number of threads


@dataclass
class ResourceUsageMetrics:
    """Comprehensive resource usage metrics."""
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    average_memory_mb: float
    memory_leak_rate_mb_per_hour: float
    peak_cpu_percent: float
    average_cpu_percent: float
    peak_threads: int
    average_threads: float
    peak_file_descriptors: int
    io_read_mb: float
    io_write_mb: float
    context_switches: int
    page_faults: int


class MemoryProfiler:
    """Profile memory usage during test execution."""
    
    def __init__(self, sampling_interval: float = 0.5):
        self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.initial_io_counters = None
    
    def start_profiling(self):
        """Start memory profiling."""
        self.snapshots.clear()
        self.monitoring = True
        self.initial_io_counters = self.process.io_counters()
        
        # Take initial snapshot
        self._take_snapshot()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_profiling(self) -> ResourceUsageMetrics:
        """Stop profiling and return metrics."""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Take final snapshot
        self._take_snapshot()
        
        return self._calculate_metrics()
    
    def _take_snapshot(self):
        """Take a memory usage snapshot."""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            system_memory = psutil.virtual_memory()
            
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=memory_percent,
                available_mb=system_memory.available / 1024 / 1024,
                num_fds=self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
                num_threads=self.process.num_threads()
            )
            
            self.snapshots.append(snapshot)
        except Exception as e:
            print(f"Error taking memory snapshot: {e}")
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread."""
        while self.monitoring:
            self._take_snapshot()
            time.sleep(self.sampling_interval)
    
    def _calculate_metrics(self) -> ResourceUsageMetrics:
        """Calculate comprehensive resource usage metrics."""
        if not self.snapshots:
            return ResourceUsageMetrics(
                initial_memory_mb=0, peak_memory_mb=0, final_memory_mb=0,
                memory_growth_mb=0, average_memory_mb=0, memory_leak_rate_mb_per_hour=0,
                peak_cpu_percent=0, average_cpu_percent=0, peak_threads=0,
                average_threads=0, peak_file_descriptors=0, io_read_mb=0,
                io_write_mb=0, context_switches=0, page_faults=0
            )
        
        # Memory metrics
        memory_values = [s.rss_mb for s in self.snapshots]
        initial_memory = memory_values[0]
        peak_memory = max(memory_values)
        final_memory = memory_values[-1]
        average_memory = sum(memory_values) / len(memory_values)
        memory_growth = final_memory - initial_memory
        
        # Calculate memory leak rate (MB per hour)
        if len(self.snapshots) > 1:
            duration_hours = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds() / 3600
            memory_leak_rate = memory_growth / duration_hours if duration_hours > 0 else 0
        else:
            memory_leak_rate = 0
        
        # CPU metrics (if available)
        try:
            cpu_percent = self.process.cpu_percent()
            peak_cpu = average_cpu = cpu_percent
        except:
            peak_cpu = average_cpu = 0
        
        # Thread metrics
        thread_counts = [s.num_threads for s in self.snapshots]
        peak_threads = max(thread_counts)
        average_threads = sum(thread_counts) / len(thread_counts)
        
        # File descriptor metrics
        fd_counts = [s.num_fds for s in self.snapshots]
        peak_fds = max(fd_counts) if fd_counts else 0
        
        # I/O metrics
        try:
            current_io = self.process.io_counters()
            if self.initial_io_counters:
                io_read_mb = (current_io.read_bytes - self.initial_io_counters.read_bytes) / 1024 / 1024
                io_write_mb = (current_io.write_bytes - self.initial_io_counters.write_bytes) / 1024 / 1024
            else:
                io_read_mb = io_write_mb = 0
        except:
            io_read_mb = io_write_mb = 0
        
        # System metrics
        try:
            context_switches = self.process.num_ctx_switches().voluntary
            page_faults = self.process.memory_info().pfaults if hasattr(self.process.memory_info(), 'pfaults') else 0
        except:
            context_switches = page_faults = 0
        
        return ResourceUsageMetrics(
            initial_memory_mb=initial_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_growth_mb=memory_growth,
            average_memory_mb=average_memory,
            memory_leak_rate_mb_per_hour=memory_leak_rate,
            peak_cpu_percent=peak_cpu,
            average_cpu_percent=average_cpu,
            peak_threads=peak_threads,
            average_threads=average_threads,
            peak_file_descriptors=peak_fds,
            io_read_mb=io_read_mb,
            io_write_mb=io_write_mb,
            context_switches=context_switches,
            page_faults=page_faults
        )


class TestMemoryUsage:
    """Test cases for memory usage and resource consumption."""
    
    @pytest.mark.asyncio
    async def test_single_session_memory_usage(self):
        """Test memory usage for a single research session."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        profiler = MemoryProfiler(sampling_interval=0.2)
        
        async with framework.test_environment():
            # Start memory profiling
            profiler.start_profiling()
            
            try:
                # Execute single research session
                query = "Memory usage test query for single session"
                result = await framework.run_end_to_end_test(query)
                
                # Verify session completed
                assert result["session"].status == TaskStatus.COMPLETED
                
                # Force garbage collection
                gc.collect()
                await asyncio.sleep(0.5)  # Allow cleanup
                
            finally:
                # Stop profiling and get metrics
                metrics = profiler.stop_profiling()
            
            # Memory usage assertions
            assert metrics.initial_memory_mb > 0
            assert metrics.peak_memory_mb >= metrics.initial_memory_mb
            assert metrics.memory_growth_mb < 200  # Should not grow by more than 200MB
            assert metrics.peak_threads < 50  # Should not create excessive threads
            assert metrics.peak_file_descriptors < 100  # Should not leak file descriptors
            
            print(f"\nSingle Session Memory Usage:")
            print(f"Initial Memory: {metrics.initial_memory_mb:.2f}MB")
            print(f"Peak Memory: {metrics.peak_memory_mb:.2f}MB")
            print(f"Final Memory: {metrics.final_memory_mb:.2f}MB")
            print(f"Memory Growth: {metrics.memory_growth_mb:.2f}MB")
            print(f"Peak Threads: {metrics.peak_threads}")
            print(f"Peak File Descriptors: {metrics.peak_file_descriptors}")
            print(f"I/O Read: {metrics.io_read_mb:.2f}MB")
            print(f"I/O Write: {metrics.io_write_mb:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_multiple_sessions_memory_pattern(self):
        """Test memory usage patterns across multiple sessions."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        profiler = MemoryProfiler(sampling_interval=0.1)
        
        async with framework.test_environment():
            profiler.start_profiling()
            
            try:
                queries = TestDataGenerator.generate_research_queries()[:10]
                session_metrics = []
                
                for i, query in enumerate(queries):
                    print(f"Executing session {i+1}/10")
                    
                    # Take memory snapshot before session
                    pre_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    # Execute session
                    result = await framework.run_end_to_end_test(query)
                    assert result["session"].status == TaskStatus.COMPLETED
                    
                    # Take memory snapshot after session
                    post_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    session_metrics.append({
                        "session": i + 1,
                        "pre_memory_mb": pre_memory,
                        "post_memory_mb": post_memory,
                        "memory_delta_mb": post_memory - pre_memory
                    })
                    
                    # Force garbage collection between sessions
                    gc.collect()
                    await asyncio.sleep(0.2)
                
            finally:
                metrics = profiler.stop_profiling()
            
            # Analyze memory patterns
            memory_deltas = [s["memory_delta_mb"] for s in session_metrics]
            average_delta = sum(memory_deltas) / len(memory_deltas)
            max_delta = max(memory_deltas)
            
            # Check for memory leaks
            increasing_sessions = sum(1 for delta in memory_deltas if delta > 10)  # > 10MB increase
            leak_indicator = increasing_sessions / len(memory_deltas)
            
            # Assertions
            assert metrics.memory_growth_mb < 500  # Total growth < 500MB
            assert average_delta < 50  # Average growth per session < 50MB
            assert max_delta < 100  # No single session should grow by > 100MB
            assert leak_indicator < 0.6  # Less than 60% of sessions should show significant growth
            assert metrics.memory_leak_rate_mb_per_hour < 1000  # Leak rate < 1GB/hour
            
            print(f"\nMultiple Sessions Memory Pattern:")
            print(f"Total Sessions: {len(session_metrics)}")
            print(f"Total Memory Growth: {metrics.memory_growth_mb:.2f}MB")
            print(f"Average Growth per Session: {average_delta:.2f}MB")
            print(f"Max Growth per Session: {max_delta:.2f}MB")
            print(f"Memory Leak Rate: {metrics.memory_leak_rate_mb_per_hour:.2f}MB/hour")
            print(f"Leak Indicator: {leak_indicator:.2%}")
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions_memory_scaling(self):
        """Test memory scaling with concurrent sessions."""
        config = MockServiceConfig(
            web_search_delay=0.002,
            api_query_delay=0.002,
            analysis_delay=0.005,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Test different concurrency levels
            concurrency_levels = [1, 3, 5, 10]
            scaling_results = []
            
            for concurrency in concurrency_levels:
                profiler = MemoryProfiler(sampling_interval=0.1)
                profiler.start_profiling()
                
                try:
                    # Execute concurrent sessions
                    queries = TestDataGenerator.generate_research_queries()[:concurrency]
                    tasks = [
                        asyncio.create_task(framework.run_end_to_end_test(query))
                        for query in queries
                    ]
                    
                    results = await asyncio.gather(*tasks)
                    
                    # Verify all sessions completed
                    successful_sessions = sum(
                        1 for r in results 
                        if r["session"].status == TaskStatus.COMPLETED
                    )
                    
                finally:
                    metrics = profiler.stop_profiling()
                
                scaling_results.append({
                    "concurrency": concurrency,
                    "successful_sessions": successful_sessions,
                    "memory_growth_mb": metrics.memory_growth_mb,
                    "peak_memory_mb": metrics.peak_memory_mb,
                    "peak_threads": metrics.peak_threads,
                    "peak_fds": metrics.peak_file_descriptors
                })
                
                print(f"\nConcurrency {concurrency} Memory Scaling:")
                print(f"Successful Sessions: {successful_sessions}/{concurrency}")
                print(f"Memory Growth: {metrics.memory_growth_mb:.2f}MB")
                print(f"Peak Memory: {metrics.peak_memory_mb:.2f}MB")
                print(f"Peak Threads: {metrics.peak_threads}")
                
                # Brief pause between tests
                gc.collect()
                await asyncio.sleep(1.0)
            
            # Analyze scaling characteristics
            baseline = scaling_results[0]
            
            for result in scaling_results[1:]:
                concurrency_factor = result["concurrency"] / baseline["concurrency"]
                memory_factor = result["memory_growth_mb"] / max(baseline["memory_growth_mb"], 1)
                
                # Memory should scale sub-linearly (not 1:1 with concurrency)
                assert memory_factor < concurrency_factor * 1.5  # Allow some overhead
                assert result["peak_memory_mb"] < baseline["peak_memory_mb"] * concurrency_factor * 2
                assert result["peak_threads"] < baseline["peak_threads"] * concurrency_factor * 2
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks over extended operation."""
        config = MockServiceConfig(
            web_search_delay=0.001,
            api_query_delay=0.001,
            analysis_delay=0.002,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        profiler = MemoryProfiler(sampling_interval=0.5)
        
        async with framework.test_environment():
            profiler.start_profiling()
            
            try:
                # Run many short sessions to detect leaks
                queries = TestDataGenerator.generate_research_queries()
                num_sessions = 50
                
                for i in range(num_sessions):
                    query = queries[i % len(queries)]
                    
                    try:
                        result = await framework.run_end_to_end_test(query)
                        # Don't assert completion - focus on memory behavior
                    except Exception:
                        pass  # Continue even if some sessions fail
                    
                    # Periodic garbage collection
                    if i % 10 == 0:
                        gc.collect()
                        await asyncio.sleep(0.1)
                
                # Final garbage collection
                gc.collect()
                await asyncio.sleep(1.0)
                
            finally:
                metrics = profiler.stop_profiling()
            
            # Analyze for memory leaks
            memory_samples = [s.rss_mb for s in profiler.snapshots]
            
            if len(memory_samples) > 10:
                # Calculate trend in memory usage
                early_samples = memory_samples[:len(memory_samples)//3]
                late_samples = memory_samples[-len(memory_samples)//3:]
                
                early_avg = sum(early_samples) / len(early_samples)
                late_avg = sum(late_samples) / len(late_samples)
                
                memory_trend = late_avg - early_avg
                
                print(f"\nMemory Leak Detection Results:")
                print(f"Sessions Executed: {num_sessions}")
                print(f"Initial Memory: {metrics.initial_memory_mb:.2f}MB")
                print(f"Final Memory: {metrics.final_memory_mb:.2f}MB")
                print(f"Total Growth: {metrics.memory_growth_mb:.2f}MB")
                print(f"Memory Trend: {memory_trend:.2f}MB")
                print(f"Leak Rate: {metrics.memory_leak_rate_mb_per_hour:.2f}MB/hour")
                
                # Memory leak assertions
                assert metrics.memory_growth_mb < 300  # Total growth < 300MB
                assert abs(memory_trend) < 100  # Trend should be stable
                assert metrics.memory_leak_rate_mb_per_hour < 500  # Leak rate < 500MB/hour
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_after_failures(self):
        """Test resource cleanup when sessions fail."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.5  # High failure rate
        )
        
        framework = IntegrationTestFramework(config)
        profiler = MemoryProfiler(sampling_interval=0.2)
        
        async with framework.test_environment():
            profiler.start_profiling()
            
            try:
                # Execute sessions with high failure rate
                queries = TestDataGenerator.generate_research_queries()[:20]
                results = []
                
                for query in queries:
                    try:
                        result = await framework.run_end_to_end_test(query)
                        results.append(result)
                    except Exception as e:
                        results.append({"error": str(e)})
                
                # Force cleanup
                gc.collect()
                await asyncio.sleep(1.0)
                
            finally:
                metrics = profiler.stop_profiling()
            
            # Count successful vs failed sessions
            successful_sessions = sum(
                1 for r in results 
                if "session" in r and r["session"].status == TaskStatus.COMPLETED
            )
            failed_sessions = len(results) - successful_sessions
            
            print(f"\nResource Cleanup After Failures:")
            print(f"Successful Sessions: {successful_sessions}")
            print(f"Failed Sessions: {failed_sessions}")
            print(f"Memory Growth: {metrics.memory_growth_mb:.2f}MB")
            print(f"Peak Memory: {metrics.peak_memory_mb:.2f}MB")
            print(f"Final Memory: {metrics.final_memory_mb:.2f}MB")
            
            # Even with failures, resource usage should be reasonable
            assert metrics.memory_growth_mb < 400  # Growth should be limited
            assert metrics.peak_memory_mb < metrics.initial_memory_mb + 500  # Peak should be reasonable
            assert failed_sessions > 0  # Should have some failures (validates test setup)
    
    @pytest.mark.asyncio
    async def test_long_running_session_memory_stability(self):
        """Test memory stability during long-running sessions."""
        config = MockServiceConfig(
            web_search_delay=0.02,  # Longer delays for extended session
            api_query_delay=0.02,
            analysis_delay=0.05,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        profiler = MemoryProfiler(sampling_interval=1.0)  # Sample every second
        
        async with framework.test_environment():
            profiler.start_profiling()
            
            try:
                # Execute a longer research session
                query = "Long-running memory stability test with comprehensive research requirements"
                result = await framework.run_end_to_end_test(query)
                
                assert result["session"].status == TaskStatus.COMPLETED
                
            finally:
                metrics = profiler.stop_profiling()
            
            # Analyze memory stability during execution
            memory_samples = [s.rss_mb for s in profiler.snapshots]
            
            if len(memory_samples) > 5:
                # Calculate memory variance during execution
                import statistics
                memory_variance = statistics.variance(memory_samples)
                memory_std_dev = statistics.stdev(memory_samples)
                
                print(f"\nLong-Running Session Memory Stability:")
                print(f"Session Duration: {len(profiler.snapshots)} samples")
                print(f"Memory Variance: {memory_variance:.2f}")
                print(f"Memory Std Dev: {memory_std_dev:.2f}MB")
                print(f"Memory Growth: {metrics.memory_growth_mb:.2f}MB")
                print(f"Peak Memory: {metrics.peak_memory_mb:.2f}MB")
                
                # Memory should be relatively stable during execution
                assert memory_std_dev < 100  # Standard deviation < 100MB
                assert metrics.memory_growth_mb < 200  # Total growth < 200MB


    @pytest.mark.asyncio
    async def test_memory_usage_under_concurrent_load(self):
        """Test memory usage patterns under concurrent load."""
        config = MockServiceConfig(
            web_search_delay=0.002,
            api_query_delay=0.002,
            analysis_delay=0.005,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Test memory usage with different concurrent loads
            concurrency_levels = [1, 5, 10, 15]
            memory_results = []
            
            for concurrency in concurrency_levels:
                profiler = MemoryProfiler(sampling_interval=0.2)
                profiler.start_profiling()
                
                try:
                    queries = TestDataGenerator.generate_research_queries()[:concurrency]
                    tasks = [
                        asyncio.create_task(framework.run_end_to_end_test(query))
                        for query in queries
                    ]
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    successful = sum(
                        1 for r in results 
                        if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                    )
                    
                finally:
                    metrics = profiler.stop_profiling()
                
                memory_per_request = metrics.memory_growth_mb / concurrency if concurrency > 0 else 0
                
                memory_results.append({
                    "concurrency": concurrency,
                    "successful_requests": successful,
                    "memory_growth_mb": metrics.memory_growth_mb,
                    "peak_memory_mb": metrics.peak_memory_mb,
                    "memory_per_request_mb": memory_per_request,
                    "peak_threads": metrics.peak_threads
                })
                
                print(f"\nConcurrency {concurrency} Memory Results:")
                print(f"  Memory Growth: {metrics.memory_growth_mb:.2f}MB")
                print(f"  Memory per Request: {memory_per_request:.2f}MB")
                print(f"  Peak Memory: {metrics.peak_memory_mb:.2f}MB")
                print(f"  Peak Threads: {metrics.peak_threads}")
                
                # Memory usage assertions
                assert metrics.memory_growth_mb < concurrency * 50  # < 50MB per concurrent request
                assert memory_per_request < 100  # < 100MB per request
                assert metrics.peak_threads < concurrency * 10  # Reasonable thread usage
                
                # Brief pause between tests
                gc.collect()
                await asyncio.sleep(1.0)
            
            # Analyze memory scaling
            baseline = memory_results[0]
            
            for result in memory_results[1:]:
                concurrency_factor = result["concurrency"] / baseline["concurrency"]
                memory_factor = result["memory_growth_mb"] / max(baseline["memory_growth_mb"], 1)
                
                # Memory should scale sub-linearly
                assert memory_factor <= concurrency_factor * 1.5
                
                print(f"Concurrency {result['concurrency']}: Memory scaling factor {memory_factor:.2f}")
    
    @pytest.mark.asyncio
    async def test_resource_consumption_monitoring(self):
        """Test comprehensive resource consumption monitoring."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        profiler = MemoryProfiler(sampling_interval=0.5)
        
        async with framework.test_environment():
            profiler.start_profiling()
            
            try:
                # Execute a series of operations to monitor resource consumption
                queries = TestDataGenerator.generate_research_queries()[:8]
                
                for i, query in enumerate(queries):
                    print(f"Executing operation {i+1}/{len(queries)}")
                    
                    try:
                        result = await framework.run_end_to_end_test(query)
                        if result["session"].status == TaskStatus.COMPLETED:
                            print(f"  ✓ Operation {i+1} completed successfully")
                        else:
                            print(f"  ⚠ Operation {i+1} completed with issues")
                    except Exception as e:
                        print(f"  ✗ Operation {i+1} failed: {e}")
                    
                    # Monitor resources between operations
                    current_metrics = profiler.resource_monitor.get_metrics()
                    print(f"  Memory: {current_metrics['current_memory_mb']:.1f}MB, "
                          f"CPU: {current_metrics['average_cpu_percent']:.1f}%")
                    
                    await asyncio.sleep(0.5)
                
            finally:
                metrics = profiler.stop_profiling()
            
            # Comprehensive resource analysis
            print(f"\nComprehensive Resource Consumption Analysis:")
            print(f"Initial Memory: {metrics.initial_memory_mb:.2f}MB")
            print(f"Peak Memory: {metrics.peak_memory_mb:.2f}MB")
            print(f"Final Memory: {metrics.final_memory_mb:.2f}MB")
            print(f"Total Memory Growth: {metrics.memory_growth_mb:.2f}MB")
            print(f"Average Memory: {metrics.average_memory_mb:.2f}MB")
            print(f"Memory Leak Rate: {metrics.memory_leak_rate_mb_per_hour:.2f}MB/hour")
            print(f"Peak CPU: {metrics.peak_cpu_percent:.2f}%")
            print(f"Average CPU: {metrics.average_cpu_percent:.2f}%")
            print(f"Peak Threads: {metrics.peak_threads}")
            print(f"Average Threads: {metrics.average_threads:.1f}")
            print(f"Peak File Descriptors: {metrics.peak_file_descriptors}")
            print(f"I/O Read: {metrics.io_read_mb:.2f}MB")
            print(f"I/O Write: {metrics.io_write_mb:.2f}MB")
            print(f"Context Switches: {metrics.context_switches}")
            print(f"Page Faults: {metrics.page_faults}")
            
            # Resource consumption assertions
            assert metrics.memory_growth_mb < 400  # Total growth < 400MB
            assert metrics.memory_leak_rate_mb_per_hour < 1000  # Leak rate < 1GB/hour
            assert metrics.peak_threads < 100  # Reasonable thread count
            assert metrics.peak_file_descriptors < 200  # Reasonable FD usage
            assert metrics.io_read_mb >= 0 and metrics.io_write_mb >= 0  # Valid I/O metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])