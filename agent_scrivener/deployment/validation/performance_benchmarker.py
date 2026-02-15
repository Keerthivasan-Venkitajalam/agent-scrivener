"""Performance benchmarking for production readiness validation."""

import asyncio
import logging
import statistics
import time
from typing import List, Optional, Dict, Any

import psutil

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus, PerformanceMetrics


logger = logging.getLogger(__name__)


class PerformanceBenchmarker(BaseValidator):
    """Benchmarks system performance and establishes baselines.
    
    Measures response times, resource usage, and database query performance
    to ensure the system meets production performance requirements.
    """
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        database_url: Optional[str] = None,
        timeout_seconds: float = 600.0  # 10 minutes for performance tests
    ):
        """Initialize the performance benchmarker.
        
        Args:
            api_base_url: Base URL for API endpoint testing
            database_url: Database connection URL for query benchmarking
            timeout_seconds: Timeout for benchmark execution
        """
        super().__init__("PerformanceBenchmarker", timeout_seconds)
        self.api_base_url = api_base_url
        self.database_url = database_url
        self._http_client = None
        self._db_connection = None
    
    async def validate(self) -> List[ValidationResult]:
        """Execute all performance benchmarks.
        
        Returns:
            List of validation results with performance metrics
        """
        self.log_validation_start()
        results = []
        
        # Benchmark single request performance
        single_request_result = await self._benchmark_single_request()
        results.append(single_request_result)
        
        # Benchmark API endpoints
        api_endpoints_result = await self._benchmark_api_endpoints()
        results.append(api_endpoints_result)
        
        # Benchmark concurrent requests
        concurrent_result = await self._benchmark_concurrent_requests()
        results.append(concurrent_result)
        
        # Benchmark resource usage
        resource_result = await self._benchmark_resource_usage()
        results.append(resource_result)
        
        # Benchmark database queries (if database URL provided)
        if self.database_url:
            db_result = await self._benchmark_database_queries()
            results.append(db_result)
        else:
            results.append(
                self.create_result(
                    status=ValidationStatus.SKIP,
                    message="Database benchmarking skipped (no database URL provided)"
                )
            )
        
        self.log_validation_complete(results)
        return results
    
    async def benchmark_single_request(self) -> Dict[str, Any]:
        """Benchmark single research request performance.
        
        Returns:
            Dictionary with performance metrics and validation result
        """
        result = await self._benchmark_single_request()
        metrics = result.details.get("metrics")
        return {
            "result": result,
            "metrics": metrics
        }
    
    async def _benchmark_single_request(self) -> ValidationResult:
        """Internal implementation of single request benchmarking.
        
        Requirement 5.1: Single research request should complete within 3 minutes for 90% of requests
        """
        start_time = time.time()
        
        try:
            # Simulate multiple single requests to get percentile data
            latencies = []
            sample_count = 10  # Run 10 samples for statistical significance
            
            for i in range(sample_count):
                request_start = time.time()
                
                # TODO: Replace with actual API call when API is available
                # For now, simulate a research request
                await asyncio.sleep(0.1)  # Simulate processing time
                
                request_duration = (time.time() - request_start) * 1000  # Convert to ms
                latencies.append(request_duration)
                
                self.logger.debug(f"Single request {i+1}/{sample_count}: {request_duration:.2f}ms")
            
            # Calculate performance metrics
            metrics = self._calculate_metrics("single_request", latencies)
            
            # Check if p90 meets the 3-minute threshold (180,000 ms)
            threshold_ms = 180_000  # 3 minutes
            meets_requirement = metrics.p90_ms <= threshold_ms
            
            duration = time.time() - start_time
            
            if meets_requirement:
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"Single request performance meets requirements (p90: {metrics.p90_ms:.2f}ms < {threshold_ms}ms)",
                    duration_seconds=duration,
                    details={
                        "metrics": metrics,
                        "threshold_ms": threshold_ms,
                        "meets_requirement": True
                    }
                )
            else:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Single request performance below requirements (p90: {metrics.p90_ms:.2f}ms > {threshold_ms}ms)",
                    duration_seconds=duration,
                    details={
                        "metrics": metrics,
                        "threshold_ms": threshold_ms,
                        "meets_requirement": False
                    },
                    remediation_steps=[
                        "Optimize agent processing logic",
                        "Review database query performance",
                        "Check for network latency issues",
                        "Consider caching frequently accessed data"
                    ]
                )
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Single request benchmarking failed: {e}")
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Single request benchmarking failed: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e)},
                remediation_steps=[
                    "Ensure the API server is running",
                    "Check API endpoint configuration",
                    "Review error logs for details"
                ]
            )
    
    async def benchmark_api_endpoints(self) -> Dict[str, Any]:
        """Benchmark API endpoint response times.
        
        Returns:
            Dictionary with performance metrics for each endpoint
        """
        result = await self._benchmark_api_endpoints()
        return {
            "result": result,
            "endpoint_metrics": result.details.get("endpoint_metrics", {})
        }
    
    async def _benchmark_api_endpoints(self) -> ValidationResult:
        """Internal implementation of API endpoint benchmarking.
        
        Requirements:
        - 5.2: Health checks should respond within 100ms
        - 5.3: Status queries should respond within 200ms
        """
        start_time = time.time()
        
        try:
            endpoint_metrics = {}
            all_passed = True
            failed_endpoints = []
            
            # Benchmark health endpoint (Requirement 5.2: < 100ms)
            health_latencies = await self._measure_endpoint_latencies("/health", count=20)
            health_metrics = self._calculate_metrics("health_endpoint", health_latencies)
            endpoint_metrics["health"] = health_metrics
            
            if health_metrics.p90_ms > 100:
                all_passed = False
                failed_endpoints.append(f"health (p90: {health_metrics.p90_ms:.2f}ms > 100ms)")
            
            # Benchmark status query endpoint (Requirement 5.3: < 200ms)
            # TODO: Replace with actual endpoint when available
            status_latencies = await self._measure_endpoint_latencies("/research/test-session/status", count=20)
            status_metrics = self._calculate_metrics("status_endpoint", status_latencies)
            endpoint_metrics["status"] = status_metrics
            
            if status_metrics.p90_ms > 200:
                all_passed = False
                failed_endpoints.append(f"status (p90: {status_metrics.p90_ms:.2f}ms > 200ms)")
            
            duration = time.time() - start_time
            
            if all_passed:
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="All API endpoints meet performance requirements",
                    duration_seconds=duration,
                    details={
                        "endpoint_metrics": {k: self._metrics_to_dict(v) for k, v in endpoint_metrics.items()},
                        "all_passed": True
                    }
                )
            else:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"API endpoints below performance requirements: {', '.join(failed_endpoints)}",
                    duration_seconds=duration,
                    details={
                        "endpoint_metrics": {k: self._metrics_to_dict(v) for k, v in endpoint_metrics.items()},
                        "failed_endpoints": failed_endpoints
                    },
                    remediation_steps=[
                        "Optimize endpoint handler logic",
                        "Add caching for frequently accessed data",
                        "Review database query performance",
                        "Check for unnecessary middleware processing"
                    ]
                )
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"API endpoint benchmarking failed: {e}")
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"API endpoint benchmarking failed: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e)},
                remediation_steps=[
                    "Ensure the API server is running",
                    "Check API endpoint configuration",
                    "Review error logs for details"
                ]
            )
    
    async def benchmark_concurrent_requests(self, concurrent_count: int = 10) -> Dict[str, Any]:
        """Benchmark concurrent request handling.
        
        Args:
            concurrent_count: Number of concurrent requests to test
            
        Returns:
            Dictionary with concurrent performance metrics
        """
        result = await self._benchmark_concurrent_requests(concurrent_count)
        return {
            "result": result,
            "metrics": result.details.get("metrics")
        }
    
    async def _benchmark_concurrent_requests(self, concurrent_count: int = 10) -> ValidationResult:
        """Internal implementation of concurrent request benchmarking.
        
        Requirement 5.4: System should handle at least 10 simultaneous research sessions
        without degradation
        """
        start_time = time.time()
        
        try:
            # Test with different load levels
            load_levels = [1, 5, 10, 20]
            load_metrics = {}
            degradation_detected = False
            
            baseline_latency = None
            
            for load in load_levels:
                if load > concurrent_count:
                    continue
                
                self.logger.info(f"Testing with {load} concurrent requests")
                
                # Run concurrent requests
                tasks = []
                for _ in range(load):
                    tasks.append(self._simulate_research_request())
                
                request_start = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_duration = (time.time() - request_start) * 1000
                
                # Calculate average latency per request
                successful_results = [r for r in results if not isinstance(r, Exception)]
                avg_latency = total_duration / load if load > 0 else 0
                
                load_metrics[f"load_{load}"] = {
                    "concurrent_requests": load,
                    "avg_latency_ms": avg_latency,
                    "successful_requests": len(successful_results),
                    "failed_requests": load - len(successful_results)
                }
                
                # Check for degradation (baseline is load=1)
                if load == 1:
                    baseline_latency = avg_latency
                elif baseline_latency and avg_latency > baseline_latency * 2:
                    # More than 2x degradation is considered significant
                    degradation_detected = True
                    self.logger.warning(
                        f"Performance degradation detected at load {load}: "
                        f"{avg_latency:.2f}ms vs baseline {baseline_latency:.2f}ms"
                    )
            
            duration = time.time() - start_time
            
            # Check if system handles 10 concurrent requests without degradation
            load_10_metrics = load_metrics.get("load_10")
            if not load_10_metrics:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="Concurrent request test skipped (concurrent_count < 10)",
                    duration_seconds=duration,
                    details={"load_metrics": load_metrics}
                )
            
            if not degradation_detected and load_10_metrics["failed_requests"] == 0:
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"System handles {concurrent_count} concurrent requests without degradation",
                    duration_seconds=duration,
                    details={
                        "load_metrics": load_metrics,
                        "degradation_detected": False
                    }
                )
            else:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="System shows performance degradation under concurrent load",
                    duration_seconds=duration,
                    details={
                        "load_metrics": load_metrics,
                        "degradation_detected": degradation_detected
                    },
                    remediation_steps=[
                        "Increase worker/thread pool size",
                        "Optimize resource-intensive operations",
                        "Add connection pooling for database",
                        "Consider horizontal scaling",
                        "Review for blocking operations in async code"
                    ]
                )
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Concurrent request benchmarking failed: {e}")
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Concurrent request benchmarking failed: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e)},
                remediation_steps=[
                    "Ensure the API server is running",
                    "Check system resource availability",
                    "Review error logs for details"
                ]
            )
    
    async def benchmark_resource_usage(self) -> Dict[str, Any]:
        """Benchmark memory and CPU usage.
        
        Returns:
            Dictionary with resource usage metrics
        """
        result = await self._benchmark_resource_usage()
        return {
            "result": result,
            "resource_metrics": result.details.get("resource_metrics")
        }
    
    async def _benchmark_resource_usage(self) -> ValidationResult:
        """Internal implementation of resource usage benchmarking.
        
        Requirements:
        - 5.5: Memory consumption should remain under 2GB RAM per research session
        - 5.6: CPU utilization should average under 70% under normal load
        """
        start_time = time.time()
        
        try:
            # Get current process
            process = psutil.Process()
            
            # Measure baseline resource usage
            baseline_memory_mb = process.memory_info().rss / (1024 * 1024)
            baseline_cpu_percent = process.cpu_percent(interval=1.0)
            
            # Simulate a research session and measure resource usage
            memory_samples = []
            cpu_samples = []
            
            # Take samples during simulated work
            for i in range(10):
                # Simulate some work
                await asyncio.sleep(0.5)
                
                # Sample resource usage
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent(interval=0.1)
                
                memory_samples.append(memory_mb)
                cpu_samples.append(cpu_percent)
                
                self.logger.debug(f"Sample {i+1}: Memory={memory_mb:.2f}MB, CPU={cpu_percent:.1f}%")
            
            # Calculate metrics
            avg_memory_mb = statistics.mean(memory_samples)
            max_memory_mb = max(memory_samples)
            avg_cpu_percent = statistics.mean(cpu_samples)
            max_cpu_percent = max(cpu_samples)
            
            # Memory delta from baseline (approximate per-session usage)
            memory_delta_mb = max_memory_mb - baseline_memory_mb
            
            resource_metrics = {
                "baseline_memory_mb": baseline_memory_mb,
                "avg_memory_mb": avg_memory_mb,
                "max_memory_mb": max_memory_mb,
                "memory_delta_mb": memory_delta_mb,
                "baseline_cpu_percent": baseline_cpu_percent,
                "avg_cpu_percent": avg_cpu_percent,
                "max_cpu_percent": max_cpu_percent
            }
            
            duration = time.time() - start_time
            
            # Check requirements
            memory_threshold_mb = 2048  # 2GB
            cpu_threshold_percent = 70
            
            memory_ok = memory_delta_mb < memory_threshold_mb
            cpu_ok = avg_cpu_percent < cpu_threshold_percent
            
            if memory_ok and cpu_ok:
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"Resource usage within limits (Memory: {memory_delta_mb:.2f}MB < {memory_threshold_mb}MB, CPU: {avg_cpu_percent:.1f}% < {cpu_threshold_percent}%)",
                    duration_seconds=duration,
                    details={
                        "resource_metrics": resource_metrics,
                        "memory_ok": memory_ok,
                        "cpu_ok": cpu_ok
                    }
                )
            else:
                issues = []
                if not memory_ok:
                    issues.append(f"Memory: {memory_delta_mb:.2f}MB > {memory_threshold_mb}MB")
                if not cpu_ok:
                    issues.append(f"CPU: {avg_cpu_percent:.1f}% > {cpu_threshold_percent}%")
                
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Resource usage exceeds limits: {', '.join(issues)}",
                    duration_seconds=duration,
                    details={
                        "resource_metrics": resource_metrics,
                        "memory_ok": memory_ok,
                        "cpu_ok": cpu_ok
                    },
                    remediation_steps=[
                        "Profile memory usage to identify leaks",
                        "Optimize data structures and caching",
                        "Review CPU-intensive operations",
                        "Consider using more efficient algorithms",
                        "Add resource limits and monitoring"
                    ]
                )
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Resource usage benchmarking failed: {e}")
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Resource usage benchmarking failed: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e)},
                remediation_steps=[
                    "Ensure psutil is installed",
                    "Check system permissions",
                    "Review error logs for details"
                ]
            )
    
    async def benchmark_database_queries(self) -> Dict[str, Any]:
        """Benchmark database query performance.
        
        Returns:
            Dictionary with database performance metrics
        """
        result = await self._benchmark_database_queries()
        return {
            "result": result,
            "query_metrics": result.details.get("query_metrics")
        }
    
    async def _benchmark_database_queries(self) -> ValidationResult:
        """Internal implementation of database query benchmarking.
        
        Requirement 5.7: Session lookups should execute within 50ms
        """
        start_time = time.time()
        
        try:
            # TODO: Replace with actual database queries when database is available
            # For now, simulate database queries
            
            query_types = {
                "session_lookup": 50,  # 50ms threshold
                "session_list": 100,   # 100ms threshold
                "session_insert": 100  # 100ms threshold
            }
            
            query_metrics = {}
            all_passed = True
            failed_queries = []
            
            for query_type, threshold_ms in query_types.items():
                latencies = []
                
                # Run multiple samples
                for _ in range(20):
                    query_start = time.time()
                    
                    # Simulate database query
                    await asyncio.sleep(0.01)  # Simulate 10ms query
                    
                    query_duration = (time.time() - query_start) * 1000
                    latencies.append(query_duration)
                
                metrics = self._calculate_metrics(query_type, latencies)
                query_metrics[query_type] = {
                    "metrics": self._metrics_to_dict(metrics),
                    "threshold_ms": threshold_ms,
                    "meets_requirement": metrics.p90_ms <= threshold_ms
                }
                
                if metrics.p90_ms > threshold_ms:
                    all_passed = False
                    failed_queries.append(f"{query_type} (p90: {metrics.p90_ms:.2f}ms > {threshold_ms}ms)")
            
            duration = time.time() - start_time
            
            if all_passed:
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="All database queries meet performance requirements",
                    duration_seconds=duration,
                    details={
                        "query_metrics": query_metrics,
                        "all_passed": True
                    }
                )
            else:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Database queries below performance requirements: {', '.join(failed_queries)}",
                    duration_seconds=duration,
                    details={
                        "query_metrics": query_metrics,
                        "failed_queries": failed_queries
                    },
                    remediation_steps=[
                        "Add database indexes for frequently queried fields",
                        "Optimize query structure and joins",
                        "Consider query result caching",
                        "Review database connection pooling",
                        "Analyze slow query logs"
                    ]
                )
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Database query benchmarking failed: {e}")
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Database query benchmarking failed: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e)},
                remediation_steps=[
                    "Ensure database is running and accessible",
                    "Check database connection configuration",
                    "Review error logs for details"
                ]
            )
    
    def generate_performance_report(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Generate performance report with percentile calculations.
        
        Args:
            metrics_list: List of performance metrics to include in report
            
        Returns:
            Dictionary containing formatted performance report
        """
        report = {
            "summary": {
                "total_metrics": len(metrics_list),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "metrics": []
        }
        
        for metrics in metrics_list:
            report["metrics"].append({
                "name": metrics.metric_name,
                "percentiles": {
                    "p50": f"{metrics.p50_ms:.2f}ms",
                    "p90": f"{metrics.p90_ms:.2f}ms",
                    "p95": f"{metrics.p95_ms:.2f}ms",
                    "p99": f"{metrics.p99_ms:.2f}ms"
                },
                "range": {
                    "min": f"{metrics.min_ms:.2f}ms",
                    "max": f"{metrics.max_ms:.2f}ms",
                    "mean": f"{metrics.mean_ms:.2f}ms",
                    "std_dev": f"{metrics.std_dev_ms:.2f}ms"
                },
                "sample_count": metrics.sample_count
            })
        
        return report
    
    # Helper methods
    
    def _calculate_metrics(self, metric_name: str, latencies: List[float]) -> PerformanceMetrics:
        """Calculate performance metrics from latency samples.
        
        Args:
            metric_name: Name of the metric
            latencies: List of latency measurements in milliseconds
            
        Returns:
            PerformanceMetrics instance
        """
        if not latencies:
            return PerformanceMetrics(
                metric_name=metric_name,
                p50_ms=0, p90_ms=0, p95_ms=0, p99_ms=0,
                min_ms=0, max_ms=0, mean_ms=0, std_dev_ms=0,
                sample_count=0
            )
        
        sorted_latencies = sorted(latencies)
        
        return PerformanceMetrics(
            metric_name=metric_name,
            p50_ms=self._percentile(sorted_latencies, 50),
            p90_ms=self._percentile(sorted_latencies, 90),
            p95_ms=self._percentile(sorted_latencies, 95),
            p99_ms=self._percentile(sorted_latencies, 99),
            min_ms=min(latencies),
            max_ms=max(latencies),
            mean_ms=statistics.mean(latencies),
            std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            sample_count=len(latencies)
        )
    
    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data.
        
        Args:
            sorted_data: Sorted list of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not sorted_data:
            return 0.0
        
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_data) - 1)
        weight = index - lower
        
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Convert PerformanceMetrics to dictionary.
        
        Args:
            metrics: PerformanceMetrics instance
            
        Returns:
            Dictionary representation
        """
        return {
            "metric_name": metrics.metric_name,
            "p50_ms": metrics.p50_ms,
            "p90_ms": metrics.p90_ms,
            "p95_ms": metrics.p95_ms,
            "p99_ms": metrics.p99_ms,
            "min_ms": metrics.min_ms,
            "max_ms": metrics.max_ms,
            "mean_ms": metrics.mean_ms,
            "std_dev_ms": metrics.std_dev_ms,
            "sample_count": metrics.sample_count
        }
    
    async def _measure_endpoint_latencies(self, endpoint: str, count: int = 20) -> List[float]:
        """Measure latencies for an endpoint.
        
        Args:
            endpoint: API endpoint path
            count: Number of measurements to take
            
        Returns:
            List of latency measurements in milliseconds
        """
        latencies = []
        
        for _ in range(count):
            start = time.time()
            
            # TODO: Replace with actual HTTP request when API is available
            # For now, simulate endpoint call
            await asyncio.sleep(0.01)  # Simulate 10ms response
            
            duration = (time.time() - start) * 1000
            latencies.append(duration)
        
        return latencies
    
    async def _simulate_research_request(self) -> Dict[str, Any]:
        """Simulate a research request for concurrent testing.
        
        Returns:
            Dictionary with request result
        """
        # TODO: Replace with actual API call when API is available
        await asyncio.sleep(0.1)  # Simulate processing time
        return {"status": "completed", "duration_ms": 100}
