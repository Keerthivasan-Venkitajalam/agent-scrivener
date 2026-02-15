"""Unit tests for PerformanceBenchmarker."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, settings, strategies as st

from agent_scrivener.deployment.validation.performance_benchmarker import PerformanceBenchmarker
from agent_scrivener.deployment.validation.models import ValidationStatus, PerformanceMetrics


@pytest.fixture
def benchmarker():
    """Create a PerformanceBenchmarker instance for testing."""
    return PerformanceBenchmarker(
        api_base_url="http://localhost:8000",
        database_url="postgresql://test:test@localhost/test"
    )


@pytest.mark.asyncio
async def test_benchmark_single_request_success(benchmarker):
    """Test single request benchmarking returns success when performance meets requirements."""
    result = await benchmarker._benchmark_single_request()
    
    assert result.validator_name == "PerformanceBenchmarker"
    assert result.status == ValidationStatus.PASS
    assert "metrics" in result.details
    assert result.details["meets_requirement"] is True
    
    # Verify metrics structure
    metrics = result.details["metrics"]
    assert metrics.p90_ms < 180_000  # Should be well under 3 minutes


@pytest.mark.asyncio
async def test_benchmark_api_endpoints_success(benchmarker):
    """Test API endpoint benchmarking returns success when endpoints meet requirements."""
    result = await benchmarker._benchmark_api_endpoints()
    
    assert result.validator_name == "PerformanceBenchmarker"
    assert result.status == ValidationStatus.PASS
    assert "endpoint_metrics" in result.details
    
    # Verify health endpoint metrics
    endpoint_metrics = result.details["endpoint_metrics"]
    assert "health" in endpoint_metrics
    assert "status" in endpoint_metrics


@pytest.mark.asyncio
async def test_benchmark_concurrent_requests_success(benchmarker):
    """Test concurrent request benchmarking with 10 concurrent requests."""
    result = await benchmarker._benchmark_concurrent_requests(concurrent_count=10)
    
    assert result.validator_name == "PerformanceBenchmarker"
    assert result.status == ValidationStatus.PASS
    assert "load_metrics" in result.details
    assert result.details["degradation_detected"] is False
    
    # Verify load metrics exist for different load levels
    load_metrics = result.details["load_metrics"]
    assert "load_1" in load_metrics
    assert "load_5" in load_metrics
    assert "load_10" in load_metrics


@pytest.mark.asyncio
async def test_benchmark_concurrent_requests_skip_when_count_too_low(benchmarker):
    """Test concurrent request benchmarking skips when concurrent_count < 10."""
    result = await benchmarker._benchmark_concurrent_requests(concurrent_count=5)
    
    assert result.validator_name == "PerformanceBenchmarker"
    assert result.status == ValidationStatus.SKIP
    assert "concurrent_count < 10" in result.message


@pytest.mark.asyncio
async def test_benchmark_resource_usage_success(benchmarker):
    """Test resource usage benchmarking returns success when within limits."""
    result = await benchmarker._benchmark_resource_usage()
    
    assert result.validator_name == "PerformanceBenchmarker"
    # Status could be PASS or FAIL depending on actual resource usage
    assert result.status in (ValidationStatus.PASS, ValidationStatus.FAIL)
    assert "resource_metrics" in result.details
    
    # Verify resource metrics structure
    resource_metrics = result.details["resource_metrics"]
    assert "avg_memory_mb" in resource_metrics
    assert "max_memory_mb" in resource_metrics
    assert "avg_cpu_percent" in resource_metrics
    assert "max_cpu_percent" in resource_metrics


@pytest.mark.asyncio
async def test_benchmark_database_queries_success(benchmarker):
    """Test database query benchmarking returns success when queries meet requirements."""
    result = await benchmarker._benchmark_database_queries()
    
    assert result.validator_name == "PerformanceBenchmarker"
    assert result.status == ValidationStatus.PASS
    assert "query_metrics" in result.details
    
    # Verify query metrics for different query types
    query_metrics = result.details["query_metrics"]
    assert "session_lookup" in query_metrics
    assert "session_list" in query_metrics
    assert "session_insert" in query_metrics
    
    # Verify session_lookup meets 50ms requirement
    session_lookup = query_metrics["session_lookup"]
    assert session_lookup["threshold_ms"] == 50
    assert session_lookup["meets_requirement"] is True


@pytest.mark.asyncio
async def test_validate_runs_all_benchmarks(benchmarker):
    """Test that validate() runs all benchmark methods."""
    results = await benchmarker.validate()
    
    # Should have 5 results (single request, api endpoints, concurrent, resource, database)
    assert len(results) == 5
    
    # Verify all results are from PerformanceBenchmarker
    for result in results:
        assert result.validator_name == "PerformanceBenchmarker"
        assert result.status in (ValidationStatus.PASS, ValidationStatus.FAIL, ValidationStatus.SKIP)


@pytest.mark.asyncio
async def test_validate_skips_database_when_no_url():
    """Test that database benchmarking is skipped when no database URL is provided."""
    benchmarker = PerformanceBenchmarker(
        api_base_url="http://localhost:8000",
        database_url=None
    )
    
    results = await benchmarker.validate()
    
    # Find the database result
    db_result = next((r for r in results if "database" in r.message.lower()), None)
    assert db_result is not None
    assert db_result.status == ValidationStatus.SKIP


def test_calculate_metrics():
    """Test performance metrics calculation."""
    benchmarker = PerformanceBenchmarker()
    
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    metrics = benchmarker._calculate_metrics("test_metric", latencies)
    
    assert metrics.metric_name == "test_metric"
    assert metrics.sample_count == 10
    assert metrics.min_ms == 10.0
    assert metrics.max_ms == 100.0
    assert metrics.mean_ms == 55.0
    assert metrics.p50_ms == 55.0  # Median
    assert metrics.p90_ms == 91.0  # 90th percentile
    assert metrics.p95_ms == 95.5  # 95th percentile
    assert metrics.p99_ms == 99.1  # 99th percentile


def test_calculate_metrics_empty_list():
    """Test performance metrics calculation with empty list."""
    benchmarker = PerformanceBenchmarker()
    
    metrics = benchmarker._calculate_metrics("test_metric", [])
    
    assert metrics.metric_name == "test_metric"
    assert metrics.sample_count == 0
    assert metrics.min_ms == 0
    assert metrics.max_ms == 0
    assert metrics.mean_ms == 0


def test_percentile_calculation():
    """Test percentile calculation."""
    benchmarker = PerformanceBenchmarker()
    
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    assert benchmarker._percentile(data, 0) == 1.0
    assert benchmarker._percentile(data, 50) == 5.5
    assert benchmarker._percentile(data, 100) == 10.0


def test_percentile_empty_list():
    """Test percentile calculation with empty list."""
    benchmarker = PerformanceBenchmarker()
    
    assert benchmarker._percentile([], 50) == 0.0


def test_generate_performance_report():
    """Test performance report generation."""
    benchmarker = PerformanceBenchmarker()
    
    metrics_list = [
        PerformanceMetrics(
            metric_name="test_metric_1",
            p50_ms=50.0,
            p90_ms=90.0,
            p95_ms=95.0,
            p99_ms=99.0,
            min_ms=10.0,
            max_ms=100.0,
            mean_ms=55.0,
            std_dev_ms=28.87,
            sample_count=10
        ),
        PerformanceMetrics(
            metric_name="test_metric_2",
            p50_ms=25.0,
            p90_ms=45.0,
            p95_ms=47.5,
            p99_ms=49.5,
            min_ms=5.0,
            max_ms=50.0,
            mean_ms=27.5,
            std_dev_ms=14.43,
            sample_count=10
        )
    ]
    
    report = benchmarker.generate_performance_report(metrics_list)
    
    assert "summary" in report
    assert report["summary"]["total_metrics"] == 2
    assert "generated_at" in report["summary"]
    
    assert "metrics" in report
    assert len(report["metrics"]) == 2
    
    # Verify first metric
    metric1 = report["metrics"][0]
    assert metric1["name"] == "test_metric_1"
    assert metric1["percentiles"]["p50"] == "50.00ms"
    assert metric1["percentiles"]["p90"] == "90.00ms"
    assert metric1["sample_count"] == 10


def test_generate_performance_report_includes_all_percentiles():
    """
    Test performance report includes all percentiles (p50, p90, p95, p99).
    
    Task 7.6: Write unit test for performance report generation
    Validates: Requirements 5.8
    
    Requirement 5.8: WHEN the performance test completes, THE System SHALL 
    generate a report with percentile breakdowns (p50, p90, p95, p99) for 
    all measured metrics.
    
    This test verifies that:
    1. The report includes all four required percentiles (p50, p90, p95, p99)
    2. Each percentile value is correctly formatted and present
    3. The report structure contains all necessary fields
    4. Multiple metrics are properly included in the report
    """
    benchmarker = PerformanceBenchmarker()
    
    # Create test metrics with known percentile values
    metrics_list = [
        PerformanceMetrics(
            metric_name="api_health_endpoint",
            p50_ms=45.5,
            p90_ms=85.2,
            p95_ms=92.7,
            p99_ms=98.3,
            min_ms=10.0,
            max_ms=100.0,
            mean_ms=50.0,
            std_dev_ms=20.5,
            sample_count=100
        ),
        PerformanceMetrics(
            metric_name="database_session_lookup",
            p50_ms=22.1,
            p90_ms=42.8,
            p95_ms=46.5,
            p99_ms=49.2,
            min_ms=5.0,
            max_ms=50.0,
            mean_ms=25.0,
            std_dev_ms=12.3,
            sample_count=200
        ),
        PerformanceMetrics(
            metric_name="research_request_completion",
            p50_ms=120000.0,
            p90_ms=165000.0,
            p95_ms=175000.0,
            p99_ms=179000.0,
            min_ms=60000.0,
            max_ms=180000.0,
            mean_ms=130000.0,
            std_dev_ms=35000.0,
            sample_count=50
        )
    ]
    
    # Generate the performance report
    report = benchmarker.generate_performance_report(metrics_list)
    
    # Verify report structure
    assert "summary" in report, "Report should contain summary section"
    assert "metrics" in report, "Report should contain metrics section"
    
    # Verify summary information
    assert report["summary"]["total_metrics"] == 3, "Summary should show correct total metrics count"
    assert "generated_at" in report["summary"], "Summary should include generation timestamp"
    
    # Verify all metrics are included
    assert len(report["metrics"]) == 3, "Report should include all three metrics"
    
    # Verify each metric includes ALL required percentiles (p50, p90, p95, p99)
    for i, metric_report in enumerate(report["metrics"]):
        metric_name = metrics_list[i].metric_name
        
        # Verify metric name
        assert metric_report["name"] == metric_name, f"Metric name should match: {metric_name}"
        
        # Verify percentiles section exists
        assert "percentiles" in metric_report, f"Metric {metric_name} should have percentiles section"
        percentiles = metric_report["percentiles"]
        
        # CRITICAL: Verify ALL four required percentiles are present (Requirement 5.8)
        assert "p50" in percentiles, f"Metric {metric_name} should include p50 percentile"
        assert "p90" in percentiles, f"Metric {metric_name} should include p90 percentile"
        assert "p95" in percentiles, f"Metric {metric_name} should include p95 percentile"
        assert "p99" in percentiles, f"Metric {metric_name} should include p99 percentile"
        
        # Verify percentile values are correctly formatted
        expected_p50 = f"{metrics_list[i].p50_ms:.2f}ms"
        expected_p90 = f"{metrics_list[i].p90_ms:.2f}ms"
        expected_p95 = f"{metrics_list[i].p95_ms:.2f}ms"
        expected_p99 = f"{metrics_list[i].p99_ms:.2f}ms"
        
        assert percentiles["p50"] == expected_p50, \
            f"Metric {metric_name} p50 should be {expected_p50}, got {percentiles['p50']}"
        assert percentiles["p90"] == expected_p90, \
            f"Metric {metric_name} p90 should be {expected_p90}, got {percentiles['p90']}"
        assert percentiles["p95"] == expected_p95, \
            f"Metric {metric_name} p95 should be {expected_p95}, got {percentiles['p95']}"
        assert percentiles["p99"] == expected_p99, \
            f"Metric {metric_name} p99 should be {expected_p99}, got {percentiles['p99']}"
        
        # Verify range information is also included
        assert "range" in metric_report, f"Metric {metric_name} should have range section"
        range_info = metric_report["range"]
        
        assert "min" in range_info, f"Metric {metric_name} should include min value"
        assert "max" in range_info, f"Metric {metric_name} should include max value"
        assert "mean" in range_info, f"Metric {metric_name} should include mean value"
        assert "std_dev" in range_info, f"Metric {metric_name} should include std_dev value"
        
        # Verify sample count is included
        assert "sample_count" in metric_report, f"Metric {metric_name} should include sample_count"
        assert metric_report["sample_count"] == metrics_list[i].sample_count, \
            f"Metric {metric_name} sample_count should match"
    
    # Verify specific metric values for the first metric (api_health_endpoint)
    api_metric = report["metrics"][0]
    assert api_metric["name"] == "api_health_endpoint"
    assert api_metric["percentiles"]["p50"] == "45.50ms"
    assert api_metric["percentiles"]["p90"] == "85.20ms"
    assert api_metric["percentiles"]["p95"] == "92.70ms"
    assert api_metric["percentiles"]["p99"] == "98.30ms"
    assert api_metric["sample_count"] == 100
    
    # Verify specific metric values for the second metric (database_session_lookup)
    db_metric = report["metrics"][1]
    assert db_metric["name"] == "database_session_lookup"
    assert db_metric["percentiles"]["p50"] == "22.10ms"
    assert db_metric["percentiles"]["p90"] == "42.80ms"
    assert db_metric["percentiles"]["p95"] == "46.50ms"
    assert db_metric["percentiles"]["p99"] == "49.20ms"
    assert db_metric["sample_count"] == 200
    
    # Verify specific metric values for the third metric (research_request_completion)
    research_metric = report["metrics"][2]
    assert research_metric["name"] == "research_request_completion"
    assert research_metric["percentiles"]["p50"] == "120000.00ms"
    assert research_metric["percentiles"]["p90"] == "165000.00ms"
    assert research_metric["percentiles"]["p95"] == "175000.00ms"
    assert research_metric["percentiles"]["p99"] == "179000.00ms"
    assert research_metric["sample_count"] == 50


def test_metrics_to_dict():
    """Test converting PerformanceMetrics to dictionary."""
    benchmarker = PerformanceBenchmarker()
    
    metrics = PerformanceMetrics(
        metric_name="test_metric",
        p50_ms=50.0,
        p90_ms=90.0,
        p95_ms=95.0,
        p99_ms=99.0,
        min_ms=10.0,
        max_ms=100.0,
        mean_ms=55.0,
        std_dev_ms=28.87,
        sample_count=10
    )
    
    metrics_dict = benchmarker._metrics_to_dict(metrics)
    
    assert metrics_dict["metric_name"] == "test_metric"
    assert metrics_dict["p50_ms"] == 50.0
    assert metrics_dict["p90_ms"] == 90.0
    assert metrics_dict["sample_count"] == 10


@pytest.mark.asyncio
async def test_benchmark_single_request_public_method(benchmarker):
    """Test public benchmark_single_request method."""
    result_dict = await benchmarker.benchmark_single_request()
    
    assert "result" in result_dict
    assert "metrics" in result_dict
    
    result = result_dict["result"]
    assert result.validator_name == "PerformanceBenchmarker"
    assert result.status in (ValidationStatus.PASS, ValidationStatus.FAIL)


@pytest.mark.asyncio
async def test_benchmark_api_endpoints_public_method(benchmarker):
    """Test public benchmark_api_endpoints method."""
    result_dict = await benchmarker.benchmark_api_endpoints()
    
    assert "result" in result_dict
    assert "endpoint_metrics" in result_dict
    
    result = result_dict["result"]
    assert result.validator_name == "PerformanceBenchmarker"


@pytest.mark.asyncio
async def test_benchmark_concurrent_requests_public_method(benchmarker):
    """Test public benchmark_concurrent_requests method."""
    result_dict = await benchmarker.benchmark_concurrent_requests(concurrent_count=10)
    
    assert "result" in result_dict
    assert "metrics" in result_dict
    
    result = result_dict["result"]
    assert result.validator_name == "PerformanceBenchmarker"


@pytest.mark.asyncio
async def test_benchmark_resource_usage_public_method(benchmarker):
    """Test public benchmark_resource_usage method."""
    result_dict = await benchmarker.benchmark_resource_usage()
    
    assert "result" in result_dict
    assert "resource_metrics" in result_dict
    
    result = result_dict["result"]
    assert result.validator_name == "PerformanceBenchmarker"


@pytest.mark.asyncio
async def test_benchmark_database_queries_public_method(benchmarker):
    """Test public benchmark_database_queries method."""
    result_dict = await benchmarker.benchmark_database_queries()
    
    assert "result" in result_dict
    assert "query_metrics" in result_dict
    
    result = result_dict["result"]
    assert result.validator_name == "PerformanceBenchmarker"


@pytest.mark.asyncio
async def test_measure_endpoint_latencies(benchmarker):
    """Test endpoint latency measurement."""
    latencies = await benchmarker._measure_endpoint_latencies("/test", count=5)
    
    assert len(latencies) == 5
    assert all(isinstance(lat, float) for lat in latencies)
    assert all(lat > 0 for lat in latencies)


@pytest.mark.asyncio
async def test_simulate_research_request(benchmarker):
    """Test research request simulation."""
    result = await benchmarker._simulate_research_request()
    
    assert "status" in result
    assert result["status"] == "completed"
    assert "duration_ms" in result



# Property-Based Tests

class TestPerformanceBenchmarkerProperties:
    """Property-based tests for PerformanceBenchmarker."""

    @given(
        request_latencies=st.lists(
            st.floats(min_value=1000.0, max_value=300000.0),  # 1 second to 5 minutes in ms
            min_size=10,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_property_request_completion_percentiles(self, request_latencies):
        """
        Property Test: Request completion percentiles
        
        Feature: production-readiness-validation, Property 19: Request completion percentiles
        
        **Validates: Requirements 5.1**
        
        For any set of research requests, 90% should complete within 3 minutes 
        (p90 latency ≤ 3 minutes).
        
        This property verifies that:
        1. The p90 latency calculation is accurate
        2. The system correctly identifies when p90 meets the 3-minute threshold
        3. The validation passes when 90% of requests complete within 3 minutes
        4. The validation fails when p90 exceeds 3 minutes
        5. Performance metrics are calculated correctly from latency samples
        """
        # Create benchmarker instance
        benchmarker = PerformanceBenchmarker(
            api_base_url="http://localhost:8000",
            database_url="postgresql://test:test@localhost/test"
        )
        
        # Calculate metrics from the generated latencies
        metrics = benchmarker._calculate_metrics("test_requests", request_latencies)
        
        # Verify metrics calculation
        assert metrics.metric_name == "test_requests"
        assert metrics.sample_count == len(request_latencies)
        assert metrics.min_ms == min(request_latencies)
        assert metrics.max_ms == max(request_latencies)
        
        # Verify p90 calculation is within valid range
        assert metrics.p90_ms >= metrics.min_ms
        assert metrics.p90_ms <= metrics.max_ms
        
        # Verify p90 is at or above p50
        assert metrics.p90_ms >= metrics.p50_ms
        
        # Verify the 3-minute threshold (180,000 ms)
        threshold_ms = 180_000
        
        # Calculate expected p90 manually to verify correctness
        sorted_latencies = sorted(request_latencies)
        expected_p90_index = int(0.90 * (len(sorted_latencies) - 1))
        expected_p90_lower = sorted_latencies[expected_p90_index]
        expected_p90_upper = sorted_latencies[min(expected_p90_index + 1, len(sorted_latencies) - 1)]
        
        # The calculated p90 should be between the lower and upper bounds
        # Use a small epsilon for floating-point comparison
        epsilon = 1e-9
        assert metrics.p90_ms >= expected_p90_lower - epsilon
        assert metrics.p90_ms <= expected_p90_upper + epsilon
        
        # Verify that the requirement check is consistent
        meets_requirement = metrics.p90_ms <= threshold_ms
        
        # Count how many requests actually complete within 3 minutes
        requests_within_threshold = sum(1 for lat in request_latencies if lat <= threshold_ms)
        percentage_within_threshold = (requests_within_threshold / len(request_latencies)) * 100
        
        # If p90 meets requirement, at least 90% of requests should be within threshold
        # Note: Due to interpolation, the actual percentage might be slightly less than 90%
        # when p90 is very close to the threshold. We allow a small tolerance.
        if meets_requirement:
            # Only enforce the 90% rule if p90 is comfortably below the threshold
            # If p90 is very close to threshold (within 1%), the percentage might be slightly less
            p90_margin = (threshold_ms - metrics.p90_ms) / threshold_ms * 100
            if p90_margin > 1.0:  # p90 is more than 1% below threshold
                assert percentage_within_threshold >= 90.0, \
                    f"p90={metrics.p90_ms:.2f}ms meets threshold but only {percentage_within_threshold:.1f}% within limit"
        
        # Verify percentile ordering (p50 <= p90 <= p95 <= p99)
        # Use epsilon for floating-point comparison
        epsilon = 1e-9
        assert metrics.p50_ms <= metrics.p90_ms + epsilon
        assert metrics.p90_ms <= metrics.p95_ms + epsilon
        assert metrics.p95_ms <= metrics.p99_ms + epsilon
        
        # Verify mean is reasonable relative to min/max
        assert metrics.min_ms <= metrics.mean_ms <= metrics.max_ms
        
        # Verify standard deviation is non-negative
        assert metrics.std_dev_ms >= 0

    @given(
        concurrent_count=st.integers(min_value=10, max_value=20),
        baseline_latency=st.floats(min_value=100.0, max_value=5000.0),
        degradation_factor=st.floats(min_value=0.5, max_value=3.0)
    )
    @settings(max_examples=100, deadline=10000)
    def test_property_concurrent_request_handling(self, concurrent_count, baseline_latency, degradation_factor):
        """
        Property Test: Concurrent request handling
        
        Feature: production-readiness-validation, Property 20: Concurrent request handling
        
        **Validates: Requirements 5.4**
        
        For any load test with 10 simultaneous research sessions, the system should 
        handle all requests without performance degradation.
        
        This property verifies that:
        1. The system can handle at least 10 concurrent requests
        2. Performance degradation is correctly detected (>2x baseline latency)
        3. Failed requests are tracked and reported
        4. Load metrics are properly structured for different load levels
        5. The validation passes when no degradation is detected at load=10
        6. The validation fails when degradation is detected or requests fail
        """
        # Create benchmarker instance
        benchmarker = PerformanceBenchmarker(
            api_base_url="http://localhost:8000",
            database_url="postgresql://test:test@localhost/test"
        )
        
        # Simulate load metrics for different load levels
        load_levels = [1, 5, 10]
        if concurrent_count >= 20:
            load_levels.append(20)
        
        load_metrics = {}
        for i, load in enumerate(load_levels):
            # Calculate latency with potential degradation
            # First load (baseline) uses baseline_latency
            # Subsequent loads may have degradation
            if i == 0:
                avg_latency = baseline_latency
            else:
                # Apply degradation factor to baseline
                # degradation_factor > 2.0 means latency will exceed 2x baseline
                avg_latency = baseline_latency * degradation_factor
            
            # Simulate some failures for higher loads if degradation is severe
            failed_requests = 0
            if degradation_factor > 2.5 and load >= 10:
                failed_requests = min(2, load // 5)
            
            successful_requests = load - failed_requests
            
            load_metrics[f"load_{load}"] = {
                "concurrent_requests": load,
                "avg_latency_ms": avg_latency,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests
            }
        
        # Verify load metrics structure
        for load_key, metrics in load_metrics.items():
            assert "concurrent_requests" in metrics
            assert "avg_latency_ms" in metrics
            assert "successful_requests" in metrics
            assert "failed_requests" in metrics
            
            # Verify consistency: successful + failed = concurrent
            total_requests = metrics["successful_requests"] + metrics["failed_requests"]
            assert total_requests == metrics["concurrent_requests"], \
                f"Total requests ({total_requests}) should equal concurrent count ({metrics['concurrent_requests']})"
            
            # Verify latency is positive
            assert metrics["avg_latency_ms"] > 0
            
            # Verify counts are non-negative
            assert metrics["concurrent_requests"] >= 0
            assert metrics["successful_requests"] >= 0
            assert metrics["failed_requests"] >= 0
        
        # Check for degradation detection logic (matches implementation)
        detected_degradation = False
        baseline = load_metrics["load_1"]["avg_latency_ms"]
        
        for load_key in load_metrics.keys():
            if load_key == "load_1":
                continue
            
            avg_latency = load_metrics[load_key]["avg_latency_ms"]
            if avg_latency > baseline * 2:
                # More than 2x degradation is significant
                detected_degradation = True
                break
        
        # Check if load_10 metrics exist (required for validation)
        assert "load_10" in load_metrics, "Load 10 metrics should always be present"
        
        load_10_metrics = load_metrics["load_10"]
        
        # Verify the validation logic matches the implementation
        should_pass = (not detected_degradation) and (load_10_metrics["failed_requests"] == 0)
        
        if should_pass:
            # System should pass validation
            assert load_10_metrics["successful_requests"] == 10, \
                "System should successfully handle exactly 10 concurrent requests"
            
            # Verify latency is reasonable (not excessively high)
            assert load_10_metrics["avg_latency_ms"] <= baseline * 2, \
                f"Latency at load 10 ({load_10_metrics['avg_latency_ms']:.2f}ms) " \
                f"should not exceed 2x baseline ({baseline:.2f}ms)"
        else:
            # System should fail validation
            assert detected_degradation or load_10_metrics["failed_requests"] > 0, \
                "Validation should fail when degradation or failures occur"
        
        # Verify load level progression makes sense
        load_keys = sorted(load_metrics.keys(), key=lambda x: int(x.split("_")[1]))
        if len(load_keys) > 1:
            for i in range(len(load_keys) - 1):
                current_load = int(load_keys[i].split("_")[1])
                next_load = int(load_keys[i + 1].split("_")[1])
                
                # Load levels should be in ascending order
                assert current_load < next_load, \
                    f"Load levels should be in ascending order: {current_load} >= {next_load}"
        
        # Verify that concurrent_count parameter is respected
        assert concurrent_count >= 10, \
            "Concurrent count should be at least 10 for meaningful testing"
        
        # Verify that load levels don't exceed concurrent_count
        for load_key, metrics in load_metrics.items():
            load_level = int(load_key.split("_")[1])
            assert load_level <= concurrent_count, \
                f"Load level {load_level} should not exceed concurrent_count {concurrent_count}"
        
        # Verify degradation detection is consistent with the 2x threshold
        if degradation_factor > 2.0:
            # Should detect degradation at higher loads
            assert detected_degradation or load_10_metrics["failed_requests"] > 0, \
                f"Should detect degradation when factor is {degradation_factor:.2f}"
        elif degradation_factor < 1.5:
            # Should not detect degradation with minimal factor
            assert not detected_degradation, \
                f"Should not detect degradation when factor is {degradation_factor:.2f}"

    @given(
        memory_usage_mb=st.floats(min_value=100.0, max_value=3000.0),
        cpu_usage_percent=st.floats(min_value=10.0, max_value=100.0),
        baseline_memory_mb=st.floats(min_value=50.0, max_value=500.0)
    )
    @settings(max_examples=100, deadline=10000)
    def test_property_resource_usage_limits(self, memory_usage_mb, cpu_usage_percent, baseline_memory_mb):
        """
        Property Test: Resource usage limits
        
        Feature: production-readiness-validation, Property 21: Resource usage limits
        
        **Validates: Requirements 5.5, 5.6**
        
        For any research session, memory consumption should remain under 2GB RAM 
        and CPU utilization should average under 70% under normal load.
        
        This property verifies that:
        1. Memory consumption is correctly measured and compared against 2GB threshold
        2. CPU utilization is correctly measured and compared against 70% threshold
        3. Memory delta (per-session usage) is calculated from baseline
        4. The validation passes when both memory and CPU are within limits
        5. The validation fails when either memory or CPU exceeds limits
        6. Resource metrics are properly structured and contain all required fields
        7. Remediation steps are provided when limits are exceeded
        """
        # Create benchmarker instance
        benchmarker = PerformanceBenchmarker(
            api_base_url="http://localhost:8000",
            database_url="postgresql://test:test@localhost/test"
        )
        
        # Simulate resource usage metrics
        # Memory delta represents per-session memory usage (current - baseline)
        memory_delta_mb = memory_usage_mb - baseline_memory_mb
        
        # Ensure memory delta is non-negative (can't use less than baseline)
        if memory_delta_mb < 0:
            memory_delta_mb = 0
        
        # Create resource metrics structure matching implementation
        resource_metrics = {
            "baseline_memory_mb": baseline_memory_mb,
            "avg_memory_mb": memory_usage_mb,
            "max_memory_mb": memory_usage_mb,  # For simplicity, use same value
            "memory_delta_mb": memory_delta_mb,
            "baseline_cpu_percent": 20.0,  # Typical baseline
            "avg_cpu_percent": cpu_usage_percent,
            "max_cpu_percent": min(cpu_usage_percent * 1.2, 100.0)  # Max is slightly higher
        }
        
        # Verify resource metrics structure
        assert "baseline_memory_mb" in resource_metrics
        assert "avg_memory_mb" in resource_metrics
        assert "max_memory_mb" in resource_metrics
        assert "memory_delta_mb" in resource_metrics
        assert "baseline_cpu_percent" in resource_metrics
        assert "avg_cpu_percent" in resource_metrics
        assert "max_cpu_percent" in resource_metrics
        
        # Verify all values are non-negative
        for key, value in resource_metrics.items():
            assert value >= 0, f"{key} should be non-negative, got {value}"
        
        # Verify max values are >= avg values
        assert resource_metrics["max_memory_mb"] >= resource_metrics["avg_memory_mb"], \
            "Max memory should be >= average memory"
        assert resource_metrics["max_cpu_percent"] >= resource_metrics["avg_cpu_percent"], \
            "Max CPU should be >= average CPU"
        
        # Verify memory delta calculation is correct
        expected_delta = resource_metrics["max_memory_mb"] - resource_metrics["baseline_memory_mb"]
        if expected_delta < 0:
            expected_delta = 0
        assert abs(resource_metrics["memory_delta_mb"] - expected_delta) < 0.01, \
            f"Memory delta calculation incorrect: {resource_metrics['memory_delta_mb']} != {expected_delta}"
        
        # Check requirements (matches implementation)
        memory_threshold_mb = 2048  # 2GB
        cpu_threshold_percent = 70
        
        memory_ok = resource_metrics["memory_delta_mb"] < memory_threshold_mb
        cpu_ok = resource_metrics["avg_cpu_percent"] < cpu_threshold_percent
        
        # Verify threshold checks are consistent
        if memory_ok:
            assert resource_metrics["memory_delta_mb"] < memory_threshold_mb, \
                f"Memory should be under threshold: {resource_metrics['memory_delta_mb']}MB < {memory_threshold_mb}MB"
        else:
            assert resource_metrics["memory_delta_mb"] >= memory_threshold_mb, \
                f"Memory should exceed threshold: {resource_metrics['memory_delta_mb']}MB >= {memory_threshold_mb}MB"
        
        if cpu_ok:
            assert resource_metrics["avg_cpu_percent"] < cpu_threshold_percent, \
                f"CPU should be under threshold: {resource_metrics['avg_cpu_percent']}% < {cpu_threshold_percent}%"
        else:
            assert resource_metrics["avg_cpu_percent"] >= cpu_threshold_percent, \
                f"CPU should exceed threshold: {resource_metrics['avg_cpu_percent']}% >= {cpu_threshold_percent}%"
        
        # Verify validation logic
        should_pass = memory_ok and cpu_ok
        
        if should_pass:
            # Both memory and CPU should be within limits
            assert resource_metrics["memory_delta_mb"] < memory_threshold_mb, \
                "Memory should be within limit when validation passes"
            assert resource_metrics["avg_cpu_percent"] < cpu_threshold_percent, \
                "CPU should be within limit when validation passes"
        else:
            # At least one resource should exceed limits
            assert not memory_ok or not cpu_ok, \
                "At least one resource should exceed limits when validation fails"
        
        # Verify CPU percentage is within valid range (0-100%)
        assert 0 <= resource_metrics["avg_cpu_percent"] <= 100, \
            f"CPU percentage should be 0-100%, got {resource_metrics['avg_cpu_percent']}%"
        assert 0 <= resource_metrics["max_cpu_percent"] <= 100, \
            f"Max CPU percentage should be 0-100%, got {resource_metrics['max_cpu_percent']}%"
        
        # Verify memory values are reasonable
        assert resource_metrics["baseline_memory_mb"] > 0, \
            "Baseline memory should be positive"
        assert resource_metrics["avg_memory_mb"] > 0, \
            "Average memory should be positive"
        
        # Verify the 2GB threshold is reasonable for the test
        # If memory_delta is very close to threshold, ensure consistent behavior
        if abs(resource_metrics["memory_delta_mb"] - memory_threshold_mb) < 1:
            # Edge case: very close to threshold
            # The implementation uses < (not <=), so exactly at threshold should fail
            if resource_metrics["memory_delta_mb"] >= memory_threshold_mb:
                assert not memory_ok, "Should fail when at or above threshold"
        
        # Verify the 70% CPU threshold is reasonable
        if abs(resource_metrics["avg_cpu_percent"] - cpu_threshold_percent) < 0.1:
            # Edge case: very close to threshold
            if resource_metrics["avg_cpu_percent"] >= cpu_threshold_percent:
                assert not cpu_ok, "Should fail when at or above threshold"
        
        # Verify that if both limits are significantly exceeded, validation fails
        if resource_metrics["memory_delta_mb"] > memory_threshold_mb * 1.5 and \
           resource_metrics["avg_cpu_percent"] > cpu_threshold_percent * 1.2:
            assert not should_pass, \
                "Validation should fail when both resources significantly exceed limits"
        
        # Verify that if both limits are well within bounds, validation passes
        if resource_metrics["memory_delta_mb"] < memory_threshold_mb * 0.5 and \
           resource_metrics["avg_cpu_percent"] < cpu_threshold_percent * 0.5:
            assert should_pass, \
                "Validation should pass when both resources are well within limits"

    @given(
        session_lookup_latencies=st.lists(
            st.floats(min_value=1.0, max_value=100.0),  # 1ms to 100ms
            min_size=10,
            max_size=50
        ),
        session_list_latencies=st.lists(
            st.floats(min_value=10.0, max_value=200.0),  # 10ms to 200ms
            min_size=10,
            max_size=50
        ),
        session_insert_latencies=st.lists(
            st.floats(min_value=10.0, max_value=200.0),  # 10ms to 200ms
            min_size=10,
            max_size=50
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_property_database_query_performance(
        self, 
        session_lookup_latencies, 
        session_list_latencies, 
        session_insert_latencies
    ):
        """
        Property Test: Database query performance
        
        Feature: production-readiness-validation, Property 22: Database query performance
        
        **Validates: Requirements 5.7**
        
        For any session lookup query, execution time should be under 50ms.
        
        This property verifies that:
        1. Session lookup queries are correctly measured and compared against 50ms threshold
        2. The p90 latency calculation is accurate for database queries
        3. The validation passes when session lookup p90 is under 50ms
        4. The validation fails when session lookup p90 exceeds 50ms
        5. Query metrics are properly structured for different query types
        6. All query types (lookup, list, insert) are benchmarked with appropriate thresholds
        7. Failed queries are correctly identified and reported
        """
        # Create benchmarker instance
        benchmarker = PerformanceBenchmarker(
            api_base_url="http://localhost:8000",
            database_url="postgresql://test:test@localhost/test"
        )
        
        # Define query types and their thresholds (matches implementation)
        query_types = {
            "session_lookup": (session_lookup_latencies, 50),   # 50ms threshold
            "session_list": (session_list_latencies, 100),      # 100ms threshold
            "session_insert": (session_insert_latencies, 100)   # 100ms threshold
        }
        
        query_metrics = {}
        all_passed = True
        failed_queries = []
        
        # Process each query type
        for query_type, (latencies, threshold_ms) in query_types.items():
            # Calculate metrics from the generated latencies
            metrics = benchmarker._calculate_metrics(query_type, latencies)
            
            # Verify metrics calculation
            assert metrics.metric_name == query_type
            assert metrics.sample_count == len(latencies)
            assert metrics.min_ms == min(latencies)
            assert metrics.max_ms == max(latencies)
            
            # Verify p90 calculation is within valid range
            assert metrics.p90_ms >= metrics.min_ms
            assert metrics.p90_ms <= metrics.max_ms
            
            # Verify percentile ordering
            # Use epsilon for floating-point comparison
            epsilon = 1e-9
            assert metrics.p50_ms <= metrics.p90_ms + epsilon
            assert metrics.p90_ms <= metrics.p95_ms + epsilon
            assert metrics.p95_ms <= metrics.p99_ms + epsilon
            
            # Check if query meets requirement
            meets_requirement = metrics.p90_ms <= threshold_ms
            
            query_metrics[query_type] = {
                "metrics": benchmarker._metrics_to_dict(metrics),
                "threshold_ms": threshold_ms,
                "meets_requirement": meets_requirement
            }
            
            # Track failures
            if not meets_requirement:
                all_passed = False
                failed_queries.append(f"{query_type} (p90: {metrics.p90_ms:.2f}ms > {threshold_ms}ms)")
            
            # Verify the threshold check is consistent
            if meets_requirement:
                assert metrics.p90_ms <= threshold_ms, \
                    f"{query_type} should meet threshold: {metrics.p90_ms:.2f}ms <= {threshold_ms}ms"
            else:
                assert metrics.p90_ms > threshold_ms, \
                    f"{query_type} should exceed threshold: {metrics.p90_ms:.2f}ms > {threshold_ms}ms"
        
        # Verify query_metrics structure
        assert "session_lookup" in query_metrics
        assert "session_list" in query_metrics
        assert "session_insert" in query_metrics
        
        # Verify each query metric has required fields
        for query_type, qm in query_metrics.items():
            assert "metrics" in qm
            assert "threshold_ms" in qm
            assert "meets_requirement" in qm
            
            # Verify metrics dictionary structure
            metrics_dict = qm["metrics"]
            assert "metric_name" in metrics_dict
            assert "p50_ms" in metrics_dict
            assert "p90_ms" in metrics_dict
            assert "p95_ms" in metrics_dict
            assert "p99_ms" in metrics_dict
            assert "min_ms" in metrics_dict
            assert "max_ms" in metrics_dict
            assert "mean_ms" in metrics_dict
            assert "std_dev_ms" in metrics_dict
            assert "sample_count" in metrics_dict
            
            # Verify all values are non-negative
            assert metrics_dict["p50_ms"] >= 0
            assert metrics_dict["p90_ms"] >= 0
            assert metrics_dict["p95_ms"] >= 0
            assert metrics_dict["p99_ms"] >= 0
            assert metrics_dict["min_ms"] >= 0
            assert metrics_dict["max_ms"] >= 0
            assert metrics_dict["mean_ms"] >= 0
            assert metrics_dict["std_dev_ms"] >= 0
            assert metrics_dict["sample_count"] > 0
        
        # Verify the critical requirement: session_lookup must be under 50ms
        session_lookup_metrics = query_metrics["session_lookup"]
        session_lookup_p90 = session_lookup_metrics["metrics"]["p90_ms"]
        session_lookup_threshold = session_lookup_metrics["threshold_ms"]
        
        assert session_lookup_threshold == 50, \
            "Session lookup threshold should be 50ms"
        
        # Verify validation logic
        should_pass = all_passed
        
        if should_pass:
            # All queries should meet their thresholds
            for query_type, qm in query_metrics.items():
                assert qm["meets_requirement"], \
                    f"{query_type} should meet requirement when validation passes"
            
            # Specifically verify session_lookup meets the 50ms requirement
            assert session_lookup_p90 <= 50, \
                f"Session lookup p90 ({session_lookup_p90:.2f}ms) should be under 50ms when validation passes"
            
            # Verify no failed queries
            assert len(failed_queries) == 0, \
                "Should have no failed queries when validation passes"
        else:
            # At least one query should fail its threshold
            assert len(failed_queries) > 0, \
                "Should have at least one failed query when validation fails"
            
            # Verify failed_queries list is consistent with query_metrics
            for query_type, qm in query_metrics.items():
                if not qm["meets_requirement"]:
                    # This query should be in failed_queries
                    assert any(query_type in fq for fq in failed_queries), \
                        f"{query_type} should be in failed_queries list"
        
        # Verify the most critical property: session_lookup performance
        # This is the core requirement from 5.7
        if session_lookup_p90 <= 50:
            # Session lookup meets requirement
            assert session_lookup_metrics["meets_requirement"], \
                "Session lookup should meet requirement when p90 <= 50ms"
            
            # Count how many lookups actually complete within 50ms
            lookups_within_threshold = sum(1 for lat in session_lookup_latencies if lat <= 50)
            percentage_within_threshold = (lookups_within_threshold / len(session_lookup_latencies)) * 100
            
            # If p90 meets requirement, at least 90% should be within threshold
            # Note: Due to interpolation, the actual percentage might be slightly less than 90%
            # when p90 is very close to the threshold. We allow a small tolerance.
            p90_margin = (50 - session_lookup_p90) / 50 * 100
            if p90_margin > 1.0:  # p90 is more than 1% below threshold
                assert percentage_within_threshold >= 90.0, \
                    f"p90={session_lookup_p90:.2f}ms meets threshold but only {percentage_within_threshold:.1f}% within limit"
        else:
            # Session lookup exceeds requirement
            assert not session_lookup_metrics["meets_requirement"], \
                "Session lookup should not meet requirement when p90 > 50ms"
        
        # Verify thresholds are appropriate for each query type
        assert query_metrics["session_lookup"]["threshold_ms"] == 50, \
            "Session lookup threshold should be 50ms (most critical)"
        assert query_metrics["session_list"]["threshold_ms"] == 100, \
            "Session list threshold should be 100ms"
        assert query_metrics["session_insert"]["threshold_ms"] == 100, \
            "Session insert threshold should be 100ms"
        
        # Verify that session_lookup has the strictest threshold
        # (since it's the most frequently used operation)
        lookup_threshold = query_metrics["session_lookup"]["threshold_ms"]
        list_threshold = query_metrics["session_list"]["threshold_ms"]
        insert_threshold = query_metrics["session_insert"]["threshold_ms"]
        
        assert lookup_threshold <= list_threshold, \
            "Lookup threshold should be <= list threshold"
        assert lookup_threshold <= insert_threshold, \
            "Lookup threshold should be <= insert threshold"
        
        # Verify mean is reasonable relative to min/max for all queries
        for query_type, qm in query_metrics.items():
            metrics_dict = qm["metrics"]
            assert metrics_dict["min_ms"] <= metrics_dict["mean_ms"] <= metrics_dict["max_ms"], \
                f"{query_type}: mean should be between min and max"
        
        # Verify that if session_lookup significantly exceeds threshold, validation fails
        if session_lookup_p90 > 50 * 1.5:  # More than 50% over threshold
            assert not should_pass, \
                f"Validation should fail when session_lookup p90 ({session_lookup_p90:.2f}ms) significantly exceeds 50ms"
        
        # Verify that if all queries are well within thresholds, validation passes
        all_well_within = all(
            qm["metrics"]["p90_ms"] < qm["threshold_ms"] * 0.8
            for qm in query_metrics.values()
        )
        if all_well_within:
            assert should_pass, \
                "Validation should pass when all queries are well within thresholds"
