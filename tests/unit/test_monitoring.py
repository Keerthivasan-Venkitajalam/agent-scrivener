"""
Unit tests for the system monitoring and health checks.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from agent_scrivener.utils.monitoring import (
    HealthChecker, MetricsCollector, AlertManager, SystemMonitor,
    HealthStatus, MetricType, HealthCheckResult, Metric, Alert,
    timed, counted, system_monitor
)
from agent_scrivener.models.errors import ErrorSeverity


class TestHealthChecker:
    """Test health check functionality."""
    
    @pytest.fixture
    def health_checker(self):
        return HealthChecker()
    
    @pytest.mark.asyncio
    async def test_register_and_run_check(self, health_checker):
        async def sample_check():
            return HealthCheckResult(
                name="test_check",
                status=HealthStatus.HEALTHY,
                message="All good"
            )
        
        health_checker.register_check("test_check", sample_check)
        result = await health_checker.run_check("test_check")
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.response_time_ms is not None
        assert result.response_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_run_nonexistent_check(self, health_checker):
        result = await health_checker.run_check("nonexistent")
        
        assert result.name == "nonexistent"
        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.message
    
    @pytest.mark.asyncio
    async def test_run_failing_check(self, health_checker):
        async def failing_check():
            raise ValueError("Test error")
        
        health_checker.register_check("failing_check", failing_check)
        result = await health_checker.run_check("failing_check")
        
        assert result.name == "failing_check"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Test error" in result.message
        assert result.response_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_run_all_checks(self, health_checker):
        async def healthy_check():
            return HealthCheckResult("healthy", HealthStatus.HEALTHY, "OK")
        
        async def degraded_check():
            return HealthCheckResult("degraded", HealthStatus.DEGRADED, "Warning")
        
        health_checker.register_check("healthy", healthy_check)
        health_checker.register_check("degraded", degraded_check)
        
        results = await health_checker.run_all_checks()
        
        assert len(results) == 2
        assert results["healthy"].status == HealthStatus.HEALTHY
        assert results["degraded"].status == HealthStatus.DEGRADED
    
    def test_overall_status_calculation(self, health_checker):
        # All healthy
        results = {
            "check1": HealthCheckResult("check1", HealthStatus.HEALTHY, "OK"),
            "check2": HealthCheckResult("check2", HealthStatus.HEALTHY, "OK")
        }
        assert health_checker.get_overall_status(results) == HealthStatus.HEALTHY
        
        # One degraded
        results["check2"].status = HealthStatus.DEGRADED
        assert health_checker.get_overall_status(results) == HealthStatus.DEGRADED
        
        # One unhealthy
        results["check2"].status = HealthStatus.UNHEALTHY
        assert health_checker.get_overall_status(results) == HealthStatus.UNHEALTHY
        
        # Empty results
        assert health_checker.get_overall_status({}) == HealthStatus.UNKNOWN


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector()
    
    def test_counter_operations(self, metrics_collector):
        # Test increment
        metrics_collector.increment_counter("test_counter", 5)
        assert metrics_collector.get_counter("test_counter") == 5
        
        # Test multiple increments
        metrics_collector.increment_counter("test_counter", 3)
        assert metrics_collector.get_counter("test_counter") == 8
        
        # Test default increment
        metrics_collector.increment_counter("test_counter")
        assert metrics_collector.get_counter("test_counter") == 9
        
        # Test nonexistent counter
        assert metrics_collector.get_counter("nonexistent") == 0
    
    def test_gauge_operations(self, metrics_collector):
        # Test set gauge
        metrics_collector.set_gauge("test_gauge", 42.5)
        assert metrics_collector.get_gauge("test_gauge") == 42.5
        
        # Test overwrite gauge
        metrics_collector.set_gauge("test_gauge", 100.0)
        assert metrics_collector.get_gauge("test_gauge") == 100.0
        
        # Test nonexistent gauge
        assert metrics_collector.get_gauge("nonexistent") is None
    
    def test_histogram_operations(self, metrics_collector):
        # Record some values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            metrics_collector.record_histogram("test_histogram", value)
        
        stats = metrics_collector.get_histogram_stats("test_histogram")
        
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["p50"] == 3.0
        
        # Test empty histogram
        empty_stats = metrics_collector.get_histogram_stats("empty")
        assert empty_stats == {"count": 0}
    
    def test_timer_operations(self, metrics_collector):
        # Record some timer values
        durations = [10.0, 20.0, 30.0, 40.0, 50.0]
        for duration in durations:
            metrics_collector.record_timer("test_timer", duration)
        
        stats = metrics_collector.get_timer_stats("test_timer")
        
        assert stats["count"] == 5
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0
    
    def test_get_all_metrics(self, metrics_collector):
        # Set up various metrics
        metrics_collector.increment_counter("counter1", 10)
        metrics_collector.set_gauge("gauge1", 42.0)
        metrics_collector.record_histogram("histogram1", 5.0)
        metrics_collector.record_timer("timer1", 100.0)
        
        all_metrics = metrics_collector.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert "timers" in all_metrics
        
        assert all_metrics["counters"]["counter1"] == 10
        assert all_metrics["gauges"]["gauge1"] == 42.0
        assert all_metrics["histograms"]["histogram1"]["count"] == 1
        assert all_metrics["timers"]["timer1"]["count"] == 1
    
    def test_reset_metrics(self, metrics_collector):
        # Set up some metrics
        metrics_collector.increment_counter("counter1", 10)
        metrics_collector.set_gauge("gauge1", 42.0)
        
        # Reset
        metrics_collector.reset_metrics()
        
        # Check everything is cleared
        assert metrics_collector.get_counter("counter1") == 0
        assert metrics_collector.get_gauge("gauge1") is None
        
        all_metrics = metrics_collector.get_all_metrics()
        assert len(all_metrics["counters"]) == 0
        assert len(all_metrics["gauges"]) == 0
    
    def test_metrics_with_tags(self, metrics_collector):
        tags = {"service": "test", "version": "1.0"}
        
        metrics_collector.increment_counter("tagged_counter", 1, tags)
        metrics_collector.set_gauge("tagged_gauge", 50.0, tags)
        
        # Verify metrics are recorded (tags are stored in history)
        assert metrics_collector.get_counter("tagged_counter") == 1
        assert metrics_collector.get_gauge("tagged_gauge") == 50.0


class TestAlertManager:
    """Test alert management functionality."""
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector()
    
    @pytest.fixture
    def alert_manager(self, metrics_collector):
        return AlertManager(metrics_collector)
    
    def test_set_threshold(self, alert_manager):
        alert_manager.set_threshold("test_metric", 100.0, ErrorSeverity.HIGH, "greater_than")
        
        assert "test_metric" in alert_manager.thresholds
        config = alert_manager.thresholds["test_metric"]
        assert config["threshold"] == 100.0
        assert config["severity"] == ErrorSeverity.HIGH
        assert config["comparison"] == "greater_than"
    
    @pytest.mark.asyncio
    async def test_threshold_checking_counter(self, alert_manager, metrics_collector):
        # Set up threshold
        alert_manager.set_threshold("test_counter", 50.0, ErrorSeverity.MEDIUM, "greater_than")
        
        # Set up alert handler
        triggered_alerts = []
        
        async def test_handler(alert: Alert):
            triggered_alerts.append(alert)
        
        alert_manager.add_alert_handler(test_handler)
        
        # Increment counter below threshold
        metrics_collector.increment_counter("test_counter", 30)
        await alert_manager.check_thresholds()
        assert len(triggered_alerts) == 0
        
        # Increment counter above threshold
        metrics_collector.increment_counter("test_counter", 30)  # Total: 60
        await alert_manager.check_thresholds()
        assert len(triggered_alerts) == 1
        
        alert = triggered_alerts[0]
        assert alert.metric_name == "test_counter"
        assert alert.current_value == 60.0
        assert alert.threshold == 50.0
        assert not alert.resolved
    
    @pytest.mark.asyncio
    async def test_threshold_checking_gauge(self, alert_manager, metrics_collector):
        # Set up threshold
        alert_manager.set_threshold("test_gauge", 75.0, ErrorSeverity.HIGH, "greater_than")
        
        triggered_alerts = []
        
        async def test_handler(alert: Alert):
            triggered_alerts.append(alert)
        
        alert_manager.add_alert_handler(test_handler)
        
        # Set gauge above threshold
        metrics_collector.set_gauge("test_gauge", 80.0)
        await alert_manager.check_thresholds()
        
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].current_value == 80.0
    
    @pytest.mark.asyncio
    async def test_alert_resolution(self, alert_manager, metrics_collector):
        # Set up threshold
        alert_manager.set_threshold("test_gauge", 50.0, ErrorSeverity.MEDIUM, "greater_than")
        
        # Trigger alert
        metrics_collector.set_gauge("test_gauge", 60.0)
        await alert_manager.check_thresholds()
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        
        # Resolve alert
        metrics_collector.set_gauge("test_gauge", 40.0)
        await alert_manager.check_thresholds()
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
        
        # Check that alert is marked as resolved
        all_alerts = alert_manager.get_all_alerts()
        assert len(all_alerts) == 1
        assert all_alerts[0].resolved
        assert all_alerts[0].resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_different_comparison_types(self, alert_manager, metrics_collector):
        # Test less_than comparison
        alert_manager.set_threshold("low_metric", 10.0, ErrorSeverity.MEDIUM, "less_than")
        
        triggered_alerts = []
        
        async def test_handler(alert: Alert):
            triggered_alerts.append(alert)
        
        alert_manager.add_alert_handler(test_handler)
        
        # Set value below threshold
        metrics_collector.set_gauge("low_metric", 5.0)
        await alert_manager.check_thresholds()
        
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].current_value == 5.0
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, alert_manager):
        # Test start monitoring
        alert_manager.start_monitoring(0.1)  # Very short interval for testing
        assert alert_manager._monitoring
        assert alert_manager._monitor_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.05)
        
        # Test stop monitoring
        alert_manager.stop_monitoring()
        assert not alert_manager._monitoring


class TestSystemMonitor:
    """Test system monitor integration."""
    
    @pytest.fixture
    def monitor(self):
        return SystemMonitor()
    
    @pytest.mark.asyncio
    async def test_system_status(self, monitor):
        status = await monitor.get_system_status()
        
        assert "overall_status" in status
        assert "timestamp" in status
        assert "health_checks" in status
        assert "metrics" in status
        assert "active_alerts" in status
        
        # Should have default health checks
        assert len(status["health_checks"]) >= 2
        assert "system_resources" in status["health_checks"]
        assert "memory_usage" in status["health_checks"]
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitor):
        # Test start
        monitor.start_monitoring(0.1)
        assert monitor.alert_manager._monitoring
        
        # Let it run briefly
        await asyncio.sleep(0.05)
        
        # Test stop
        monitor.stop_monitoring()
        assert not monitor.alert_manager._monitoring
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_system_resources_health_check(self, mock_disk, mock_memory, mock_cpu, monitor):
        # Mock system resources
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(percent=70.0)
        
        result = await monitor.health_checker.run_check("system_resources")
        
        assert result.status == HealthStatus.HEALTHY
        assert "50.0%" in result.message
        assert result.details["cpu_percent"] == 50.0
        assert result.details["memory_percent"] == 60.0
        assert result.details["disk_percent"] == 70.0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_system_resources_unhealthy(self, mock_disk, mock_memory, mock_cpu, monitor):
        # Mock high resource usage
        mock_cpu.return_value = 95.0
        mock_memory.return_value = MagicMock(percent=92.0)
        mock_disk.return_value = MagicMock(percent=85.0)
        
        result = await monitor.health_checker.run_check("system_resources")
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "High resource usage" in result.message


class TestDecorators:
    """Test monitoring decorators."""
    
    def test_timed_decorator_sync(self):
        @timed("test_function_timer")
        def test_function():
            time.sleep(0.01)  # Small delay
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # Check that timer was recorded
        stats = system_monitor.metrics_collector.get_timer_stats("test_function_timer")
        assert stats["count"] == 1
        assert stats["mean"] > 0  # Should have some duration
    
    @pytest.mark.asyncio
    async def test_timed_decorator_async(self):
        @timed("test_async_timer")
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await test_async_function()
        assert result == "async_result"
        
        # Check that timer was recorded
        stats = system_monitor.metrics_collector.get_timer_stats("test_async_timer")
        assert stats["count"] == 1
        assert stats["mean"] > 0
    
    def test_counted_decorator_sync(self):
        @counted("test_function_counter")
        def test_function():
            return "result"
        
        # Call function multiple times
        for _ in range(3):
            test_function()
        
        # Check counter
        count = system_monitor.metrics_collector.get_counter("test_function_counter")
        assert count == 3
    
    @pytest.mark.asyncio
    async def test_counted_decorator_async(self):
        @counted("test_async_counter")
        async def test_async_function():
            return "async_result"
        
        # Call function multiple times
        for _ in range(5):
            await test_async_function()
        
        # Check counter
        count = system_monitor.metrics_collector.get_counter("test_async_counter")
        assert count == 5
    
    def test_decorators_with_tags(self):
        @timed("tagged_timer", {"service": "test"})
        @counted("tagged_counter", {"service": "test"})
        def test_function():
            return "result"
        
        test_function()
        
        # Verify metrics were recorded
        assert system_monitor.metrics_collector.get_counter("tagged_counter") == 1
        stats = system_monitor.metrics_collector.get_timer_stats("tagged_timer")
        assert stats["count"] == 1


class TestGlobalSystemMonitor:
    """Test global system monitor instance."""
    
    def test_global_instance_exists(self):
        assert system_monitor is not None
        assert isinstance(system_monitor, SystemMonitor)
    
    def test_global_instance_has_components(self):
        assert system_monitor.health_checker is not None
        assert system_monitor.metrics_collector is not None
        assert system_monitor.alert_manager is not None
    
    @pytest.mark.asyncio
    async def test_global_instance_functionality(self):
        # Test that global instance works
        status = await system_monitor.get_system_status()
        assert "overall_status" in status
        assert "health_checks" in status


@pytest.mark.asyncio
async def test_integration_scenario():
    """Test a realistic monitoring scenario."""
    monitor = SystemMonitor()
    
    # Set up custom health check
    async def custom_service_check():
        # Simulate checking an external service
        await asyncio.sleep(0.01)
        return HealthCheckResult(
            name="external_service",
            status=HealthStatus.HEALTHY,
            message="Service is responding"
        )
    
    monitor.health_checker.register_check("external_service", custom_service_check)
    
    # Record some metrics
    monitor.metrics_collector.increment_counter("requests_total", 100)
    monitor.metrics_collector.set_gauge("active_connections", 25)
    monitor.metrics_collector.record_timer("request_duration", 150.0)
    
    # Set up alert threshold
    monitor.alert_manager.set_threshold("active_connections", 30.0, ErrorSeverity.MEDIUM, "greater_than")
    
    # Get system status
    status = await monitor.get_system_status()
    
    # Verify comprehensive status
    assert status["overall_status"] in ["healthy", "degraded", "unhealthy"]
    assert "external_service" in status["health_checks"]
    assert status["metrics"]["counters"]["requests_total"] == 100
    assert status["metrics"]["gauges"]["active_connections"] == 25
    assert len(status["active_alerts"]) == 0  # No alerts should be active
    
    # Trigger an alert
    monitor.metrics_collector.set_gauge("active_connections", 35)
    await monitor.alert_manager.check_thresholds()
    
    # Check alert was triggered
    active_alerts = monitor.alert_manager.get_active_alerts()
    assert len(active_alerts) == 1
    assert active_alerts[0].metric_name == "active_connections"