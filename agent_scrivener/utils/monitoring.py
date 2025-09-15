"""
System monitoring and health checks for Agent Scrivener.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque

from ..models.errors import ErrorCategory, ErrorSeverity


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: Optional[float] = None


@dataclass
class Metric:
    """A system metric."""
    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: ErrorSeverity
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class HealthChecker:
    """Health check manager."""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable[[], Awaitable[HealthCheckResult]]):
        """Register a health check function."""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        start_time = time.time()
        try:
            result = await self.checks[name]()
            result.response_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            self.logger.error(f"Health check '{name}' failed: {str(e)}")
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        tasks = []
        
        for name in self.checks:
            tasks.append(self.run_check(name))
        
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, name in enumerate(self.checks):
            result = check_results[i]
            if isinstance(result, Exception):
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check exception: {str(result)}"
                )
            results[name] = result
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status."""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


class MetricsCollector:
    """Metrics collection and aggregation."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            self._record_metric(name, MetricType.COUNTER, self.counters[name], tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            self.gauges[name] = value
            self._record_metric(name, MetricType.GAUGE, value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._record_metric(name, MetricType.HISTOGRAM, value, tags)
    
    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer value."""
        with self._lock:
            self.timers[name].append(duration_ms)
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            self._record_metric(name, MetricType.TIMER, duration_ms, tags)
    
    def _record_metric(self, name: str, metric_type: MetricType, value: float, tags: Optional[Dict[str, str]]):
        """Record a metric in the history."""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            tags=tags or {}
        )
        self.metrics[name].append(metric)
    
    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        return self.gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self.histograms.get(name, [])
        if not values:
            return {"count": 0}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        values = self.timers.get(name, [])
        if not values:
            return {"count": 0}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {name: self.get_histogram_stats(name) for name in self.histograms},
                "timers": {name: self.get_timer_stats(name) for name in self.timers}
            }
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.metrics.clear()


class AlertManager:
    """Alert management system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.thresholds: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable[[Alert], Awaitable[None]]] = []
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def set_threshold(
        self,
        metric_name: str,
        threshold: float,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        comparison: str = "greater_than"  # "greater_than", "less_than", "equals"
    ):
        """Set an alert threshold for a metric."""
        self.thresholds[metric_name] = {
            "threshold": threshold,
            "severity": severity,
            "comparison": comparison
        }
        self.logger.info(f"Set alert threshold for {metric_name}: {comparison} {threshold}")
    
    def add_alert_handler(self, handler: Callable[[Alert], Awaitable[None]]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    async def check_thresholds(self):
        """Check all metrics against their thresholds."""
        current_metrics = self.metrics.get_all_metrics()
        
        for metric_name, config in self.thresholds.items():
            threshold = config["threshold"]
            severity = config["severity"]
            comparison = config["comparison"]
            
            # Get current value based on metric type
            current_value = None
            if metric_name in current_metrics["counters"]:
                current_value = current_metrics["counters"][metric_name]
            elif metric_name in current_metrics["gauges"]:
                current_value = current_metrics["gauges"][metric_name]
            elif metric_name in current_metrics["histograms"]:
                stats = current_metrics["histograms"][metric_name]
                if stats:
                    current_value = stats.get("mean")
            elif metric_name in current_metrics["timers"]:
                stats = current_metrics["timers"][metric_name]
                if stats:
                    current_value = stats.get("mean")
            
            if current_value is None:
                continue
            
            # Check threshold
            alert_triggered = False
            if comparison == "greater_than" and current_value > threshold:
                alert_triggered = True
            elif comparison == "less_than" and current_value < threshold:
                alert_triggered = True
            elif comparison == "equals" and current_value == threshold:
                alert_triggered = True
            
            alert_id = f"{metric_name}_{comparison}_{threshold}"
            
            if alert_triggered:
                if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                    # New alert
                    alert = Alert(
                        id=alert_id,
                        severity=severity,
                        message=f"Metric {metric_name} ({current_value}) {comparison} threshold ({threshold})",
                        metric_name=metric_name,
                        threshold=threshold,
                        current_value=current_value
                    )
                    self.alerts[alert_id] = alert
                    await self._trigger_alert(alert)
            else:
                # Check if we need to resolve an existing alert
                if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                    self.alerts[alert_id].resolved = True
                    self.alerts[alert_id].resolved_at = datetime.now()
                    await self._resolve_alert(self.alerts[alert_id])
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        self.logger.warning(f"ALERT TRIGGERED: {alert.message}")
        
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {str(e)}")
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert."""
        self.logger.info(f"ALERT RESOLVED: {alert.message}")
        
        # You could add resolution handlers here if needed
    
    def start_monitoring(self, check_interval: float = 30.0):
        """Start continuous threshold monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(check_interval))
        self.logger.info("Started alert monitoring")
    
    def stop_monitoring(self):
        """Stop continuous threshold monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        self.logger.info("Stopped alert monitoring")
    
    async def _monitor_loop(self, check_interval: float):
        """Continuous monitoring loop."""
        while self._monitoring:
            try:
                await self.check_thresholds()
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(check_interval)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        return list(self.alerts.values())


class SystemMonitor:
    """Main system monitoring coordinator."""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.logger = logging.getLogger(__name__)
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Set up default metrics collection
        self._setup_default_metrics()
        
        # Set up default alert thresholds
        self._setup_default_alerts()
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        
        async def system_resources_check() -> HealthCheckResult:
            """Check system resource usage."""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Update metrics
                self.metrics_collector.set_gauge("system.cpu_percent", cpu_percent)
                self.metrics_collector.set_gauge("system.memory_percent", memory.percent)
                self.metrics_collector.set_gauge("system.disk_percent", disk.percent)
                
                # Determine status
                if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                    status = HealthStatus.UNHEALTHY
                    message = f"High resource usage: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
                elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
                    status = HealthStatus.DEGRADED
                    message = f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Resource usage normal: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
                
                return HealthCheckResult(
                    name="system_resources",
                    status=status,
                    message=message,
                    details={
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "disk_percent": disk.percent
                    }
                )
            except Exception as e:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check system resources: {str(e)}"
                )
        
        async def memory_usage_check() -> HealthCheckResult:
            """Check application memory usage."""
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self.metrics_collector.set_gauge("app.memory_mb", memory_mb)
                
                if memory_mb > 1000:  # 1GB
                    status = HealthStatus.UNHEALTHY
                    message = f"High memory usage: {memory_mb:.1f}MB"
                elif memory_mb > 500:  # 500MB
                    status = HealthStatus.DEGRADED
                    message = f"Moderate memory usage: {memory_mb:.1f}MB"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage normal: {memory_mb:.1f}MB"
                
                return HealthCheckResult(
                    name="memory_usage",
                    status=status,
                    message=message,
                    details={"memory_mb": memory_mb}
                )
            except Exception as e:
                return HealthCheckResult(
                    name="memory_usage",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check memory usage: {str(e)}"
                )
        
        self.health_checker.register_check("system_resources", system_resources_check)
        self.health_checker.register_check("memory_usage", memory_usage_check)
    
    def _setup_default_metrics(self):
        """Set up default metrics collection."""
        # Initialize some basic counters
        self.metrics_collector.increment_counter("app.startup", 1)
    
    def _setup_default_alerts(self):
        """Set up default alert thresholds."""
        # System resource alerts
        self.alert_manager.set_threshold("system.cpu_percent", 80.0, ErrorSeverity.MEDIUM, "greater_than")
        self.alert_manager.set_threshold("system.memory_percent", 85.0, ErrorSeverity.MEDIUM, "greater_than")
        self.alert_manager.set_threshold("system.disk_percent", 90.0, ErrorSeverity.HIGH, "greater_than")
        
        # Application memory alerts
        self.alert_manager.set_threshold("app.memory_mb", 800.0, ErrorSeverity.MEDIUM, "greater_than")
        
        # Add default alert handler
        async def log_alert_handler(alert: Alert):
            if alert.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"CRITICAL ALERT: {alert.message}")
            elif alert.severity == ErrorSeverity.HIGH:
                self.logger.error(f"HIGH ALERT: {alert.message}")
            else:
                self.logger.warning(f"ALERT: {alert.message}")
        
        self.alert_manager.add_alert_handler(log_alert_handler)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_results = await self.health_checker.run_all_checks()
        overall_status = self.health_checker.get_overall_status(health_results)
        metrics = self.metrics_collector.get_all_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "health_checks": {name: {
                "status": result.status.value,
                "message": result.message,
                "response_time_ms": result.response_time_ms,
                "details": result.details
            } for name, result in health_results.items()},
            "metrics": metrics,
            "active_alerts": [{
                "id": alert.id,
                "severity": alert.severity.value,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "threshold": alert.threshold,
                "current_value": alert.current_value,
                "timestamp": alert.timestamp.isoformat()
            } for alert in active_alerts]
        }
    
    def start_monitoring(self, check_interval: float = 30.0):
        """Start all monitoring systems."""
        self.alert_manager.start_monitoring(check_interval)
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.alert_manager.stop_monitoring()
        self.logger.info("System monitoring stopped")


# Global system monitor instance
system_monitor = SystemMonitor()


def timed(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    system_monitor.metrics_collector.record_timer(metric_name, duration_ms, tags)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    system_monitor.metrics_collector.record_timer(metric_name, duration_ms, tags)
            return sync_wrapper
    return decorator


def counted(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                system_monitor.metrics_collector.increment_counter(metric_name, 1, tags)
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                system_monitor.metrics_collector.increment_counter(metric_name, 1, tags)
                return func(*args, **kwargs)
            return sync_wrapper
    return decorator