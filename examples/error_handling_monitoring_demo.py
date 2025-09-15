"""
Demonstration of error handling and monitoring systems working together.
"""

import asyncio
import time
from agent_scrivener.utils.error_handler import (
    error_handler, with_error_handling, RetryConfig, CircuitBreakerConfig
)
from agent_scrivener.utils.monitoring import system_monitor, timed, counted
from agent_scrivener.models.errors import NetworkError, ExternalAPIError


@timed("demo_function_duration")
@counted("demo_function_calls")
@with_error_handling(
    retry_config=RetryConfig(max_retries=3, base_delay=0.1),
    circuit_breaker_service="demo_service",
    circuit_config=CircuitBreakerConfig(failure_threshold=2),
    error_types=(NetworkError, ExternalAPIError)
)
async def flaky_service_call(fail_count: int = 0):
    """Simulate a flaky external service call."""
    # Increment a counter to track attempts
    system_monitor.metrics_collector.increment_counter("service_attempts")
    
    # Simulate some processing time
    await asyncio.sleep(0.05)
    
    if fail_count > 0:
        system_monitor.metrics_collector.increment_counter("service_failures")
        raise NetworkError(f"Service temporarily unavailable (fail_count: {fail_count})")
    
    system_monitor.metrics_collector.increment_counter("service_successes")
    return {"status": "success", "data": "Important data"}


async def demo_error_handling_and_monitoring():
    """Demonstrate error handling and monitoring integration."""
    print("=== Error Handling and Monitoring Demo ===\n")
    
    # Start monitoring
    system_monitor.start_monitoring(check_interval=1.0)
    
    # Set up custom alert for service failures
    system_monitor.alert_manager.set_threshold(
        "service_failures", 
        3.0, 
        severity="medium", 
        comparison="greater_than"
    )
    
    # Add custom alert handler
    async def demo_alert_handler(alert):
        print(f"🚨 ALERT: {alert.message}")
    
    system_monitor.alert_manager.add_alert_handler(demo_alert_handler)
    
    print("1. Testing successful service calls...")
    for i in range(3):
        try:
            result = await flaky_service_call(fail_count=0)
            print(f"   Call {i+1}: {result['status']}")
        except Exception as e:
            print(f"   Call {i+1}: Failed - {e}")
    
    print("\n2. Testing service calls with failures (will retry)...")
    for i in range(2):
        try:
            # This will fail initially but succeed after retries
            result = await flaky_service_call(fail_count=2)  # Will fail 2 times then succeed
            print(f"   Call {i+1}: {result['status']} (after retries)")
        except Exception as e:
            print(f"   Call {i+1}: Failed after retries - {e}")
    
    print("\n3. Testing circuit breaker (multiple failures)...")
    for i in range(4):
        try:
            result = await flaky_service_call(fail_count=5)  # Will always fail
            print(f"   Call {i+1}: {result['status']}")
        except NetworkError as e:
            print(f"   Call {i+1}: Network error - {e}")
        except ExternalAPIError as e:
            print(f"   Call {i+1}: Circuit breaker blocked - {e}")
    
    # Wait a moment for metrics to be collected
    await asyncio.sleep(0.1)
    
    print("\n4. System Status Report:")
    status = await system_monitor.get_system_status()
    
    print(f"   Overall Status: {status['overall_status'].upper()}")
    print(f"   Active Alerts: {len(status['active_alerts'])}")
    
    # Show metrics
    metrics = status['metrics']
    print(f"\n   Service Metrics:")
    print(f"   - Total Attempts: {metrics['counters'].get('service_attempts', 0)}")
    print(f"   - Successes: {metrics['counters'].get('service_successes', 0)}")
    print(f"   - Failures: {metrics['counters'].get('service_failures', 0)}")
    print(f"   - Function Calls: {metrics['counters'].get('demo_function_calls', 0)}")
    
    # Show timing stats
    if 'demo_function_duration' in metrics['timers']:
        timing = metrics['timers']['demo_function_duration']
        print(f"   - Average Duration: {timing.get('mean', 0):.2f}ms")
        print(f"   - Min Duration: {timing.get('min', 0):.2f}ms")
        print(f"   - Max Duration: {timing.get('max', 0):.2f}ms")
    
    # Show health checks
    print(f"\n   Health Checks:")
    for name, check in status['health_checks'].items():
        print(f"   - {name}: {check['status'].upper()} ({check['response_time_ms']:.1f}ms)")
    
    # Show active alerts
    if status['active_alerts']:
        print(f"\n   Active Alerts:")
        for alert in status['active_alerts']:
            print(f"   - {alert['severity'].upper()}: {alert['message']}")
    
    print("\n5. Error Handler Statistics:")
    error_stats = error_handler.get_error_stats()
    for agent, stats in error_stats.items():
        print(f"   - {agent}: {stats}")
    
    # Stop monitoring
    system_monitor.stop_monitoring()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(demo_error_handling_and_monitoring())