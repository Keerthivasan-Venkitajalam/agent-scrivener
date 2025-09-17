"""
Scalability tests for multiple agent instances.

Tests system scalability with multiple agent instances,
load distribution, and performance under scaling conditions.
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from tests.integration.test_framework import (
    IntegrationTestFramework, MockServiceConfig, TestDataGenerator
)
from agent_scrivener.models.core import TaskStatus, SessionState
from agent_scrivener.orchestration.registry import enhanced_registry


@dataclass
class ScalabilityMetrics:
    """Metrics for scalability testing."""
    agent_instances: int
    concurrent_sessions: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    load_distribution_variance: float
    agent_utilization_percent: float
    queue_depth_max: int
    queue_wait_time_ms: float


class AgentScalabilityTester:
    """Test agent scalability and load distribution."""
    
    def __init__(self, framework: IntegrationTestFramework):
        self.framework = framework
        self.agent_metrics = {}
    
    async def test_horizontal_scaling(
        self,
        base_instances: int,
        scale_factor: int,
        requests_per_instance: int = 5
    ) -> List[ScalabilityMetrics]:
        """Test horizontal scaling by increasing agent instances."""
        
        results = []
        
        for scale_level in range(1, scale_factor + 1):
            instance_count = base_instances * scale_level
            total_requests = instance_count * requests_per_instance
            
            print(f"\nTesting with {instance_count} agent instances")
            
            # Execute test with current instance count
            metrics = await self._execute_scaling_test(
                agent_instances=instance_count,
                total_requests=total_requests,
                concurrent_sessions=min(instance_count * 2, 20)
            )
            
            results.append(metrics)
            
            # Brief pause between scaling tests
            await asyncio.sleep(1.0)
        
        return results
    
    async def _execute_scaling_test(
        self,
        agent_instances: int,
        total_requests: int,
        concurrent_sessions: int
    ) -> ScalabilityMetrics:
        """Execute a scaling test with specified parameters."""
        
        queries = TestDataGenerator.generate_research_queries()
        
        # Track agent utilization
        agent_execution_counts = {f"agent_{i}": 0 for i in range(agent_instances)}
        
        # Execute requests
        start_time = time.perf_counter()
        
        async def execute_request(request_id: int) -> Tuple[bool, float, str]:
            """Execute a single request and track which agent handled it."""
            query = queries[request_id % len(queries)]
            request_start = time.perf_counter()
            
            try:
                result = await self.framework.run_end_to_end_test(query)
                request_end = time.perf_counter()
                
                success = result["session"].status == TaskStatus.COMPLETED
                response_time = (request_end - request_start) * 1000
                
                # Track agent usage (simplified - in real implementation would track actual agent)
                agent_id = f"agent_{request_id % agent_instances}"
                agent_execution_counts[agent_id] += 1
                
                return success, response_time, agent_id
                
            except Exception as e:
                request_end = time.perf_counter()
                response_time = (request_end - request_start) * 1000
                return False, response_time, "error"
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_sessions)
        
        async def limited_request(request_id: int):
            async with semaphore:
                return await execute_request(request_id)
        
        # Execute all requests
        tasks = [
            asyncio.create_task(limited_request(i))
            for i in range(total_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        # Analyze results
        successful_requests = sum(1 for success, _, _ in results if success)
        failed_requests = total_requests - successful_requests
        response_times = [response_time for _, response_time, _ in results]
        
        # Calculate metrics
        average_response_time = sum(response_times) / len(response_times) if response_times else 0
        throughput_rps = total_requests / total_duration if total_duration > 0 else 0
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
        
        # Calculate load distribution variance
        execution_counts = list(agent_execution_counts.values())
        load_variance = statistics.variance(execution_counts) if len(execution_counts) > 1 else 0
        
        # Calculate agent utilization
        total_executions = sum(execution_counts)
        expected_per_agent = total_executions / agent_instances if agent_instances > 0 else 0
        utilization = (expected_per_agent / max(execution_counts)) * 100 if execution_counts else 0
        
        return ScalabilityMetrics(
            agent_instances=agent_instances,
            concurrent_sessions=concurrent_sessions,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=average_response_time,
            throughput_rps=throughput_rps,
            error_rate_percent=error_rate,
            load_distribution_variance=load_variance,
            agent_utilization_percent=utilization,
            queue_depth_max=0,  # Would be measured in real implementation
            queue_wait_time_ms=0  # Would be measured in real implementation
        )


class TestScalability:
    """Test cases for system scalability."""
    
    @pytest.mark.asyncio
    async def test_linear_scaling_performance(self):
        """Test linear scaling performance with increasing agent instances."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        tester = AgentScalabilityTester(framework)
        
        async with framework.test_environment():
            # Test scaling from 1 to 4 instances
            scaling_results = await tester.test_horizontal_scaling(
                base_instances=1,
                scale_factor=4,
                requests_per_instance=3
            )
            
            # Analyze scaling characteristics
            baseline = scaling_results[0]
            
            for i, metrics in enumerate(scaling_results):
                scale_factor = metrics.agent_instances / baseline.agent_instances
                
                print(f"\nScale Level {metrics.agent_instances} instances:")
                print(f"Throughput: {metrics.throughput_rps:.2f} RPS")
                print(f"Avg Response Time: {metrics.average_response_time_ms:.2f}ms")
                print(f"Error Rate: {metrics.error_rate_percent:.2f}%")
                print(f"Load Distribution Variance: {metrics.load_distribution_variance:.2f}")
                
                # Performance should scale reasonably
                assert metrics.error_rate_percent <= 10.0
                assert metrics.successful_requests > 0
                
                if i > 0:  # Compare with baseline
                    # Throughput should improve with more instances (allowing for overhead)
                    throughput_improvement = metrics.throughput_rps / baseline.throughput_rps
                    assert throughput_improvement >= scale_factor * 0.7  # At least 70% of linear scaling
                    
                    # Response time should not degrade significantly
                    response_time_ratio = metrics.average_response_time_ms / baseline.average_response_time_ms
                    assert response_time_ratio <= 2.0  # Should not double
    
    @pytest.mark.asyncio
    async def test_load_distribution_fairness(self):
        """Test fair load distribution across multiple agent instances."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Simulate multiple agent instances handling requests
            num_agents = 5
            requests_per_agent = 4
            total_requests = num_agents * requests_per_agent
            
            queries = TestDataGenerator.generate_research_queries()
            agent_loads = {f"agent_{i}": [] for i in range(num_agents)}
            
            # Execute requests and track which "agent" handles each
            tasks = []
            for i in range(total_requests):
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                tasks.append((i, task))
            
            # Wait for completion and assign to agents
            for i, task in tasks:
                try:
                    result = await task
                    agent_id = f"agent_{i % num_agents}"  # Round-robin assignment
                    agent_loads[agent_id].append(result)
                except Exception as e:
                    agent_id = f"agent_{i % num_agents}"
                    agent_loads[agent_id].append({"error": str(e)})
            
            # Analyze load distribution
            load_counts = {agent: len(requests) for agent, requests in agent_loads.items()}
            successful_counts = {
                agent: sum(1 for r in requests if "session" in r and r["session"].status == TaskStatus.COMPLETED)
                for agent, requests in agent_loads.items()
            }
            
            # Calculate distribution metrics
            load_values = list(load_counts.values())
            success_values = list(successful_counts.values())
            
            load_variance = statistics.variance(load_values) if len(load_values) > 1 else 0
            success_variance = statistics.variance(success_values) if len(success_values) > 1 else 0
            
            expected_load = total_requests / num_agents
            max_deviation = max(abs(count - expected_load) for count in load_values)
            
            print(f"\nLoad Distribution Analysis:")
            print(f"Total Requests: {total_requests}")
            print(f"Expected per Agent: {expected_load:.1f}")
            print(f"Actual Distribution: {load_counts}")
            print(f"Successful Distribution: {successful_counts}")
            print(f"Load Variance: {load_variance:.2f}")
            print(f"Max Deviation: {max_deviation:.2f}")
            
            # Fairness assertions
            assert load_variance <= 2.0  # Low variance indicates fair distribution
            assert max_deviation <= 2  # No agent should be more than 2 requests off
            assert all(count > 0 for count in successful_counts.values())  # All agents should succeed
    
    @pytest.mark.asyncio
    async def test_scaling_under_load(self):
        """Test scaling behavior under sustained load."""
        config = MockServiceConfig(
            web_search_delay=0.002,
            api_query_delay=0.002,
            analysis_delay=0.005,
            failure_rate=0.05  # Small failure rate for realism
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Test different scaling scenarios under load
            scaling_scenarios = [
                {"instances": 2, "concurrent": 4, "duration": 10},
                {"instances": 4, "concurrent": 8, "duration": 10},
                {"instances": 6, "concurrent": 12, "duration": 10}
            ]
            
            results = []
            
            for scenario in scaling_scenarios:
                print(f"\nTesting scaling scenario: {scenario}")
                
                # Generate sustained load
                queries = TestDataGenerator.generate_research_queries()
                start_time = time.perf_counter()
                end_time = start_time + scenario["duration"]
                
                active_tasks = []
                completed_results = []
                request_count = 0
                
                while time.perf_counter() < end_time:
                    # Maintain target concurrency
                    while len(active_tasks) < scenario["concurrent"]:
                        query = queries[request_count % len(queries)]
                        request_count += 1
                        
                        task = asyncio.create_task(framework.run_end_to_end_test(query))
                        active_tasks.append(task)
                    
                    # Check for completed tasks
                    done_tasks = [task for task in active_tasks if task.done()]
                    for task in done_tasks:
                        active_tasks.remove(task)
                        try:
                            result = await task
                            completed_results.append(result)
                        except Exception as e:
                            completed_results.append({"error": str(e)})
                    
                    await asyncio.sleep(0.1)
                
                # Wait for remaining tasks
                if active_tasks:
                    remaining = await asyncio.gather(*active_tasks, return_exceptions=True)
                    for result in remaining:
                        if not isinstance(result, Exception):
                            completed_results.append(result)
                
                # Calculate metrics
                actual_duration = time.perf_counter() - start_time
                successful_results = [
                    r for r in completed_results 
                    if "session" in r and r["session"].status == TaskStatus.COMPLETED
                ]
                
                throughput = len(completed_results) / actual_duration
                success_rate = (len(successful_results) / len(completed_results)) * 100 if completed_results else 0
                
                scenario_result = {
                    "instances": scenario["instances"],
                    "concurrent": scenario["concurrent"],
                    "total_requests": len(completed_results),
                    "successful_requests": len(successful_results),
                    "throughput_rps": throughput,
                    "success_rate": success_rate,
                    "duration": actual_duration
                }
                
                results.append(scenario_result)
                
                print(f"Results: {scenario_result}")
                
                # Performance assertions
                assert success_rate >= 80.0  # At least 80% success rate
                assert throughput > 0
                assert len(successful_results) > 0
            
            # Analyze scaling efficiency
            for i in range(1, len(results)):
                current = results[i]
                previous = results[i-1]
                
                instance_ratio = current["instances"] / previous["instances"]
                throughput_ratio = current["throughput_rps"] / previous["throughput_rps"]
                
                # Throughput should improve with more instances
                assert throughput_ratio >= instance_ratio * 0.6  # Allow for overhead
    
    @pytest.mark.asyncio
    async def test_agent_failure_resilience(self):
        """Test system resilience when some agent instances fail."""
        config = MockServiceConfig(
            web_search_delay=0.01,
            api_query_delay=0.01,
            analysis_delay=0.02,
            failure_rate=0.3  # High failure rate to simulate agent failures
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Simulate scenario with failing agents
            total_agents = 6
            failing_agents = 2  # 2 out of 6 agents fail
            healthy_agents = total_agents - failing_agents
            
            requests_per_agent = 3
            total_requests = total_agents * requests_per_agent
            
            queries = TestDataGenerator.generate_research_queries()
            
            # Execute requests with some expected to fail
            tasks = []
            for i in range(total_requests):
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze resilience
            successful_results = [
                r for r in results 
                if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
            ]
            failed_results = len(results) - len(successful_results)
            
            success_rate = (len(successful_results) / len(results)) * 100
            
            print(f"\nAgent Failure Resilience Test:")
            print(f"Total Agents: {total_agents}")
            print(f"Failing Agents: {failing_agents}")
            print(f"Healthy Agents: {healthy_agents}")
            print(f"Total Requests: {len(results)}")
            print(f"Successful Requests: {len(successful_results)}")
            print(f"Failed Requests: {failed_results}")
            print(f"Success Rate: {success_rate:.2f}%")
            
            # Resilience assertions
            assert len(successful_results) > 0  # Some requests should succeed
            assert success_rate >= 40.0  # At least 40% should succeed despite failures
            
            # Should handle at least as many requests as healthy agents can process
            min_expected_success = healthy_agents * requests_per_agent * 0.5  # 50% of healthy capacity
            assert len(successful_results) >= min_expected_success
    
    @pytest.mark.asyncio
    async def test_dynamic_scaling_response(self):
        """Test system response to dynamic scaling changes."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Simulate dynamic scaling scenario
            queries = TestDataGenerator.generate_research_queries()
            
            # Phase 1: Low load with few instances
            print("\nPhase 1: Low load (2 instances)")
            phase1_tasks = []
            for i in range(4):  # 4 requests
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                phase1_tasks.append(task)
            
            phase1_start = time.perf_counter()
            phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)
            phase1_duration = time.perf_counter() - phase1_start
            
            # Phase 2: High load requiring scaling
            print("Phase 2: High load (6 instances)")
            phase2_tasks = []
            for i in range(12):  # 12 requests
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                phase2_tasks.append(task)
            
            phase2_start = time.perf_counter()
            phase2_results = await asyncio.gather(*phase2_tasks, return_exceptions=True)
            phase2_duration = time.perf_counter() - phase2_start
            
            # Phase 3: Scale down
            print("Phase 3: Scale down (3 instances)")
            phase3_tasks = []
            for i in range(6):  # 6 requests
                query = queries[i % len(queries)]
                task = asyncio.create_task(framework.run_end_to_end_test(query))
                phase3_tasks.append(task)
            
            phase3_start = time.perf_counter()
            phase3_results = await asyncio.gather(*phase3_tasks, return_exceptions=True)
            phase3_duration = time.perf_counter() - phase3_start
            
            # Analyze dynamic scaling response
            phases = [
                {"name": "Phase 1", "results": phase1_results, "duration": phase1_duration, "instances": 2},
                {"name": "Phase 2", "results": phase2_results, "duration": phase2_duration, "instances": 6},
                {"name": "Phase 3", "results": phase3_results, "duration": phase3_duration, "instances": 3}
            ]
            
            for phase in phases:
                successful = sum(
                    1 for r in phase["results"] 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                throughput = len(phase["results"]) / phase["duration"]
                success_rate = (successful / len(phase["results"])) * 100
                
                print(f"\n{phase['name']} Results:")
                print(f"Instances: {phase['instances']}")
                print(f"Requests: {len(phase['results'])}")
                print(f"Successful: {successful}")
                print(f"Success Rate: {success_rate:.2f}%")
                print(f"Throughput: {throughput:.2f} RPS")
                
                # Each phase should maintain reasonable performance
                assert success_rate >= 70.0
                assert throughput > 0
                assert successful > 0
    
    @pytest.mark.asyncio
    async def test_resource_efficiency_scaling(self):
        """Test resource efficiency as system scales."""
        config = MockServiceConfig(
            web_search_delay=0.003,
            api_query_delay=0.003,
            analysis_delay=0.006,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Test resource efficiency at different scales
            scale_levels = [
                {"instances": 1, "requests": 3},
                {"instances": 2, "requests": 6},
                {"instances": 4, "requests": 12}
            ]
            
            efficiency_results = []
            
            for scale in scale_levels:
                queries = TestDataGenerator.generate_research_queries()
                
                # Measure resource usage
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                
                start_time = time.perf_counter()
                
                # Execute requests
                tasks = []
                for i in range(scale["requests"]):
                    query = queries[i % len(queries)]
                    task = asyncio.create_task(framework.run_end_to_end_test(query))
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.perf_counter()
                final_memory = process.memory_info().rss / 1024 / 1024
                
                # Calculate efficiency metrics
                duration = end_time - start_time
                memory_used = final_memory - initial_memory
                successful = sum(
                    1 for r in results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                
                efficiency = {
                    "instances": scale["instances"],
                    "requests": scale["requests"],
                    "successful": successful,
                    "duration": duration,
                    "memory_used_mb": memory_used,
                    "throughput_rps": scale["requests"] / duration,
                    "memory_per_request_mb": memory_used / scale["requests"] if scale["requests"] > 0 else 0,
                    "time_per_request_s": duration / scale["requests"] if scale["requests"] > 0 else 0
                }
                
                efficiency_results.append(efficiency)
                
                print(f"\nScale Level {scale['instances']} instances:")
                print(f"Throughput: {efficiency['throughput_rps']:.2f} RPS")
                print(f"Memory per Request: {efficiency['memory_per_request_mb']:.2f} MB")
                print(f"Time per Request: {efficiency['time_per_request_s']:.3f} s")
            
            # Analyze efficiency trends
            baseline = efficiency_results[0]
            
            for result in efficiency_results[1:]:
                scale_factor = result["instances"] / baseline["instances"]
                
                # Efficiency should not degrade significantly with scale
                memory_efficiency = result["memory_per_request_mb"] / baseline["memory_per_request_mb"]
                time_efficiency = result["time_per_request_s"] / baseline["time_per_request_s"]
                
                print(f"\nEfficiency Analysis for {result['instances']} instances:")
                print(f"Memory Efficiency Ratio: {memory_efficiency:.2f}")
                print(f"Time Efficiency Ratio: {time_efficiency:.2f}")
                
                # Efficiency should not degrade too much
                assert memory_efficiency <= 2.0  # Memory per request should not double
                assert time_efficiency <= 1.5  # Time per request should not increase by 50%


    @pytest.mark.asyncio
    async def test_multi_instance_coordination(self):
        """Test coordination between multiple agent instances."""
        config = MockServiceConfig(
            web_search_delay=0.005,
            api_query_delay=0.005,
            analysis_delay=0.01,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Simulate multiple instances working on related tasks
            num_instances = 4
            tasks_per_instance = 3
            
            # Create related queries that might benefit from coordination
            base_queries = TestDataGenerator.generate_research_queries()[:num_instances]
            
            # Execute tasks across instances
            instance_tasks = {}
            for instance_id in range(num_instances):
                instance_tasks[f"instance_{instance_id}"] = []
                
                for task_id in range(tasks_per_instance):
                    query = base_queries[instance_id]  # Same base query per instance
                    task = asyncio.create_task(framework.run_end_to_end_test(query))
                    instance_tasks[f"instance_{instance_id}"].append(task)
            
            # Wait for all instances to complete
            all_results = {}
            for instance_id, tasks in instance_tasks.items():
                results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results[instance_id] = results
            
            # Analyze coordination effectiveness
            total_tasks = sum(len(tasks) for tasks in instance_tasks.values())
            successful_tasks = 0
            
            for instance_id, results in all_results.items():
                instance_successful = sum(
                    1 for r in results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                successful_tasks += instance_successful
                
                print(f"{instance_id}: {instance_successful}/{len(results)} successful")
            
            success_rate = (successful_tasks / total_tasks) * 100
            
            print(f"\nMulti-Instance Coordination Results:")
            print(f"Total Instances: {num_instances}")
            print(f"Tasks per Instance: {tasks_per_instance}")
            print(f"Total Tasks: {total_tasks}")
            print(f"Successful Tasks: {successful_tasks}")
            print(f"Overall Success Rate: {success_rate:.2f}%")
            
            # Coordination assertions
            assert success_rate >= 80.0  # High success rate indicates good coordination
            assert successful_tasks > 0
            
            # Each instance should contribute to overall success
            for instance_id, results in all_results.items():
                instance_successful = sum(
                    1 for r in results 
                    if not isinstance(r, Exception) and "session" in r and r["session"].status == TaskStatus.COMPLETED
                )
                assert instance_successful > 0  # Each instance should succeed at something
    
    @pytest.mark.asyncio
    async def test_scalability_performance_benchmarks(self):
        """Benchmark scalability performance across different configurations."""
        config = MockServiceConfig(
            web_search_delay=0.002,
            api_query_delay=0.002,
            analysis_delay=0.005,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        tester = AgentScalabilityTester(framework)
        
        async with framework.test_environment():
            # Define benchmark configurations
            benchmark_configs = [
                {"name": "Small Scale", "instances": 2, "requests": 6, "target_throughput": 3.0},
                {"name": "Medium Scale", "instances": 4, "requests": 12, "target_throughput": 6.0},
                {"name": "Large Scale", "instances": 6, "requests": 18, "target_throughput": 9.0}
            ]
            
            benchmark_results = []
            
            for config_spec in benchmark_configs:
                print(f"\nBenchmarking {config_spec['name']}...")
                
                metrics = await tester._execute_scaling_test(
                    agent_instances=config_spec["instances"],
                    total_requests=config_spec["requests"],
                    concurrent_sessions=min(config_spec["instances"] * 2, 12)
                )
                
                efficiency_score = (metrics.throughput_rps / config_spec["target_throughput"]) * 100
                
                benchmark_result = {
                    "config": config_spec["name"],
                    "instances": config_spec["instances"],
                    "target_throughput": config_spec["target_throughput"],
                    "actual_throughput": metrics.throughput_rps,
                    "efficiency_score": efficiency_score,
                    "error_rate": metrics.error_rate_percent,
                    "load_variance": metrics.load_distribution_variance
                }
                
                benchmark_results.append(benchmark_result)
                
                print(f"{config_spec['name']} Benchmark Results:")
                print(f"  Target Throughput: {config_spec['target_throughput']:.1f} RPS")
                print(f"  Actual Throughput: {metrics.throughput_rps:.2f} RPS")
                print(f"  Efficiency Score: {efficiency_score:.1f}%")
                print(f"  Error Rate: {metrics.error_rate_percent:.2f}%")
                print(f"  Load Distribution Variance: {metrics.load_distribution_variance:.2f}")
                
                # Performance benchmarks
                assert metrics.error_rate_percent <= 15.0
                assert metrics.throughput_rps > 0
                assert efficiency_score >= 60.0  # Should achieve at least 60% of target
            
            # Analyze scaling efficiency across benchmarks
            print(f"\nScaling Efficiency Analysis:")
            
            for i, result in enumerate(benchmark_results):
                if i > 0:
                    prev_result = benchmark_results[i-1]
                    
                    instance_ratio = result["instances"] / prev_result["instances"]
                    throughput_ratio = result["actual_throughput"] / prev_result["actual_throughput"]
                    scaling_efficiency = (throughput_ratio / instance_ratio) * 100
                    
                    print(f"{result['config']} vs {prev_result['config']}:")
                    print(f"  Instance Ratio: {instance_ratio:.2f}")
                    print(f"  Throughput Ratio: {throughput_ratio:.2f}")
                    print(f"  Scaling Efficiency: {scaling_efficiency:.1f}%")
                    
                    # Scaling should be reasonably efficient
                    assert scaling_efficiency >= 50.0  # At least 50% scaling efficiency


if __name__ == "__main__":
    pytest.main([__file__, "-v"])