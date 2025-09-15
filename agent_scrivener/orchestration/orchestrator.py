"""
Agent Orchestration System for Agent Scrivener.

This module provides the core orchestration functionality for managing
multiple agents, task dispatching, result aggregation, and progress tracking.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..agents.base import BaseAgent, AgentResult, agent_registry
from ..models.core import (
    ResearchSession, ResearchPlan, ResearchTask, TaskStatus, 
    SessionState, WorkflowStep, AgentExecution, SessionMetrics
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskExecution:
    """Represents an active task execution."""
    task: ResearchTask
    agent: BaseAgent
    execution: AgentExecution
    future: Optional[asyncio.Future] = None
    started_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: int = 3600  # 1 hour default timeout


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration system."""
    max_concurrent_tasks: int = 5
    task_timeout_seconds: int = 3600
    progress_update_interval_seconds: int = 30
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    enable_parallel_execution: bool = True


class ProgressTracker:
    """Tracks and reports progress of research sessions."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ProgressTracker")
        self._progress_callbacks: Dict[str, List[Callable]] = {}
        self._session_progress: Dict[str, Dict[str, Any]] = {}
    
    def register_progress_callback(self, session_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for progress updates."""
        if session_id not in self._progress_callbacks:
            self._progress_callbacks[session_id] = []
        self._progress_callbacks[session_id].append(callback)
    
    def update_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """Update progress for a session and notify callbacks."""
        self._session_progress[session_id] = {
            **self._session_progress.get(session_id, {}),
            **progress_data,
            "updated_at": datetime.now().isoformat()
        }
        
        # Notify callbacks
        callbacks = self._progress_callbacks.get(session_id, [])
        for callback in callbacks:
            try:
                callback(self._session_progress[session_id])
            except Exception as e:
                self.logger.error(f"Progress callback failed for session {session_id}: {e}")
    
    def get_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a session."""
        return self._session_progress.get(session_id)
    
    def cleanup_session(self, session_id: str):
        """Clean up progress tracking for a completed session."""
        self._progress_callbacks.pop(session_id, None)
        self._session_progress.pop(session_id, None)


class TaskDispatcher:
    """Handles task dispatching and execution management."""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.TaskDispatcher")
        self._active_executions: Dict[str, TaskExecution] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self._priority_queue: List[TaskExecution] = []
        self._task_metrics: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch_task(self, task: ResearchTask, session: ResearchSession) -> AgentExecution:
        """
        Dispatch a task to the appropriate agent.
        
        Args:
            task: Task to execute
            session: Research session context
            
        Returns:
            AgentExecution: Execution tracking object
        """
        # Get the appropriate agent
        agent = agent_registry.get_agent(task.assigned_agent)
        if not agent:
            raise ValueError(f"Agent not found: {task.assigned_agent}")
        
        # Create execution tracking
        execution = AgentExecution(
            execution_id=str(uuid.uuid4()),
            agent_name=task.assigned_agent,
            task_id=task.task_id,
            input_data=task.parameters
        )
        
        # Create task execution wrapper
        task_execution = TaskExecution(
            task=task,
            agent=agent,
            execution=execution,
            timeout_seconds=self.config.task_timeout_seconds
        )
        
        # Execute the task
        await self._execute_task_with_semaphore(task_execution, session)
        
        return execution
    
    async def _execute_task_with_semaphore(self, task_execution: TaskExecution, session: ResearchSession):
        """Execute a task with concurrency control."""
        async with self._semaphore:
            await self._execute_task(task_execution, session)
    
    async def _execute_task(self, task_execution: TaskExecution, session: ResearchSession):
        """Execute a single task with timeout and error handling."""
        task = task_execution.task
        agent = task_execution.agent
        execution = task_execution.execution
        
        self.logger.info(f"Starting task {task.task_id} with agent {agent.name}")
        
        # Mark task as in progress
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        execution.status = TaskStatus.IN_PROGRESS
        
        # Store active execution
        self._active_executions[execution.execution_id] = task_execution
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute(**task.parameters),
                timeout=task_execution.timeout_seconds
            )
            
            if result.success:
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result.data
                
                execution.mark_completed({"result": result.data})
                
                self.logger.info(f"Task {task.task_id} completed successfully")
            else:
                # Task failed
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error_message = result.error
                
                execution.mark_failed(result.error or "Unknown error")
                
                self.logger.error(f"Task {task.task_id} failed: {result.error}")
        
        except asyncio.TimeoutError:
            # Task timed out
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = f"Task timed out after {task_execution.timeout_seconds} seconds"
            
            execution.mark_failed(task.error_message)
            
            self.logger.error(f"Task {task.task_id} timed out")
        
        except Exception as e:
            # Unexpected error
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = str(e)
            
            execution.mark_failed(str(e))
            
            self.logger.error(f"Task {task.task_id} failed with exception: {e}")
        
        finally:
            # Clean up active execution
            self._active_executions.pop(execution.execution_id, None)
    
    def get_active_executions(self) -> List[TaskExecution]:
        """Get list of currently active task executions."""
        return list(self._active_executions.values())
    
    def cancel_task(self, execution_id: str) -> bool:
        """Cancel a running task execution."""
        task_execution = self._active_executions.get(execution_id)
        if not task_execution or not task_execution.future:
            return False
        
        task_execution.future.cancel()
        task_execution.task.status = TaskStatus.CANCELLED
        task_execution.execution.status = TaskStatus.CANCELLED
        
        self.logger.info(f"Cancelled task execution {execution_id}")
        return True
    
    def prioritize_tasks(self, tasks: List[ResearchTask]) -> List[ResearchTask]:
        """
        Prioritize tasks based on various factors.
        
        Args:
            tasks: List of tasks to prioritize
            
        Returns:
            List of tasks sorted by priority (highest first)
        """
        def calculate_priority(task: ResearchTask) -> float:
            priority = 0.0
            
            # Base priority by task type
            type_priorities = {
                "web_search": 1.0,
                "academic_search": 1.0,
                "content_analysis": 0.8,
                "document_generation": 0.6,
                "citation_formatting": 0.4
            }
            priority += type_priorities.get(task.task_type, 0.5)
            
            # Boost priority for tasks with fewer dependencies
            dependency_factor = 1.0 / (len(task.dependencies) + 1)
            priority += dependency_factor * 0.5
            
            # Boost priority for tasks that unblock many other tasks
            # (This would require dependency graph analysis in practice)
            
            # Consider historical performance of the assigned agent
            agent_metrics = self._task_metrics.get(task.assigned_agent, {})
            if agent_metrics:
                success_rate = agent_metrics.get("success_rate", 0.5)
                avg_time = agent_metrics.get("avg_execution_time", 60.0)
                
                # Prefer agents with higher success rates and faster execution
                priority += success_rate * 0.3
                priority += (1.0 / max(avg_time, 1.0)) * 0.2
            
            return priority
        
        return sorted(tasks, key=calculate_priority, reverse=True)
    
    def update_task_metrics(self, agent_name: str, success: bool, execution_time: float):
        """Update performance metrics for an agent."""
        if agent_name not in self._task_metrics:
            self._task_metrics[agent_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "avg_execution_time": 0.0,
                "success_rate": 0.0
            }
        
        metrics = self._task_metrics[agent_name]
        metrics["total_executions"] += 1
        metrics["total_time"] += execution_time
        
        if success:
            metrics["successful_executions"] += 1
        
        # Update derived metrics
        metrics["success_rate"] = metrics["successful_executions"] / metrics["total_executions"]
        metrics["avg_execution_time"] = metrics["total_time"] / metrics["total_executions"]
    
    def get_load_balancing_info(self) -> Dict[str, Any]:
        """Get information for load balancing decisions."""
        return {
            "active_executions": len(self._active_executions),
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "available_slots": self.config.max_concurrent_tasks - len(self._active_executions),
            "queue_size": self._task_queue.qsize(),
            "agent_metrics": self._task_metrics.copy()
        }


class ResultAggregator:
    """Aggregates results from multiple agents."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ResultAggregator")
    
    async def aggregate_session_results(self, session: ResearchSession) -> Dict[str, Any]:
        """
        Aggregate all results from a research session.
        
        Args:
            session: Research session to aggregate
            
        Returns:
            Dict containing aggregated results
        """
        # Calculate execution metrics
        execution_metrics = self._calculate_execution_metrics(session)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(session)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(session)
        
        aggregated = {
            "session_id": session.session_id,
            "query": session.original_query,
            "status": session.status.value,
            "session_state": session.session_state.value,
            "created_at": session.created_at.isoformat(),
            "completed_at": session.updated_at.isoformat(),
            "duration_seconds": (session.updated_at - session.created_at).total_seconds(),
            "sources": {
                "web_articles": len(session.extracted_articles),
                "academic_papers": len(session.academic_papers),
                "total_sources": len(session.get_all_sources()),
                "source_types": self._get_source_type_breakdown(session)
            },
            "analysis": {
                "insights_generated": len(session.insights),
                "average_confidence": self._calculate_average_confidence(session.insights),
                "confidence_distribution": self._get_confidence_distribution(session.insights),
                "topic_coverage": self._get_topic_coverage(session.insights)
            },
            "document": {
                "has_final_document": session.final_document is not None,
                "word_count": len(session.final_document.split()) if session.final_document else 0,
                "citations": len(session.citations),
                "sections": len(session.document_sections.get_all_sections()) if session.document_sections else 0
            },
            "execution": execution_metrics,
            "quality": quality_metrics,
            "performance": performance_metrics,
            "agent_contributions": self._get_agent_contributions(session)
        }
        
        return aggregated
    
    def _calculate_execution_metrics(self, session: ResearchSession) -> Dict[str, Any]:
        """Calculate detailed execution metrics."""
        tasks = session.plan.tasks
        executions = session.agent_executions
        
        return {
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "cancelled_tasks": len([t for t in tasks if t.status == TaskStatus.CANCELLED]),
            "success_rate": self._calculate_task_success_rate(tasks),
            "total_executions": len(executions),
            "successful_executions": len([e for e in executions if e.status == TaskStatus.COMPLETED]),
            "failed_executions": len([e for e in executions if e.status == TaskStatus.FAILED]),
            "execution_success_rate": self._calculate_execution_success_rate(executions),
            "average_execution_time": self._calculate_average_execution_time(executions),
            "total_execution_time": sum(e.execution_time_seconds or 0 for e in executions),
            "task_distribution": self._get_task_type_distribution(tasks)
        }
    
    def _calculate_quality_metrics(self, session: ResearchSession) -> Dict[str, Any]:
        """Calculate quality metrics for the research session."""
        return {
            "source_quality": {
                "average_confidence": self._calculate_average_source_confidence(session.extracted_articles),
                "high_confidence_sources": len([a for a in session.extracted_articles if a.confidence_score >= 0.8]),
                "low_confidence_sources": len([a for a in session.extracted_articles if a.confidence_score < 0.5])
            },
            "insight_quality": {
                "average_confidence": self._calculate_average_confidence(session.insights),
                "high_confidence_insights": len([i for i in session.insights if i.confidence_score >= 0.8]),
                "evidence_coverage": self._calculate_evidence_coverage(session.insights)
            },
            "citation_quality": {
                "total_citations": len(session.citations),
                "unique_sources_cited": len(set(c.source.url for c in session.citations)),
                "citation_density": len(session.citations) / max(1, len(session.final_document.split())) if session.final_document else 0
            }
        }
    
    def _calculate_performance_metrics(self, session: ResearchSession) -> Dict[str, Any]:
        """Calculate performance metrics."""
        return {
            "throughput": {
                "sources_per_minute": len(session.get_all_sources()) / max(1, (session.updated_at - session.created_at).total_seconds() / 60),
                "insights_per_minute": len(session.insights) / max(1, (session.updated_at - session.created_at).total_seconds() / 60),
                "tasks_per_minute": len([t for t in session.plan.tasks if t.status == TaskStatus.COMPLETED]) / max(1, (session.updated_at - session.created_at).total_seconds() / 60)
            },
            "efficiency": {
                "parallel_execution_ratio": self._calculate_parallel_execution_ratio(session),
                "resource_utilization": self._calculate_resource_utilization(session),
                "bottleneck_analysis": self._identify_bottlenecks(session)
            }
        }
    
    def _get_agent_contributions(self, session: ResearchSession) -> Dict[str, Any]:
        """Get breakdown of contributions by agent."""
        contributions = {}
        
        for execution in session.agent_executions:
            agent_name = execution.agent_name
            if agent_name not in contributions:
                contributions[agent_name] = {
                    "executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "total_execution_time": 0.0,
                    "average_execution_time": 0.0,
                    "success_rate": 0.0
                }
            
            contrib = contributions[agent_name]
            contrib["executions"] += 1
            
            if execution.status == TaskStatus.COMPLETED:
                contrib["successful_executions"] += 1
            elif execution.status == TaskStatus.FAILED:
                contrib["failed_executions"] += 1
            
            if execution.execution_time_seconds:
                contrib["total_execution_time"] += execution.execution_time_seconds
        
        # Calculate derived metrics
        for agent_name, contrib in contributions.items():
            if contrib["executions"] > 0:
                contrib["success_rate"] = contrib["successful_executions"] / contrib["executions"]
                contrib["average_execution_time"] = contrib["total_execution_time"] / contrib["executions"]
        
        return contributions
    
    def _get_source_type_breakdown(self, session: ResearchSession) -> Dict[str, int]:
        """Get breakdown of sources by type."""
        breakdown = {}
        for source in session.get_all_sources():
            source_type = source.source_type.value
            breakdown[source_type] = breakdown.get(source_type, 0) + 1
        return breakdown
    
    def _get_confidence_distribution(self, insights: List) -> Dict[str, int]:
        """Get distribution of confidence scores."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        for insight in insights:
            if insight.confidence_score >= 0.8:
                distribution["high"] += 1
            elif insight.confidence_score >= 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        return distribution
    
    def _get_topic_coverage(self, insights: List) -> List[str]:
        """Get list of topics covered by insights."""
        return list(set(insight.topic for insight in insights))
    
    def _calculate_task_success_rate(self, tasks: List[ResearchTask]) -> float:
        """Calculate task success rate."""
        if not tasks:
            return 0.0
        completed = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
        return completed / len(tasks)
    
    def _calculate_execution_success_rate(self, executions: List[AgentExecution]) -> float:
        """Calculate execution success rate."""
        if not executions:
            return 0.0
        successful = len([e for e in executions if e.status == TaskStatus.COMPLETED])
        return successful / len(executions)
    
    def _get_task_type_distribution(self, tasks: List[ResearchTask]) -> Dict[str, int]:
        """Get distribution of task types."""
        distribution = {}
        for task in tasks:
            task_type = task.task_type
            distribution[task_type] = distribution.get(task_type, 0) + 1
        return distribution
    
    def _calculate_average_source_confidence(self, articles: List) -> float:
        """Calculate average confidence of extracted articles."""
        if not articles:
            return 0.0
        return sum(article.confidence_score for article in articles) / len(articles)
    
    def _calculate_evidence_coverage(self, insights: List) -> float:
        """Calculate how well insights are supported by evidence."""
        if not insights:
            return 0.0
        total_evidence = sum(len(insight.supporting_evidence) for insight in insights)
        return total_evidence / len(insights)
    
    def _calculate_parallel_execution_ratio(self, session: ResearchSession) -> float:
        """Calculate ratio of parallel to sequential execution."""
        # This is a simplified calculation - in practice would analyze execution timelines
        total_tasks = len(session.plan.tasks)
        if total_tasks <= 1:
            return 0.0
        
        # Count tasks that could have run in parallel (no dependencies)
        parallel_tasks = len([t for t in session.plan.tasks if not t.dependencies])
        return parallel_tasks / total_tasks
    
    def _calculate_resource_utilization(self, session: ResearchSession) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        # Simplified calculation - would be more sophisticated in practice
        return {
            "agent_utilization": len(set(e.agent_name for e in session.agent_executions)) / max(1, len(session.agent_executions)),
            "time_utilization": 0.8  # Placeholder - would calculate actual utilization
        }
    
    def _identify_bottlenecks(self, session: ResearchSession) -> List[str]:
        """Identify potential bottlenecks in execution."""
        bottlenecks = []
        
        # Check for agents with high failure rates
        agent_failures = {}
        for execution in session.agent_executions:
            agent_name = execution.agent_name
            if agent_name not in agent_failures:
                agent_failures[agent_name] = {"total": 0, "failed": 0}
            
            agent_failures[agent_name]["total"] += 1
            if execution.status == TaskStatus.FAILED:
                agent_failures[agent_name]["failed"] += 1
        
        for agent_name, stats in agent_failures.items():
            if stats["total"] > 0 and stats["failed"] / stats["total"] > 0.3:
                bottlenecks.append(f"High failure rate for {agent_name} agent")
        
        # Check for long-running tasks
        long_tasks = [
            task for task in session.plan.tasks 
            if task.started_at and task.completed_at and 
            (task.completed_at - task.started_at).total_seconds() > 300  # 5 minutes
        ]
        
        if long_tasks:
            bottlenecks.append(f"{len(long_tasks)} tasks took longer than 5 minutes")
        
        return bottlenecks
    
    def _calculate_average_confidence(self, insights: List) -> float:
        """Calculate average confidence score from insights."""
        if not insights:
            return 0.0
        
        total_confidence = sum(insight.confidence_score for insight in insights)
        return total_confidence / len(insights)
    
    def _calculate_average_execution_time(self, executions: List[AgentExecution]) -> float:
        """Calculate average execution time from agent executions."""
        completed_executions = [e for e in executions if e.execution_time_seconds is not None]
        if not completed_executions:
            return 0.0
        
        total_time = sum(e.execution_time_seconds for e in completed_executions)
        return total_time / len(completed_executions)


class AgentOrchestrator:
    """
    Main orchestrator for managing multi-agent research workflows.
    
    Coordinates task execution, manages agent lifecycle, tracks progress,
    and aggregates results from multiple specialized agents.
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or OrchestrationConfig()
        self.logger = get_logger(__name__)
        
        # Core components
        self.progress_tracker = ProgressTracker()
        self.task_dispatcher = TaskDispatcher(self.config)
        self.result_aggregator = ResultAggregator()
        
        # Session management
        self._active_sessions: Dict[str, ResearchSession] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
    
    async def start_research_session(self, plan: ResearchPlan) -> ResearchSession:
        """
        Start a new research session with the given plan.
        
        Args:
            plan: Research plan to execute
            
        Returns:
            ResearchSession: Active research session
        """
        session = ResearchSession(
            session_id=plan.session_id,
            original_query=plan.query,
            plan=plan,
            status=TaskStatus.IN_PROGRESS,
            session_state=SessionState.RESEARCHING
        )
        
        # Store session
        self._active_sessions[session.session_id] = session
        self._session_locks[session.session_id] = asyncio.Lock()
        
        # Start background execution
        execution_task = asyncio.create_task(self._execute_session(session))
        self._background_tasks.add(execution_task)
        execution_task.add_done_callback(self._background_tasks.discard)
        
        self.logger.info(f"Started research session {session.session_id}")
        return session
    
    async def _execute_session(self, session: ResearchSession):
        """Execute all tasks in a research session."""
        try:
            session_lock = self._session_locks[session.session_id]
            
            while not self._shutdown_event.is_set():
                async with session_lock:
                    # Get ready tasks
                    ready_tasks = session.plan.get_ready_tasks()
                    
                    if not ready_tasks:
                        # Check if all tasks are complete
                        if all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
                               for task in session.plan.tasks):
                            break
                        
                        # Wait for running tasks to complete
                        await asyncio.sleep(1)
                        continue
                    
                    # Prioritize and execute ready tasks
                    if ready_tasks:
                        # Prioritize tasks for optimal execution order
                        prioritized_tasks = self.task_dispatcher.prioritize_tasks(ready_tasks)
                        
                        if self.config.enable_parallel_execution:
                            # Execute tasks in parallel with intelligent batching
                            batch_size = min(len(prioritized_tasks), self.config.max_concurrent_tasks)
                            
                            for i in range(0, len(prioritized_tasks), batch_size):
                                batch = prioritized_tasks[i:i + batch_size]
                                execution_tasks = []
                                
                                for task in batch:
                                    exec_task = asyncio.create_task(
                                        self._execute_task_in_session(task, session)
                                    )
                                    execution_tasks.append(exec_task)
                                
                                if execution_tasks:
                                    # Wait for batch to complete before starting next batch
                                    results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                                    
                                    # Log any exceptions
                                    for i, result in enumerate(results):
                                        if isinstance(result, Exception):
                                            self.logger.error(f"Task execution failed: {result}")
                        else:
                            # Execute tasks sequentially in priority order
                            for task in prioritized_tasks:
                                await self._execute_task_in_session(task, session)
                    
                    # Update progress
                    await self._update_session_progress(session)
            
            # Finalize session
            await self._finalize_session(session)
            
        except Exception as e:
            self.logger.error(f"Session execution failed for {session.session_id}: {e}")
            session.status = TaskStatus.FAILED
            session.session_state = SessionState.FAILED
    
    async def _execute_task_in_session(self, task: ResearchTask, session: ResearchSession):
        """Execute a single task within a session context."""
        start_time = datetime.now()
        
        try:
            execution = await self.task_dispatcher.dispatch_task(task, session)
            session.add_agent_execution(execution)
            
            # Update session data based on task results
            await self._update_session_data(session, task, execution)
            
            # Update task metrics for future prioritization
            execution_time = (datetime.now() - start_time).total_seconds()
            success = execution.status == TaskStatus.COMPLETED
            
            if task.assigned_agent:
                self.task_dispatcher.update_task_metrics(
                    task.assigned_agent, 
                    success, 
                    execution_time
                )
            
        except Exception as e:
            self.logger.error(f"Task execution failed for {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
            # Update metrics for failed execution
            execution_time = (datetime.now() - start_time).total_seconds()
            if task.assigned_agent:
                self.task_dispatcher.update_task_metrics(
                    task.assigned_agent, 
                    False, 
                    execution_time
                )
    
    async def _update_session_data(self, session: ResearchSession, task: ResearchTask, execution: AgentExecution):
        """Update session data based on task execution results."""
        if execution.status != TaskStatus.COMPLETED or not execution.output_data:
            return
        
        result_data = execution.output_data.get("result", {})
        
        # Update session based on task type
        if task.task_type == "web_search" and isinstance(result_data, list):
            # Add extracted articles
            for article_data in result_data:
                if hasattr(article_data, 'source'):  # Assuming ExtractedArticle objects
                    session.extracted_articles.append(article_data)
        
        elif task.task_type == "academic_search" and isinstance(result_data, list):
            # Add academic papers
            for paper_data in result_data:
                if hasattr(paper_data, 'title'):  # Assuming AcademicPaper objects
                    session.academic_papers.append(paper_data)
        
        elif task.task_type == "content_analysis" and isinstance(result_data, list):
            # Add insights
            for insight_data in result_data:
                if hasattr(insight_data, 'topic'):  # Assuming Insight objects
                    session.insights.append(insight_data)
        
        elif task.task_type == "document_generation" and isinstance(result_data, str):
            # Set final document
            session.final_document = result_data
        
        elif task.task_type == "citation_formatting" and isinstance(result_data, list):
            # Add citations
            for citation_data in result_data:
                if hasattr(citation_data, 'citation_id'):  # Assuming Citation objects
                    session.citations.append(citation_data)
    
    async def _update_session_progress(self, session: ResearchSession):
        """Update progress tracking for a session."""
        completed_tasks = len([t for t in session.plan.tasks if t.status == TaskStatus.COMPLETED])
        total_tasks = len(session.plan.tasks)
        progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        progress_data = {
            "session_id": session.session_id,
            "progress_percentage": progress_percentage,
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "current_state": session.session_state.value,
            "sources_found": len(session.get_all_sources()),
            "insights_generated": len(session.insights),
            "has_final_document": session.final_document is not None
        }
        
        self.progress_tracker.update_progress(session.session_id, progress_data)
    
    async def _finalize_session(self, session: ResearchSession):
        """Finalize a completed research session."""
        # Update session status
        if all(task.status == TaskStatus.COMPLETED for task in session.plan.tasks):
            session.status = TaskStatus.COMPLETED
            session.session_state = SessionState.COMPLETED
        else:
            session.status = TaskStatus.FAILED
            session.session_state = SessionState.FAILED
        
        # Update metrics
        session.metrics.update_from_session(session)
        
        # Final progress update
        await self._update_session_progress(session)
        
        self.logger.info(f"Finalized session {session.session_id} with status {session.status}")
    
    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get an active research session by ID."""
        return self._active_sessions.get(session_id)
    
    def list_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self._active_sessions.keys())
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active research session."""
        session = self._active_sessions.get(session_id)
        if not session:
            return False
        
        # Cancel all active task executions
        active_executions = self.task_dispatcher.get_active_executions()
        for execution in active_executions:
            if execution.execution.task_id in [t.task_id for t in session.plan.tasks]:
                self.task_dispatcher.cancel_task(execution.execution.execution_id)
        
        # Update session status
        session.status = TaskStatus.CANCELLED
        session.session_state = SessionState.CANCELLED
        
        self.logger.info(f"Cancelled session {session_id}")
        return True
    
    async def get_session_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a session."""
        return self.progress_tracker.get_progress(session_id)
    
    def register_progress_callback(self, session_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for progress updates."""
        self.progress_tracker.register_progress_callback(session_id, callback)
    
    async def get_session_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get aggregated results for a session."""
        session = self._active_sessions.get(session_id)
        if not session:
            return None
        
        return await self.result_aggregator.aggregate_session_results(session)
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get overall orchestrator status and metrics."""
        return {
            "active_sessions": len(self._active_sessions),
            "background_tasks": len(self._background_tasks),
            "load_balancing": self.task_dispatcher.get_load_balancing_info(),
            "session_states": {
                session_id: {
                    "status": session.status.value,
                    "state": session.session_state.value,
                    "progress": session.calculate_progress_percentage(),
                    "tasks": {
                        "total": len(session.plan.tasks),
                        "completed": len([t for t in session.plan.tasks if t.status == TaskStatus.COMPLETED]),
                        "failed": len([t for t in session.plan.tasks if t.status == TaskStatus.FAILED]),
                        "in_progress": len([t for t in session.plan.tasks if t.status == TaskStatus.IN_PROGRESS])
                    }
                }
                for session_id, session in self._active_sessions.items()
            }
        }
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a running research session."""
        session = self._active_sessions.get(session_id)
        if not session or session.status != TaskStatus.IN_PROGRESS:
            return False
        
        # Mark session as paused (using cancelled status for simplicity)
        session.status = TaskStatus.CANCELLED
        session.session_state = SessionState.CANCELLED
        
        # Cancel active executions for this session
        active_executions = self.task_dispatcher.get_active_executions()
        for execution in active_executions:
            if execution.execution.task_id in [t.task_id for t in session.plan.tasks]:
                self.task_dispatcher.cancel_task(execution.execution.execution_id)
        
        self.logger.info(f"Paused session {session_id}")
        return True
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused research session."""
        session = self._active_sessions.get(session_id)
        if not session or session.status != TaskStatus.CANCELLED:
            return False
        
        # Reset session status
        session.status = TaskStatus.IN_PROGRESS
        session.session_state = SessionState.RESEARCHING
        
        # Reset cancelled tasks to pending
        for task in session.plan.tasks:
            if task.status == TaskStatus.CANCELLED:
                task.status = TaskStatus.PENDING
        
        # Restart background execution
        execution_task = asyncio.create_task(self._execute_session(session))
        self._background_tasks.add(execution_task)
        execution_task.add_done_callback(self._background_tasks.discard)
        
        self.logger.info(f"Resumed session {session_id}")
        return True
    
    def get_session_diagnostics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed diagnostics for a session."""
        session = self._active_sessions.get(session_id)
        if not session:
            return None
        
        # Analyze task dependencies
        dependency_graph = {}
        for task in session.plan.tasks:
            dependency_graph[task.task_id] = {
                "dependencies": task.dependencies,
                "status": task.status.value,
                "assigned_agent": task.assigned_agent,
                "execution_time": None,
                "error": task.error_message
            }
            
            if task.started_at and task.completed_at:
                dependency_graph[task.task_id]["execution_time"] = (
                    task.completed_at - task.started_at
                ).total_seconds()
        
        # Identify potential issues
        issues = []
        
        # Check for failed tasks
        failed_tasks = [t for t in session.plan.tasks if t.status == TaskStatus.FAILED]
        if failed_tasks:
            issues.append(f"{len(failed_tasks)} tasks failed")
        
        # Check for long-running tasks
        long_tasks = [
            t for t in session.plan.tasks 
            if t.status == TaskStatus.IN_PROGRESS and t.started_at and
            (datetime.now() - t.started_at).total_seconds() > 300
        ]
        if long_tasks:
            issues.append(f"{len(long_tasks)} tasks running longer than 5 minutes")
        
        # Check for blocked tasks
        blocked_tasks = []
        for task in session.plan.tasks:
            if task.status == TaskStatus.PENDING and task.dependencies:
                failed_deps = [
                    dep for dep in task.dependencies
                    if any(t.task_id == dep and t.status == TaskStatus.FAILED for t in session.plan.tasks)
                ]
                if failed_deps:
                    blocked_tasks.append(task.task_id)
        
        if blocked_tasks:
            issues.append(f"{len(blocked_tasks)} tasks blocked by failed dependencies")
        
        return {
            "session_id": session_id,
            "dependency_graph": dependency_graph,
            "issues": issues,
            "performance": {
                "total_execution_time": (session.updated_at - session.created_at).total_seconds(),
                "agent_utilization": len(set(e.agent_name for e in session.agent_executions)),
                "parallel_efficiency": self._calculate_parallel_efficiency(session)
            }
        }
    
    def _calculate_parallel_efficiency(self, session: ResearchSession) -> float:
        """Calculate how efficiently parallel execution was used."""
        if not session.agent_executions:
            return 0.0
        
        # Simple calculation - could be more sophisticated
        total_execution_time = sum(
            e.execution_time_seconds or 0 for e in session.agent_executions
        )
        wall_clock_time = (session.updated_at - session.created_at).total_seconds()
        
        if wall_clock_time == 0:
            return 0.0
        
        # Efficiency is the ratio of total work done to wall clock time
        # Values > 1.0 indicate good parallelization
        return total_execution_time / wall_clock_time
    
    async def shutdown(self):
        """Shutdown the orchestrator and clean up resources."""
        self.logger.info("Shutting down orchestrator")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clean up sessions
        for session_id in list(self._active_sessions.keys()):
            self.progress_tracker.cleanup_session(session_id)
        
        self._active_sessions.clear()
        self._session_locks.clear()
        
        self.logger.info("Orchestrator shutdown complete")


# Global orchestrator instance
orchestrator = AgentOrchestrator()