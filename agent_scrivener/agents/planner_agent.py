"""
Planner Agent for Agent Scrivener.

The Planner Agent is responsible for:
- Analyzing research queries and decomposing them into executable tasks
- Creating directed acyclic graphs (DAGs) for task execution
- Managing task dependencies and orchestration logic
- Integrating with Strands SDK for workflow management
"""

import uuid
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import logging

from ..agents.base import BaseAgent, AgentResult
from ..models.core import (
    ResearchPlan, ResearchTask, TaskStatus, ResearchSession, 
    SessionState, WorkflowStep, AgentExecution
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TaskGraph:
    """
    Represents a directed acyclic graph of research tasks.
    """
    
    def __init__(self):
        self.nodes: Dict[str, ResearchTask] = {}
        self.edges: Dict[str, Set[str]] = {}  # task_id -> set of dependent task_ids
    
    def add_task(self, task: ResearchTask) -> None:
        """Add a task to the graph."""
        self.nodes[task.task_id] = task
        if task.task_id not in self.edges:
            self.edges[task.task_id] = set()
    
    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Add a dependency relationship between tasks."""
        if depends_on not in self.nodes:
            raise ValueError(f"Dependency task {depends_on} not found in graph")
        if task_id not in self.nodes:
            raise ValueError(f"Task {task_id} not found in graph")
        
        self.edges[depends_on].add(task_id)
        
        # Update the task's dependencies list
        if depends_on not in self.nodes[task_id].dependencies:
            self.nodes[task_id].dependencies.append(depends_on)
    
    def get_ready_tasks(self) -> List[ResearchTask]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        completed_tasks = {
            task_id for task_id, task in self.nodes.items() 
            if task.status == TaskStatus.COMPLETED
        }
        
        ready_tasks = []
        for task_id, task in self.nodes.items():
            if task.status == TaskStatus.PENDING:
                if all(dep_id in completed_tasks for dep_id in task.dependencies):
                    ready_tasks.append(task)
        
        return ready_tasks
    
    def get_task_by_id(self, task_id: str) -> Optional[ResearchTask]:
        """Get a task by its ID."""
        return self.nodes.get(task_id)
    
    def validate_dag(self) -> bool:
        """Validate that the graph is a valid DAG (no cycles)."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in self.edges.get(node_id, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    return False
        
        return True
    
    def get_execution_order(self) -> List[List[str]]:
        """Get tasks in topological order, grouped by execution level."""
        in_degree = {task_id: 0 for task_id in self.nodes}
        
        # Calculate in-degrees
        for task_id in self.nodes:
            for dependent in self.edges.get(task_id, set()):
                in_degree[dependent] += 1
        
        # Group tasks by execution level
        levels = []
        remaining_tasks = set(self.nodes.keys())
        
        while remaining_tasks:
            # Find tasks with no dependencies at current level
            current_level = [
                task_id for task_id in remaining_tasks 
                if in_degree[task_id] == 0
            ]
            
            if not current_level:
                raise ValueError("Circular dependency detected in task graph")
            
            levels.append(current_level)
            
            # Remove current level tasks and update in-degrees
            for task_id in current_level:
                remaining_tasks.remove(task_id)
                for dependent in self.edges.get(task_id, set()):
                    in_degree[dependent] -= 1
        
        return levels


class QueryAnalyzer:
    """
    Analyzes research queries to determine task decomposition strategy.
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.QueryAnalyzer")
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a research query to determine task requirements.
        
        Args:
            query: Research query string
            
        Returns:
            Dict containing analysis results
        """
        analysis = {
            "query_type": self._classify_query_type(query),
            "complexity": self._assess_complexity(query),
            "required_sources": self._identify_required_sources(query),
            "estimated_duration": self._estimate_duration(query),
            "key_topics": self._extract_key_topics(query),
            "analysis_requirements": self._determine_analysis_needs(query)
        }
        
        self.logger.info(f"Query analysis completed: {analysis['query_type']} complexity={analysis['complexity']}")
        return analysis
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of research query."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["compare", "versus", "vs", "difference"]):
            return "comparative"
        elif any(term in query_lower for term in ["trend", "over time", "historical", "evolution"]):
            return "temporal"
        elif any(term in query_lower for term in ["how", "what", "why", "explain"]):
            return "explanatory"
        elif any(term in query_lower for term in ["review", "survey", "overview", "summary"]):
            return "survey"
        elif any(term in query_lower for term in ["analysis", "analyze", "examine", "investigate"]):
            return "analytical"
        else:
            return "general"
    
    def _assess_complexity(self, query: str) -> str:
        """Assess the complexity level of the query."""
        word_count = len(query.split())
        
        # Count complex indicators
        complex_indicators = [
            "multiple", "various", "different", "compare", "analyze", 
            "relationship", "impact", "effect", "correlation", "causation"
        ]
        complexity_score = sum(1 for indicator in complex_indicators if indicator in query.lower())
        
        if word_count > 50 or complexity_score >= 3:
            return "high"
        elif word_count > 15 or complexity_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _identify_required_sources(self, query: str) -> List[str]:
        """Identify what types of sources are needed."""
        sources = ["web"]  # Always include web sources
        
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["research", "study", "paper", "academic", "journal"]):
            sources.append("academic")
        
        if any(term in query_lower for term in ["data", "statistics", "numbers", "metrics"]):
            sources.append("database")
        
        return sources
    
    def _estimate_duration(self, query: str) -> int:
        """Estimate research duration in minutes."""
        complexity = self._assess_complexity(query)
        required_sources = self._identify_required_sources(query)
        
        base_duration = {
            "low": 15,
            "medium": 30,
            "high": 60
        }[complexity]
        
        # Add time for additional source types
        source_multiplier = len(required_sources) * 0.5
        
        return int(base_duration * (1 + source_multiplier))
    
    def _extract_key_topics(self, query: str) -> List[str]:
        """Extract key topics from the query."""
        # Simple keyword extraction - in production, would use NLP
        import re
        
        # Remove punctuation and convert to lowercase
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        words = clean_query.split()
        
        # Filter out common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "about", "what", "how", "why", "when", "where",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had"
        }
        
        topics = [word for word in words if word not in stop_words and len(word) > 3]
        return topics[:10]  # Return top 10 topics
    
    def _determine_analysis_needs(self, query: str) -> List[str]:
        """Determine what types of analysis are needed."""
        analysis_needs = []
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["trend", "pattern", "change", "over time"]):
            analysis_needs.append("temporal_analysis")
        
        if any(term in query_lower for term in ["sentiment", "opinion", "feeling", "attitude"]):
            analysis_needs.append("sentiment_analysis")
        
        if any(term in query_lower for term in ["topic", "theme", "category", "classify"]):
            analysis_needs.append("topic_modeling")
        
        if any(term in query_lower for term in ["entity", "person", "organization", "location"]):
            analysis_needs.append("named_entity_recognition")
        
        if any(term in query_lower for term in ["statistic", "correlation", "relationship", "data"]):
            analysis_needs.append("statistical_analysis")
        
        # Default analysis if none specified
        if not analysis_needs:
            analysis_needs = ["topic_modeling", "named_entity_recognition"]
        
        return analysis_needs


class PlannerAgent(BaseAgent):
    """
    Planner Agent responsible for query analysis and task orchestration.
    """
    
    def __init__(self):
        super().__init__("planner")
        self.query_analyzer = QueryAnalyzer()
        self.logger = get_logger(__name__)
    
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the planner agent to create a research plan.
        
        Args:
            query (str): Research query to plan for
            session_id (str, optional): Existing session ID
            
        Returns:
            AgentResult: Contains ResearchPlan in data field
        """
        return await self._execute_with_timing(self._create_research_plan, **kwargs)
    
    async def _create_research_plan(self, query: str, session_id: Optional[str] = None) -> ResearchPlan:
        """
        Create a comprehensive research plan from a query.
        
        Args:
            query: Research query string
            session_id: Optional existing session ID
            
        Returns:
            ResearchPlan: Complete execution plan
        """
        self.validate_input({"query": query}, ["query"])
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Analyze the query
        analysis = await self.query_analyzer.analyze_query(query)
        
        # Create task graph
        task_graph = await self._create_task_graph(query, analysis)
        
        # Validate the DAG
        if not task_graph.validate_dag():
            raise ValueError("Generated task graph contains cycles")
        
        # Create research plan
        plan = ResearchPlan(
            query=query,
            session_id=session_id,
            tasks=list(task_graph.nodes.values()),
            estimated_duration_minutes=analysis["estimated_duration"]
        )
        
        self.logger.info(f"Created research plan with {len(plan.tasks)} tasks for session {session_id}")
        return plan
    
    async def _create_task_graph(self, query: str, analysis: Dict[str, Any]) -> TaskGraph:
        """
        Create a task graph based on query analysis.
        
        Args:
            query: Original research query
            analysis: Query analysis results
            
        Returns:
            TaskGraph: Complete task dependency graph
        """
        graph = TaskGraph()
        
        # Create tasks based on analysis
        tasks = []
        
        # 1. Web research task (always needed)
        web_task = ResearchTask(
            task_id="web_research",
            task_type="web_search",
            description=f"Search web sources for information about: {query}",
            parameters={
                "query": query,
                "max_sources": 10,
                "topics": analysis["key_topics"]
            },
            assigned_agent="research"
        )
        tasks.append(web_task)
        graph.add_task(web_task)
        
        # 2. Academic research task (if needed)
        if "academic" in analysis["required_sources"]:
            academic_task = ResearchTask(
                task_id="academic_research",
                task_type="academic_search",
                description=f"Search academic databases for papers about: {query}",
                parameters={
                    "query": query,
                    "databases": ["arxiv", "pubmed", "semantic_scholar"],
                    "topics": analysis["key_topics"]
                },
                assigned_agent="api"
            )
            tasks.append(academic_task)
            graph.add_task(academic_task)
        
        # 3. Data analysis task (depends on research tasks)
        analysis_task = ResearchTask(
            task_id="data_analysis",
            task_type="content_analysis",
            description="Analyze collected research data for insights and patterns",
            parameters={
                "analysis_types": analysis["analysis_requirements"],
                "topics": analysis["key_topics"]
            },
            dependencies=["web_research"] + (["academic_research"] if "academic" in analysis["required_sources"] else []),
            assigned_agent="analysis"
        )
        tasks.append(analysis_task)
        graph.add_task(analysis_task)
        
        # Add dependencies
        graph.add_dependency("data_analysis", "web_research")
        if "academic" in analysis["required_sources"]:
            graph.add_dependency("data_analysis", "academic_research")
        
        # 4. Content drafting task (depends on analysis)
        drafting_task = ResearchTask(
            task_id="content_drafting",
            task_type="document_generation",
            description="Generate structured document from analyzed insights",
            parameters={
                "query": query,
                "document_type": analysis["query_type"],
                "complexity": analysis["complexity"]
            },
            dependencies=["data_analysis"],
            assigned_agent="drafting"
        )
        tasks.append(drafting_task)
        graph.add_task(drafting_task)
        graph.add_dependency("content_drafting", "data_analysis")
        
        # 5. Citation management task (depends on all research tasks)
        citation_task = ResearchTask(
            task_id="citation_management",
            task_type="citation_formatting",
            description="Format citations and create bibliography",
            parameters={
                "citation_style": "APA",
                "verify_urls": True
            },
            dependencies=["web_research", "content_drafting"] + (["academic_research"] if "academic" in analysis["required_sources"] else []),
            assigned_agent="citation"
        )
        tasks.append(citation_task)
        graph.add_task(citation_task)
        graph.add_dependency("citation_management", "web_research")
        graph.add_dependency("citation_management", "content_drafting")
        if "academic" in analysis["required_sources"]:
            graph.add_dependency("citation_management", "academic_research")
        
        self.logger.info(f"Created task graph with {len(tasks)} tasks")
        return graph
    
    async def create_workflow_steps(self, plan: ResearchPlan) -> List[WorkflowStep]:
        """
        Convert a research plan into workflow steps.
        
        Args:
            plan: Research plan to convert
            
        Returns:
            List[WorkflowStep]: Ordered workflow steps
        """
        # Create task graph for execution ordering
        graph = TaskGraph()
        for task in plan.tasks:
            graph.add_task(task)
        
        # Add dependencies
        for task in plan.tasks:
            for dep_id in task.dependencies:
                graph.add_dependency(task.task_id, dep_id)
        
        # Get execution levels
        execution_levels = graph.get_execution_order()
        
        workflow_steps = []
        for level_idx, task_ids in enumerate(execution_levels):
            for task_id in task_ids:
                task = graph.get_task_by_id(task_id)
                if task:
                    step = WorkflowStep(
                        step_id=f"step_{level_idx}_{task_id}",
                        step_name=task.description,
                        description=f"Execute {task.task_type} task: {task.description}",
                        required_inputs=[f"output_{dep}" for dep in task.dependencies],
                        expected_outputs=[f"output_{task_id}"],
                        estimated_duration_minutes=self._estimate_task_duration(task)
                    )
                    workflow_steps.append(step)
        
        return workflow_steps
    
    def _estimate_task_duration(self, task: ResearchTask) -> int:
        """Estimate duration for a specific task."""
        base_durations = {
            "web_search": 10,
            "academic_search": 8,
            "content_analysis": 15,
            "document_generation": 12,
            "citation_formatting": 5
        }
        
        return base_durations.get(task.task_type, 10)
    
    async def update_plan_progress(self, plan: ResearchPlan, task_id: str, status: TaskStatus, result: Optional[Dict[str, Any]] = None) -> ResearchPlan:
        """
        Update the progress of a task in the research plan.
        
        Args:
            plan: Research plan to update
            task_id: ID of the task to update
            status: New status for the task
            result: Optional result data
            
        Returns:
            ResearchPlan: Updated plan
        """
        task = plan.get_task_by_id(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in plan")
        
        task.status = status
        if status == TaskStatus.IN_PROGRESS:
            task.started_at = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task.completed_at = datetime.now()
            if result:
                task.result = result
        
        # Update overall plan status
        if all(task.status == TaskStatus.COMPLETED for task in plan.tasks):
            plan.status = TaskStatus.COMPLETED
        elif any(task.status == TaskStatus.FAILED for task in plan.tasks):
            plan.status = TaskStatus.FAILED
        elif any(task.status == TaskStatus.IN_PROGRESS for task in plan.tasks):
            plan.status = TaskStatus.IN_PROGRESS
        
        return plan