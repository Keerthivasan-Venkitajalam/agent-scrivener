"""
Integration test framework for Agent Scrivener.

This module provides a comprehensive framework for testing the entire research pipeline
with mocked external services, performance benchmarking, and consistent test data.
"""

import pytest
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from agent_scrivener.models.core import (
    ResearchPlan, ResearchTask, ResearchSession, Source, SourceType,
    ExtractedArticle, AcademicPaper, Insight, TaskStatus, SessionState
)
from agent_scrivener.orchestration.orchestrator import AgentOrchestrator, OrchestrationConfig
from agent_scrivener.agents.base import BaseAgent, AgentResult
from agent_scrivener.utils.error_handler import ErrorHandler
from agent_scrivener.utils.monitoring import SystemMonitor


@dataclass
class TestMetrics:
    """Metrics collected during test execution."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    task_completion_times: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    success_rate: float = 0.0
    throughput_requests_per_second: Optional[float] = None


@dataclass
class MockServiceConfig:
    """Configuration for mock external services."""
    web_search_delay: float = 0.1
    api_query_delay: float = 0.15
    analysis_delay: float = 0.2
    failure_rate: float = 0.0  # 0.0 = no failures, 1.0 = all failures
    rate_limit_enabled: bool = False
    rate_limit_requests_per_second: int = 10


class MockExternalServices:
    """Mock external services for integration testing."""
    
    def __init__(self, config: MockServiceConfig):
        self.config = config
        self._request_counts = {}
        self._last_request_time = {}
    
    async def mock_web_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Mock web search functionality."""
        await self._simulate_delay(self.config.web_search_delay)
        await self._check_rate_limit("web_search")
        
        if self._should_fail():
            raise Exception("Mock web search failure")
        
        # Generate mock search results
        results = []
        for i in range(min(max_results, 5)):
            results.append({
                "url": f"https://example{i+1}.com/article",
                "title": f"Mock Article {i+1} for '{query}'",
                "snippet": f"This is a mock snippet for article {i+1} about {query}",
                "domain": f"example{i+1}.com"
            })
        
        return results
    
    async def mock_content_extraction(self, url: str) -> Dict[str, Any]:
        """Mock content extraction from URLs."""
        await self._simulate_delay(self.config.web_search_delay * 0.5)
        
        if self._should_fail():
            raise Exception(f"Mock content extraction failure for {url}")
        
        return {
            "url": url,
            "title": f"Extracted Title from {url}",
            "content": f"This is mock extracted content from {url}. " * 50,
            "author": "Mock Author",
            "publication_date": datetime.now().isoformat(),
            "word_count": 500,
            "confidence_score": 0.85
        }
    
    async def mock_academic_search(self, query: str, database: str = "arxiv") -> List[Dict[str, Any]]:
        """Mock academic database search."""
        await self._simulate_delay(self.config.api_query_delay)
        await self._check_rate_limit(f"academic_{database}")
        
        if self._should_fail():
            raise Exception(f"Mock {database} search failure")
        
        # Generate mock academic papers
        papers = []
        for i in range(3):
            papers.append({
                "title": f"Mock Academic Paper {i+1}: {query}",
                "authors": [f"Author {j+1}" for j in range(2)],
                "abstract": f"This is a mock abstract for paper {i+1} about {query}. " * 10,
                "publication_year": 2023 - i,
                "doi": f"10.1000/mock.{i+1}",
                "database_source": database,
                "citation_count": 50 - i * 10,
                "url": f"https://{database}.org/abs/mock{i+1}"
            })
        
        return papers
    
    async def mock_nlp_analysis(self, text: str) -> Dict[str, Any]:
        """Mock NLP analysis functionality."""
        await self._simulate_delay(self.config.analysis_delay)
        
        if self._should_fail():
            raise Exception("Mock NLP analysis failure")
        
        # Generate mock analysis results
        return {
            "entities": [
                {"text": "Machine Learning", "label": "TECHNOLOGY", "confidence": 0.9},
                {"text": "Research", "label": "ACTIVITY", "confidence": 0.8},
                {"text": "Data Science", "label": "FIELD", "confidence": 0.85}
            ],
            "topics": [
                {"topic_id": 0, "words": ["machine", "learning", "algorithm"], "weight": 0.4},
                {"topic_id": 1, "words": ["data", "analysis", "research"], "weight": 0.35},
                {"topic_id": 2, "words": ["artificial", "intelligence", "model"], "weight": 0.25}
            ],
            "sentiment": {"polarity": 0.1, "subjectivity": 0.3},
            "key_phrases": ["machine learning", "data analysis", "research methodology"],
            "summary": "Mock summary of the analyzed text focusing on key concepts."
        }
    
    async def _simulate_delay(self, delay: float):
        """Simulate processing delay."""
        if delay > 0:
            await asyncio.sleep(delay)
    
    def _should_fail(self) -> bool:
        """Determine if operation should fail based on failure rate."""
        import random
        return random.random() < self.config.failure_rate
    
    async def _check_rate_limit(self, service: str):
        """Check and enforce rate limiting."""
        if not self.config.rate_limit_enabled:
            return
        
        now = time.time()
        if service not in self._request_counts:
            self._request_counts[service] = 0
            self._last_request_time[service] = now
        
        # Reset counter if more than 1 second has passed
        if now - self._last_request_time[service] >= 1.0:
            self._request_counts[service] = 0
            self._last_request_time[service] = now
        
        # Check rate limit
        if self._request_counts[service] >= self.config.rate_limit_requests_per_second:
            sleep_time = 1.0 - (now - self._last_request_time[service])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self._request_counts[service] = 0
            self._last_request_time[service] = time.time()
        
        self._request_counts[service] += 1


class IntegrationTestFramework:
    """Framework for running comprehensive integration tests."""
    
    def __init__(self, mock_config: MockServiceConfig = None):
        self.mock_config = mock_config or MockServiceConfig()
        self.mock_services = MockExternalServices(self.mock_config)
        self.metrics = TestMetrics()
        self.orchestrator = None
        self._patches = []
    
    @asynccontextmanager
    async def test_environment(self):
        """Context manager for setting up test environment."""
        # Start metrics collection
        self.metrics.start_time = datetime.now()
        
        # Set up orchestrator
        config = OrchestrationConfig(
            max_concurrent_tasks=5,
            task_timeout_seconds=30,
            progress_update_interval_seconds=0.5,
            enable_parallel_execution=True
        )
        self.orchestrator = AgentOrchestrator(config)
        
        # Apply patches for external services
        self._apply_service_patches()
        
        try:
            yield self
        finally:
            # Clean up
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            # Remove patches
            for patch_obj in self._patches:
                patch_obj.stop()
            self._patches.clear()
            
            # Finalize metrics
            self.metrics.end_time = datetime.now()
            self.metrics.execution_time_ms = int(
                (self.metrics.end_time - self.metrics.start_time).total_seconds() * 1000
            )
    
    def _apply_service_patches(self):
        """Apply patches to mock external services."""
        # Mock browser tool
        browser_patch = patch('agent_scrivener.tools.browser_wrapper.BrowserToolWrapper')
        mock_browser = browser_patch.start()
        mock_browser.return_value.search_web.side_effect = self.mock_services.mock_web_search
        mock_browser.return_value.extract_content.side_effect = self.mock_services.mock_content_extraction
        self._patches.append(browser_patch)
        
        # Mock gateway tool
        gateway_patch = patch('agent_scrivener.tools.gateway_wrapper.GatewayWrapper')
        mock_gateway = gateway_patch.start()
        mock_gateway.return_value.query_academic_database.side_effect = self.mock_services.mock_academic_search
        self._patches.append(gateway_patch)
        
        # Mock code interpreter
        interpreter_patch = patch('agent_scrivener.tools.code_interpreter_wrapper.CodeInterpreterWrapper')
        mock_interpreter = interpreter_patch.start()
        mock_interpreter.return_value.analyze_text.side_effect = self.mock_services.mock_nlp_analysis
        self._patches.append(interpreter_patch)
    
    async def run_end_to_end_test(self, research_query: str) -> Dict[str, Any]:
        """Run complete end-to-end research pipeline test."""
        session_id = str(uuid.uuid4())
        
        # Create research plan
        plan = self._create_comprehensive_research_plan(research_query, session_id)
        
        # Execute research session
        session = await self.orchestrator.start_research_session(plan)
        
        # Monitor progress
        progress_updates = []
        def progress_callback(progress_data):
            progress_updates.append({
                "timestamp": datetime.now(),
                "data": progress_data.copy()
            })
        
        self.orchestrator.register_progress_callback(session_id, progress_callback)
        
        # Wait for completion
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        while session.status == TaskStatus.IN_PROGRESS:
            if time.time() - start_time > max_wait_time:
                break
            await asyncio.sleep(0.5)
            session = self.orchestrator.get_session(session_id)
        
        # Collect results
        results = await self.orchestrator.get_session_results(session_id)
        
        # Update metrics
        self._update_test_metrics(session, progress_updates)
        
        return {
            "session": session,
            "results": results,
            "progress_updates": progress_updates,
            "metrics": self.metrics
        }
    
    def _create_comprehensive_research_plan(self, query: str, session_id: str) -> ResearchPlan:
        """Create a comprehensive research plan for testing."""
        tasks = [
            ResearchTask(
                task_id="web_research",
                task_type="web_search",
                description="Search web sources for information",
                parameters={"query": query, "max_sources": 10},
                assigned_agent="research"
            ),
            ResearchTask(
                task_id="arxiv_search",
                task_type="academic_search",
                description="Search arXiv for academic papers",
                parameters={"query": query, "database": "arxiv", "max_papers": 5},
                assigned_agent="api"
            ),
            ResearchTask(
                task_id="pubmed_search",
                task_type="academic_search",
                description="Search PubMed for medical papers",
                parameters={"query": query, "database": "pubmed", "max_papers": 5},
                assigned_agent="api"
            ),
            ResearchTask(
                task_id="content_analysis",
                task_type="content_analysis",
                description="Analyze collected content",
                parameters={"analysis_types": ["ner", "topic_modeling", "sentiment"]},
                dependencies=["web_research", "arxiv_search", "pubmed_search"],
                assigned_agent="analysis"
            ),
            ResearchTask(
                task_id="insight_generation",
                task_type="insight_synthesis",
                description="Generate insights from analysis",
                parameters={"min_confidence": 0.7},
                dependencies=["content_analysis"],
                assigned_agent="analysis"
            ),
            ResearchTask(
                task_id="document_drafting",
                task_type="document_generation",
                description="Generate research document",
                parameters={"document_type": "comprehensive_report", "sections": ["intro", "methodology", "findings", "conclusion"]},
                dependencies=["insight_generation"],
                assigned_agent="drafting"
            ),
            ResearchTask(
                task_id="citation_formatting",
                task_type="citation_management",
                description="Format citations and bibliography",
                parameters={"citation_style": "APA", "include_urls": True},
                dependencies=["web_research", "arxiv_search", "pubmed_search", "document_drafting"],
                assigned_agent="citation"
            )
        ]
        
        return ResearchPlan(
            query=query,
            session_id=session_id,
            tasks=tasks,
            estimated_duration_minutes=45
        )
    
    def _update_test_metrics(self, session: ResearchSession, progress_updates: List[Dict]):
        """Update test metrics based on session results."""
        completed_tasks = [task for task in session.plan.tasks if task.status == TaskStatus.COMPLETED]
        failed_tasks = [task for task in session.plan.tasks if task.status == TaskStatus.FAILED]
        
        self.metrics.success_rate = len(completed_tasks) / len(session.plan.tasks) * 100
        self.metrics.error_count = len(failed_tasks)
        
        # Calculate task completion times
        for task in completed_tasks:
            if task.started_at and task.completed_at:
                completion_time = int((task.completed_at - task.started_at).total_seconds() * 1000)
                self.metrics.task_completion_times[task.task_id] = completion_time
        
        # Calculate throughput if we have progress updates
        if progress_updates:
            total_time_seconds = (progress_updates[-1]["timestamp"] - progress_updates[0]["timestamp"]).total_seconds()
            if total_time_seconds > 0:
                self.metrics.throughput_requests_per_second = len(completed_tasks) / total_time_seconds


class TestDataGenerator:
    """Generate consistent test data for integration tests."""
    
    @staticmethod
    def generate_research_queries() -> List[str]:
        """Generate a variety of research queries for testing."""
        return [
            "Machine learning applications in healthcare",
            "Climate change impact on agriculture",
            "Quantum computing algorithms",
            "Renewable energy storage solutions",
            "Artificial intelligence ethics",
            "Blockchain technology in finance",
            "Gene therapy recent advances",
            "Space exploration technologies",
            "Cybersecurity threat detection",
            "Sustainable urban development"
        ]
    
    @staticmethod
    def generate_mock_sources() -> List[Source]:
        """Generate mock sources for testing."""
        sources = []
        for i in range(10):
            sources.append(Source(
                url=f"https://example{i+1}.com/article",
                title=f"Mock Source {i+1}",
                author=f"Author {i+1}",
                source_type=SourceType.WEB,
                metadata={"domain": f"example{i+1}.com", "word_count": 500 + i * 100}
            ))
        return sources
    
    @staticmethod
    def generate_mock_articles(sources: List[Source]) -> List[ExtractedArticle]:
        """Generate mock extracted articles."""
        articles = []
        for i, source in enumerate(sources):
            articles.append(ExtractedArticle(
                source=source,
                content=f"This is mock content for article {i+1}. " * 50,
                confidence_score=0.8 + (i % 3) * 0.05,
                key_findings=[f"Finding {j+1} from article {i+1}" for j in range(3)]
            ))
        return articles
    
    @staticmethod
    def generate_mock_papers() -> List[AcademicPaper]:
        """Generate mock academic papers."""
        papers = []
        for i in range(5):
            papers.append(AcademicPaper(
                title=f"Mock Academic Paper {i+1}",
                authors=[f"Academic Author {j+1}" for j in range(2 + i % 3)],
                abstract=f"This is a mock abstract for paper {i+1}. " * 20,
                publication_year=2023 - i,
                doi=f"10.1000/mock.paper.{i+1}",
                database_source="arxiv" if i % 2 == 0 else "pubmed",
                citation_count=100 - i * 15
            ))
        return papers


# Test fixtures for the integration framework
@pytest.fixture
def integration_framework():
    """Create integration test framework."""
    return IntegrationTestFramework()


@pytest.fixture
def mock_service_config():
    """Create mock service configuration."""
    return MockServiceConfig(
        web_search_delay=0.05,
        api_query_delay=0.08,
        analysis_delay=0.1,
        failure_rate=0.0,
        rate_limit_enabled=False
    )


@pytest.fixture
def test_data_generator():
    """Create test data generator."""
    return TestDataGenerator()


@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return MockServiceConfig(
        web_search_delay=0.01,
        api_query_delay=0.01,
        analysis_delay=0.02,
        failure_rate=0.0,
        rate_limit_enabled=True,
        rate_limit_requests_per_second=50
    )