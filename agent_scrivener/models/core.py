"""
Core Pydantic data models for Agent Scrivener.
"""

from pydantic import BaseModel, HttpUrl, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Types of information sources."""
    WEB = "web"
    ACADEMIC = "academic"
    DATABASE = "database"
    API = "api"


class Source(BaseModel):
    """Information source metadata."""
    url: HttpUrl
    title: str = Field(..., min_length=1, max_length=500)
    author: Optional[str] = Field(None, max_length=200)
    publication_date: Optional[datetime] = None
    source_type: SourceType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class ExtractedArticle(BaseModel):
    """Article content extracted from web sources."""
    source: Source
    content: str = Field(..., min_length=1)
    key_findings: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    word_count: Optional[int] = None
    
    @field_validator('word_count', mode='before')
    @classmethod
    def calculate_word_count(cls, v, info):
        if v is None and hasattr(info, 'data') and info.data and 'content' in info.data:
            return len(info.data['content'].split())
        return v
    
    def model_post_init(self, __context):
        """Calculate word count after model initialization if not provided."""
        if self.word_count is None:
            self.word_count = len(self.content.split())


class AcademicPaper(BaseModel):
    """Academic paper metadata and content."""
    title: str = Field(..., min_length=1, max_length=500)
    authors: List[str] = Field(..., min_length=1)
    abstract: str = Field(..., min_length=1)
    publication_year: int = Field(..., ge=1900, le=2030)
    doi: Optional[str] = Field(None, pattern=r'^10\.\d{4,}/.+')
    database_source: str = Field(..., min_length=1)
    citation_count: Optional[int] = Field(None, ge=0)
    keywords: List[str] = Field(default_factory=list)
    full_text_url: Optional[HttpUrl] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Machine Learning in Research Automation",
                "authors": ["Smith, J.", "Doe, A."],
                "abstract": "This paper explores the application of ML in research...",
                "publication_year": 2023,
                "doi": "10.1234/example.2023.001",
                "database_source": "arXiv",
                "citation_count": 15,
                "keywords": ["machine learning", "automation", "research"]
            }
        }
    )


class Insight(BaseModel):
    """Structured insight derived from analysis."""
    topic: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1)
    supporting_evidence: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    related_sources: List[Source] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class TaskStatus(str, Enum):
    """Status of research tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchTask(BaseModel):
    """Individual research task definition."""
    task_id: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ResearchPlan(BaseModel):
    """Complete research execution plan."""
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., min_length=1)
    tasks: List[ResearchTask] = Field(default_factory=list)
    estimated_duration_minutes: int = Field(..., ge=1)
    created_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    
    def get_task_by_id(self, task_id: str) -> Optional[ResearchTask]:
        """Get a task by its ID."""
        return next((task for task in self.tasks if task.task_id == task_id), None)
    
    def get_ready_tasks(self) -> List[ResearchTask]:
        """Get tasks that are ready to execute (dependencies completed)."""
        completed_task_ids = {task.task_id for task in self.tasks if task.status == TaskStatus.COMPLETED}
        
        ready_tasks = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                if all(dep_id in completed_task_ids for dep_id in task.dependencies):
                    ready_tasks.append(task)
        
        return ready_tasks


class DocumentSection(BaseModel):
    """Individual document section."""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    section_type: str = Field(..., min_length=1)
    order: int = Field(..., ge=0)
    citations: List[str] = Field(default_factory=list)


class DocumentSections(BaseModel):
    """Complete document structure."""
    introduction: DocumentSection
    methodology: DocumentSection
    findings: DocumentSection
    conclusion: DocumentSection
    sections: List[DocumentSection] = Field(default_factory=list)
    table_of_contents: str = ""
    
    def get_all_sections(self) -> List[DocumentSection]:
        """Get all sections in order."""
        core_sections = [self.introduction, self.methodology, self.findings, self.conclusion]
        all_sections = core_sections + self.sections
        return sorted(all_sections, key=lambda x: x.order)


class Citation(BaseModel):
    """Citation information for sources."""
    citation_id: str = Field(..., min_length=1)
    source: Source
    citation_text: str = Field(..., min_length=1)
    page_numbers: Optional[str] = None
    quote: Optional[str] = None
    context: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class SessionState(str, Enum):
    """Research session states."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    DRAFTING = "drafting"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentExecution(BaseModel):
    """Track individual agent execution within a session."""
    execution_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    
    def mark_completed(self, output_data: Dict[str, Any]):
        """Mark execution as completed with output data."""
        self.completed_at = datetime.now()
        self.status = TaskStatus.COMPLETED
        self.output_data = output_data
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error_message: str):
        """Mark execution as failed with error message."""
        self.completed_at = datetime.now()
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()


class SessionMetrics(BaseModel):
    """Performance and quality metrics for a research session."""
    total_sources_found: int = 0
    total_sources_processed: int = 0
    average_source_confidence: float = 0.0
    total_insights_generated: int = 0
    average_insight_confidence: float = 0.0
    total_execution_time_seconds: float = 0.0
    successful_agent_executions: int = 0
    failed_agent_executions: int = 0
    final_document_word_count: Optional[int] = None
    quality_score: Optional[float] = None  # Overall quality assessment
    
    def calculate_success_rate(self) -> float:
        """Calculate the success rate of agent executions."""
        total_executions = self.successful_agent_executions + self.failed_agent_executions
        if total_executions == 0:
            return 0.0
        return self.successful_agent_executions / total_executions
    
    def update_from_session(self, session: 'ResearchSession'):
        """Update metrics from a research session."""
        self.total_sources_found = len(session.get_all_sources())
        self.total_sources_processed = len(session.extracted_articles) + len(session.academic_papers)
        
        # Calculate average source confidence
        if session.extracted_articles:
            self.average_source_confidence = sum(
                article.confidence_score for article in session.extracted_articles
            ) / len(session.extracted_articles)
        
        # Calculate insights metrics
        self.total_insights_generated = len(session.insights)
        if session.insights:
            self.average_insight_confidence = sum(
                insight.confidence_score for insight in session.insights
            ) / len(session.insights)
        
        # Calculate document metrics
        if session.final_document:
            self.final_document_word_count = len(session.final_document.split())


class WorkflowStep(BaseModel):
    """Individual step in the research workflow."""
    step_id: str = Field(..., min_length=1)
    step_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    required_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    estimated_duration_minutes: int = Field(..., ge=1)
    status: TaskStatus = TaskStatus.PENDING
    agent_executions: List[AgentExecution] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def add_execution(self, execution: AgentExecution):
        """Add an agent execution to this workflow step."""
        self.agent_executions.append(execution)
    
    def is_ready_to_execute(self, available_outputs: List[str]) -> bool:
        """Check if this step is ready to execute based on available inputs."""
        return all(required_input in available_outputs for required_input in self.required_inputs)
    
    def get_execution_by_agent(self, agent_name: str) -> Optional[AgentExecution]:
        """Get execution by agent name."""
        return next(
            (exec for exec in self.agent_executions if exec.agent_name == agent_name),
            None
        )


class ResearchSession(BaseModel):
    """Complete research session state."""
    session_id: str = Field(..., min_length=1)
    original_query: str = Field(..., min_length=1)
    plan: ResearchPlan
    extracted_articles: List[ExtractedArticle] = Field(default_factory=list)
    academic_papers: List[AcademicPaper] = Field(default_factory=list)
    insights: List[Insight] = Field(default_factory=list)
    document_sections: Optional[DocumentSections] = None
    citations: List[Citation] = Field(default_factory=list)
    final_document: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    session_state: SessionState = SessionState.INITIALIZING
    workflow_steps: List[WorkflowStep] = Field(default_factory=list)
    agent_executions: List[AgentExecution] = Field(default_factory=list)
    metrics: SessionMetrics = Field(default_factory=SessionMetrics)
    
    # API-specific attributes
    user_id: Optional[str] = None
    query: Optional[str] = None  # Alias for original_query
    estimated_duration_minutes: Optional[int] = None
    progress_percentage: float = 0.0
    current_task: Optional[str] = None
    completed_tasks: List[str] = Field(default_factory=list)
    estimated_time_remaining_minutes: Optional[int] = None
    error_message: Optional[str] = None
    document_content: Optional[str] = None  # Alias for final_document
    sources_count: Optional[int] = None
    word_count: Optional[int] = None
    completion_time_minutes: Optional[float] = None
    completed_at: Optional[datetime] = None
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()
    
    def get_all_sources(self) -> List[Source]:
        """Get all sources from articles and papers."""
        sources = [article.source for article in self.extracted_articles]
        
        # Convert academic papers to sources
        for paper in self.academic_papers:
            paper_source = Source(
                url=paper.full_text_url or f"https://doi.org/{paper.doi}" if paper.doi else "https://example.com",
                title=paper.title,
                author=", ".join(paper.authors),
                publication_date=datetime(paper.publication_year, 1, 1),
                source_type=SourceType.ACADEMIC,
                metadata={
                    "doi": paper.doi,
                    "database": paper.database_source,
                    "citation_count": paper.citation_count
                }
            )
            sources.append(paper_source)
        
        return sources
    
    def add_workflow_step(self, step: WorkflowStep):
        """Add a workflow step to the session."""
        self.workflow_steps.append(step)
        self.update_timestamp()
    
    def get_current_workflow_step(self) -> Optional[WorkflowStep]:
        """Get the current active workflow step."""
        return next(
            (step for step in self.workflow_steps if step.status == TaskStatus.IN_PROGRESS),
            None
        )
    
    def get_ready_workflow_steps(self) -> List[WorkflowStep]:
        """Get workflow steps that are ready to execute."""
        completed_outputs = []
        for step in self.workflow_steps:
            if step.status == TaskStatus.COMPLETED:
                completed_outputs.extend(step.expected_outputs)
        
        ready_steps = []
        for step in self.workflow_steps:
            if step.status == TaskStatus.PENDING and step.is_ready_to_execute(completed_outputs):
                ready_steps.append(step)
        
        return ready_steps
    
    def add_agent_execution(self, execution: AgentExecution):
        """Add an agent execution to the session."""
        self.agent_executions.append(execution)
        self.update_timestamp()
    
    def transition_state(self, new_state: SessionState):
        """Transition the session to a new state."""
        self.session_state = new_state
        self.update_timestamp()
    
    def calculate_progress_percentage(self) -> float:
        """Calculate the overall progress percentage of the session."""
        if not self.workflow_steps:
            return 0.0
        
        completed_steps = sum(1 for step in self.workflow_steps if step.status == TaskStatus.COMPLETED)
        return (completed_steps / len(self.workflow_steps)) * 100.0
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the session state and progress."""
        return {
            "session_id": self.session_id,
            "query": self.original_query,
            "status": self.status.value,
            "session_state": self.session_state.value,
            "progress_percentage": self.calculate_progress_percentage(),
            "total_sources": len(self.get_all_sources()),
            "total_insights": len(self.insights),
            "total_citations": len(self.citations),
            "has_final_document": self.final_document is not None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metrics": {
                "success_rate": self.metrics.calculate_success_rate(),
                "total_execution_time": self.metrics.total_execution_time_seconds,
                "quality_score": self.metrics.quality_score
            }
        }