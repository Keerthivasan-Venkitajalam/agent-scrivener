"""
Analysis-specific data models for Agent Scrivener.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .core import Source


class NamedEntity(BaseModel):
    """Named entity extracted from text."""
    text: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)  # PERSON, ORG, GPE, etc.
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    start_pos: int = Field(..., ge=0)
    end_pos: int = Field(..., ge=0)


class TopicModel(BaseModel):
    """Topic modeling results."""
    topic_id: int = Field(..., ge=0)
    keywords: List[str] = Field(..., min_items=1)
    weight: float = Field(..., ge=0.0, le=1.0)
    description: Optional[str] = None


class StatisticalSummary(BaseModel):
    """Statistical analysis summary."""
    metric_name: str = Field(..., min_length=1)
    value: float
    unit: Optional[str] = None
    confidence_interval: Optional[List[float]] = None
    sample_size: Optional[int] = None


class AnalysisResults(BaseModel):
    """Complete analysis results from Analysis Agent."""
    session_id: str = Field(..., min_length=1)
    named_entities: List[NamedEntity] = Field(default_factory=list)
    topics: List[TopicModel] = Field(default_factory=list)
    statistical_summaries: List[StatisticalSummary] = Field(default_factory=list)
    key_themes: List[str] = Field(default_factory=list)
    sentiment_scores: Dict[str, float] = Field(default_factory=dict)
    processed_sources: List[Source] = Field(default_factory=list)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_seconds: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "research_001",
                "named_entities": [
                    {
                        "text": "Machine Learning",
                        "label": "TECHNOLOGY",
                        "confidence_score": 0.95,
                        "start_pos": 0,
                        "end_pos": 16
                    }
                ],
                "topics": [
                    {
                        "topic_id": 0,
                        "keywords": ["machine", "learning", "algorithm"],
                        "weight": 0.8,
                        "description": "Machine Learning Algorithms"
                    }
                ],
                "key_themes": ["automation", "artificial intelligence", "research"],
                "sentiment_scores": {"overall": 0.7, "methodology": 0.8}
            }
        }