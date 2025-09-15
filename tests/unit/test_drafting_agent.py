"""
Unit tests for the Drafting Agent.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from agent_scrivener.agents.drafting_agent import DraftingAgent
from agent_scrivener.models.core import Insight, Source, SourceType, DocumentSections, DocumentSection
from agent_scrivener.models.errors import ProcessingError, ValidationError


@pytest.fixture
def drafting_agent():
    """Create a DraftingAgent instance for testing."""
    return DraftingAgent()


@pytest.fixture
def sample_sources():
    """Create sample sources for testing."""
    return [
        Source(
            url="https://example.com/article1",
            title="Machine Learning in Research",
            author="John Doe",
            source_type=SourceType.WEB,
            publication_date=datetime(2023, 1, 15)
        ),
        Source(
            url="https://example.com/article2", 
            title="AI Applications in Science",
            author="Jane Smith",
            source_type=SourceType.ACADEMIC,
            publication_date=datetime(2023, 2, 20)
        )
    ]


@pytest.fixture
def sample_insights(sample_sources):
    """Create sample insights for testing."""
    return [
        Insight(
            topic="Machine Learning Applications",
            summary="Machine learning is revolutionizing research methodologies across multiple domains.",
            supporting_evidence=[
                "ML algorithms improve data analysis efficiency by 300%",
                "Automated pattern recognition reduces human error",
                "Cross-domain applications show consistent benefits"
            ],
            confidence_score=0.95,
            related_sources=[sample_sources[0]],
            tags=["machine-learning", "research", "automation"]
        ),
        Insight(
            topic="Research Automation",
            summary="Automation tools are transforming traditional research workflows.",
            supporting_evidence=[
                "Automated literature reviews save 70% of researcher time",
                "Systematic data collection improves reproducibility"
            ],
            confidence_score=0.88,
            related_sources=[sample_sources[1]],
            tags=["automation", "workflow", "efficiency"]
        ),
        Insight(
            topic="Data Analysis Techniques",
            summary="Advanced analytical techniques enable deeper insights from research data.",
            supporting_evidence=[
                "Statistical modeling reveals hidden patterns",
                "Visualization tools improve data interpretation"
            ],
            confidence_score=0.82,
            related_sources=sample_sources,
            tags=["data-analysis", "statistics", "visualization"]
        )
    ]


class TestDraftingAgent:
    """Test cases for DraftingAgent functionality."""
    
    async def test_initialization(self, drafting_agent):
        """Test agent initialization."""
        assert drafting_agent.name == "drafting_agent"
        assert hasattr(drafting_agent, 'logger')
    
    async def test_execute_success(self, drafting_agent, sample_insights):
        """Test successful document generation."""
        result = await drafting_agent.execute(
            insights=sample_insights,
            session_id="test_session_001",
            original_query="How is machine learning transforming research?"
        )
        
        assert result.success is True
        assert result.data is not None
        assert "document_sections" in result.data
        assert "final_document" in result.data
        assert "word_count" in result.data
        assert "section_count" in result.data
        
        # Verify document sections structure
        doc_sections = result.data["document_sections"]
        assert isinstance(doc_sections, DocumentSections)
        assert doc_sections.introduction is not None
        assert doc_sections.methodology is not None
        assert doc_sections.findings is not None
        assert doc_sections.conclusion is not None
        assert doc_sections.table_of_contents != ""
    
    async def test_execute_empty_insights(self, drafting_agent):
        """Test execution with empty insights list."""
        result = await drafting_agent.execute(
            insights=[],
            session_id="test_session_002",
            original_query="Test query"
        )
        
        assert result.success is False
        assert "No insights provided" in result.error
    
    async def test_execute_missing_parameters(self, drafting_agent, sample_insights):
        """Test execution with missing required parameters."""
        # Test with None values to trigger validation error
        result = await drafting_agent.execute(
            insights=sample_insights,
            session_id=None,
            original_query="Test query"
        )
        
        assert result.success is False
        assert "session_id cannot be None" in result.error
    
    async def test_generate_document_sections(self, drafting_agent, sample_insights):
        """Test document sections generation."""
        doc_sections = await drafting_agent._generate_document_sections(
            sample_insights, 
            "How is machine learning transforming research?",
            "test_session"
        )
        
        assert isinstance(doc_sections, DocumentSections)
        
        # Test introduction section
        assert doc_sections.introduction.title == "Introduction"
        assert doc_sections.introduction.section_type == "introduction"
        assert doc_sections.introduction.order == 0
        assert "machine learning transforming research" in doc_sections.introduction.content.lower()
        
        # Test methodology section
        assert doc_sections.methodology.title == "Methodology"
        assert doc_sections.methodology.section_type == "methodology"
        assert doc_sections.methodology.order == 1
        assert "test_session" in doc_sections.methodology.content
        
        # Test findings section
        assert doc_sections.findings.title == "Findings"
        assert doc_sections.findings.section_type == "findings"
        assert doc_sections.findings.order == 2
        assert "Machine Learning Applications" in doc_sections.findings.content
        
        # Test conclusion section
        assert doc_sections.conclusion.title == "Conclusion"
        assert doc_sections.conclusion.section_type == "conclusion"
        assert doc_sections.conclusion.order == 3
        assert "comprehensive analysis" in doc_sections.conclusion.content.lower()
    
    async def test_group_insights_by_topic(self, drafting_agent, sample_insights):
        """Test insight grouping by topic."""
        topic_groups = drafting_agent._group_insights_by_topic(sample_insights)
        
        assert len(topic_groups) == 3
        assert "machine learning applications" in topic_groups
        assert "research automation" in topic_groups
        assert "data analysis techniques" in topic_groups
        
        # Test sorting by confidence score
        ml_insights = topic_groups["machine learning applications"]
        assert len(ml_insights) == 1
        assert ml_insights[0].confidence_score == 0.95
    
    async def test_generate_introduction_section(self, drafting_agent, sample_insights):
        """Test introduction section generation."""
        section = await drafting_agent._generate_introduction_section(
            "How is machine learning transforming research?",
            sample_insights
        )
        
        assert section.title == "Introduction"
        assert section.section_type == "introduction"
        assert section.order == 0
        assert "machine learning transforming research" in section.content.lower()
        assert "comprehensive analysis" in section.content.lower()
        assert str(len(sample_insights)) in section.content
    
    async def test_generate_methodology_section(self, drafting_agent, sample_insights):
        """Test methodology section generation."""
        section = await drafting_agent._generate_methodology_section(
            sample_insights,
            "test_session_123"
        )
        
        assert section.title == "Methodology"
        assert section.section_type == "methodology"
        assert section.order == 1
        assert "multi-agent approach" in section.content.lower()
        assert "test_session_123" in section.content
        assert "research agent" in section.content.lower()
        assert "api agent" in section.content.lower()
        assert "analysis agent" in section.content.lower()
    
    async def test_generate_findings_section(self, drafting_agent, sample_insights):
        """Test findings section generation."""
        topic_groups = drafting_agent._group_insights_by_topic(sample_insights)
        section = await drafting_agent._generate_findings_section(topic_groups, sample_insights)
        
        assert section.title == "Findings"
        assert section.section_type == "findings"
        assert section.order == 2
        assert "Machine Learning Applications" in section.content
        assert "Research Automation" in section.content
        assert "Data Analysis Techniques" in section.content
        assert "Key Evidence" in section.content
    
    async def test_generate_conclusion_section(self, drafting_agent, sample_insights):
        """Test conclusion section generation."""
        section = await drafting_agent._generate_conclusion_section(
            sample_insights,
            "How is machine learning transforming research?"
        )
        
        assert section.title == "Conclusion"
        assert section.section_type == "conclusion"
        assert section.order == 3
        assert "machine learning transforming research" in section.content.lower()
        assert "comprehensive analysis" in section.content.lower()
        assert "key contributions" in section.content.lower()
        assert "future directions" in section.content.lower()
    
    def test_synthesize_topic_insights(self, drafting_agent, sample_insights):
        """Test topic insight synthesis."""
        # Test with single insight
        single_insight = [sample_insights[0]]
        result = drafting_agent._synthesize_topic_insights(single_insight)
        assert result == sample_insights[0].summary
        
        # Test with multiple insights
        multiple_insights = sample_insights[:2]
        result = drafting_agent._synthesize_topic_insights(multiple_insights)
        assert sample_insights[0].summary in result
        assert len(result) > len(sample_insights[0].summary)
        
        # Test with empty list
        result = drafting_agent._synthesize_topic_insights([])
        assert "No significant insights available" in result
    
    def test_generate_table_of_contents(self, drafting_agent):
        """Test table of contents generation."""
        # Create sample document sections
        introduction = DocumentSection(
            title="Introduction",
            content="Test content",
            section_type="introduction",
            order=0
        )
        methodology = DocumentSection(
            title="Methodology",
            content="Test content",
            section_type="methodology", 
            order=1
        )
        findings = DocumentSection(
            title="Findings",
            content="Test content",
            section_type="findings",
            order=2
        )
        conclusion = DocumentSection(
            title="Conclusion",
            content="Test content",
            section_type="conclusion",
            order=3
        )
        
        doc_sections = DocumentSections(
            introduction=introduction,
            methodology=methodology,
            findings=findings,
            conclusion=conclusion
        )
        
        toc = drafting_agent._generate_table_of_contents(doc_sections)
        
        assert "# Table of Contents" in toc
        assert "[Introduction](#introduction)" in toc
        assert "[Methodology](#methodology)" in toc
        assert "[Findings](#findings)" in toc
        assert "[Conclusion](#conclusion)" in toc
    
    def test_create_anchor_link(self, drafting_agent):
        """Test anchor link creation."""
        # Test normal title
        anchor = drafting_agent._create_anchor_link("Introduction")
        assert anchor == "introduction"
        
        # Test title with spaces
        anchor = drafting_agent._create_anchor_link("Data Analysis Techniques")
        assert anchor == "data-analysis-techniques"
        
        # Test title with special characters
        anchor = drafting_agent._create_anchor_link("Results & Discussion (Part 1)")
        assert anchor == "results-discussion-part-1"
        
        # Test title with multiple spaces and hyphens
        anchor = drafting_agent._create_anchor_link("  Complex -- Title   Example  ")
        assert anchor == "complex-title-example"
    
    async def test_format_final_document(self, drafting_agent, sample_insights):
        """Test final document formatting."""
        doc_sections = await drafting_agent._generate_document_sections(
            sample_insights,
            "Test query",
            "test_session"
        )
        doc_sections.table_of_contents = "# Table of Contents\n1. [Introduction](#introduction)"
        
        final_doc = await drafting_agent._format_final_document(doc_sections)
        
        assert "# Table of Contents" in final_doc
        assert "---" in final_doc
        assert "## Introduction" in final_doc
        assert "## Methodology" in final_doc
        assert "## Findings" in final_doc
        assert "## Conclusion" in final_doc
        assert "Document generated on" in final_doc
        assert "Total sections:" in final_doc
    
    async def test_generate_section_content(self, drafting_agent, sample_insights):
        """Test individual section content generation."""
        context = {
            "original_query": "Test query",
            "session_id": "test_session"
        }
        
        # Test introduction
        intro_content = await drafting_agent.generate_section_content(
            "introduction", sample_insights, context
        )
        assert "## Introduction" in intro_content
        assert "Test query" in intro_content
        
        # Test methodology
        method_content = await drafting_agent.generate_section_content(
            "methodology", sample_insights, context
        )
        assert "## Methodology" in method_content
        assert "test_session" in method_content
        
        # Test findings
        findings_content = await drafting_agent.generate_section_content(
            "findings", sample_insights, context
        )
        assert "## Findings" in findings_content
        
        # Test conclusion
        conclusion_content = await drafting_agent.generate_section_content(
            "conclusion", sample_insights, context
        )
        assert "## Conclusion" in conclusion_content
        
        # Test invalid section type
        with pytest.raises(ValueError, match="Unknown section type"):
            await drafting_agent.generate_section_content(
                "invalid_section", sample_insights, context
            )
    
    def test_validate_document_structure_success(self, drafting_agent):
        """Test successful document structure validation."""
        # Create valid document sections
        doc_sections = DocumentSections(
            introduction=DocumentSection(
                title="Introduction",
                content="Valid content",
                section_type="introduction",
                order=0
            ),
            methodology=DocumentSection(
                title="Methodology", 
                content="Valid content",
                section_type="methodology",
                order=1
            ),
            findings=DocumentSection(
                title="Findings",
                content="Valid content", 
                section_type="findings",
                order=2
            ),
            conclusion=DocumentSection(
                title="Conclusion",
                content="Valid content",
                section_type="conclusion",
                order=3
            ),
            table_of_contents="# Table of Contents\n1. Introduction"
        )
        
        result = drafting_agent.validate_document_structure(doc_sections)
        assert result is True
    
    def test_validate_document_structure_empty_content(self, drafting_agent):
        """Test document structure validation with empty content."""
        # Create sections with valid content first, then modify
        doc_sections = DocumentSections(
            introduction=DocumentSection(
                title="Introduction",
                content="Valid content",
                section_type="introduction",
                order=0
            ),
            methodology=DocumentSection(
                title="Methodology",
                content="Valid content",
                section_type="methodology", 
                order=1
            ),
            findings=DocumentSection(
                title="Findings",
                content="Valid content",
                section_type="findings",
                order=2
            ),
            conclusion=DocumentSection(
                title="Conclusion", 
                content="Valid content",
                section_type="conclusion",
                order=3
            ),
            table_of_contents="# Table of Contents"
        )
        
        # Manually set empty content to bypass Pydantic validation
        doc_sections.introduction.content = ""
        
        with pytest.raises(ValidationError, match="Empty sections found"):
            drafting_agent.validate_document_structure(doc_sections)
    
    def test_validate_document_structure_missing_toc(self, drafting_agent):
        """Test document structure validation with missing table of contents."""
        doc_sections = DocumentSections(
            introduction=DocumentSection(
                title="Introduction",
                content="Valid content",
                section_type="introduction",
                order=0
            ),
            methodology=DocumentSection(
                title="Methodology",
                content="Valid content", 
                section_type="methodology",
                order=1
            ),
            findings=DocumentSection(
                title="Findings",
                content="Valid content",
                section_type="findings",
                order=2
            ),
            conclusion=DocumentSection(
                title="Conclusion",
                content="Valid content",
                section_type="conclusion",
                order=3
            ),
            table_of_contents=""  # Empty TOC
        )
        
        with pytest.raises(ValidationError, match="Table of contents is missing"):
            drafting_agent.validate_document_structure(doc_sections)
    
    async def test_generate_additional_sections(self, drafting_agent, sample_insights):
        """Test generation of additional sections for complex topics."""
        # Create many topic groups to trigger additional sections
        # Need to create multiple insights per topic to meet the >= 2 insights requirement
        extended_insights = sample_insights.copy()
        for i in range(4, 8):
            # Add 2 insights per new topic to meet the criteria
            for j in range(2):
                extended_insights.append(
                    Insight(
                        topic=f"Topic {i}",
                        summary=f"Summary for topic {i}, insight {j+1}",
                        supporting_evidence=[f"Evidence {i}.{j+1}.1", f"Evidence {i}.{j+1}.2"],
                        confidence_score=0.8 + (j * 0.05),
                        related_sources=[sample_insights[0].related_sources[0]],
                        tags=[f"tag{i}"]
                    )
                )
        
        topic_groups = drafting_agent._group_insights_by_topic(extended_insights)
        additional_sections = await drafting_agent._generate_additional_sections(topic_groups)
        
        # Should create additional sections for top topics (if conditions are met)
        # The logic requires topics with >= 2 insights, so let's ensure we have that
        if len(additional_sections) > 0:
            for section in additional_sections:
                assert section.section_type == "detailed_analysis"
                assert "Detailed Analysis:" in section.title
                assert section.order >= 4
        else:
            # If no additional sections, verify the topic groups don't meet criteria
            topic_counts = {topic: len(insights) for topic, insights in topic_groups.items()}
            substantial_topics = [topic for topic, count in topic_counts.items() if count >= 2]
            # Should have fewer than 3 substantial topics to not create additional sections
            assert len(substantial_topics) < 3
    
    def test_create_detailed_topic_analysis(self, drafting_agent, sample_insights):
        """Test detailed topic analysis creation."""
        topic_insights = [sample_insights[0], sample_insights[1]]
        analysis = drafting_agent._create_detailed_topic_analysis(topic_insights)
        
        assert "Key Findings" in analysis
        assert "Analysis Synthesis" in analysis
        assert "Finding 1" in analysis
        assert "Finding 2" in analysis
        assert "Supporting Evidence" in analysis
        assert str(len(topic_insights)) in analysis


class TestDraftingAgentIntegration:
    """Integration tests for DraftingAgent with realistic scenarios."""
    
    async def test_full_document_generation_workflow(self, drafting_agent, sample_insights):
        """Test complete document generation workflow."""
        result = await drafting_agent.execute(
            insights=sample_insights,
            session_id="integration_test_001",
            original_query="What are the latest developments in AI research automation?"
        )
        
        assert result.success is True
        
        # Verify complete document structure
        doc_sections = result.data["document_sections"]
        final_document = result.data["final_document"]
        
        # Validate document structure
        assert drafting_agent.validate_document_structure(doc_sections) is True
        
        # Check final document contains all sections
        assert "# Table of Contents" in final_document
        assert "## Introduction" in final_document
        assert "## Methodology" in final_document
        assert "## Findings" in final_document
        assert "## Conclusion" in final_document
        
        # Check metadata
        assert result.data["word_count"] > 0
        assert result.data["section_count"] >= 4
        
        # Verify content quality
        assert "AI research automation" in final_document
        assert "Machine Learning Applications" in final_document
        assert "integration_test_001" in final_document
    
    async def test_large_insight_set_handling(self, drafting_agent, sample_sources):
        """Test handling of large insight sets."""
        # Create a large set of insights
        large_insight_set = []
        topics = ["AI", "ML", "Data Science", "Automation", "Research", "Analytics", "NLP", "Computer Vision"]
        
        for i, topic in enumerate(topics):
            for j in range(3):  # 3 insights per topic
                large_insight_set.append(
                    Insight(
                        topic=f"{topic} Applications",
                        summary=f"Detailed analysis of {topic} in research context {j+1}.",
                        supporting_evidence=[
                            f"{topic} evidence point 1",
                            f"{topic} evidence point 2",
                            f"{topic} evidence point 3"
                        ],
                        confidence_score=0.7 + (i * 0.03),  # Varying confidence
                        related_sources=sample_sources[:1],
                        tags=[topic.lower(), "research", "applications"]
                    )
                )
        
        result = await drafting_agent.execute(
            insights=large_insight_set,
            session_id="large_test_001", 
            original_query="Comprehensive analysis of AI technologies in research"
        )
        
        assert result.success is True
        assert result.data["word_count"] > 1000  # Should be substantial
        assert len(result.data["document_sections"].sections) > 0  # Should have additional sections
    
    async def test_low_confidence_insights_handling(self, drafting_agent, sample_sources):
        """Test handling of insights with varying confidence levels."""
        mixed_confidence_insights = [
            Insight(
                topic="High Confidence Topic",
                summary="This is a high confidence insight with strong evidence.",
                supporting_evidence=["Strong evidence 1", "Strong evidence 2"],
                confidence_score=0.95,
                related_sources=sample_sources[:1],
                tags=["high-confidence"]
            ),
            Insight(
                topic="Medium Confidence Topic", 
                summary="This is a medium confidence insight with moderate evidence.",
                supporting_evidence=["Moderate evidence 1"],
                confidence_score=0.65,
                related_sources=sample_sources[:1],
                tags=["medium-confidence"]
            ),
            Insight(
                topic="Low Confidence Topic",
                summary="This is a low confidence insight requiring further investigation.",
                supporting_evidence=["Weak evidence 1"],
                confidence_score=0.35,
                related_sources=sample_sources[:1],
                tags=["low-confidence"]
            )
        ]
        
        result = await drafting_agent.execute(
            insights=mixed_confidence_insights,
            session_id="confidence_test_001",
            original_query="Analysis of research findings with varying confidence levels"
        )
        
        assert result.success is True
        
        # High confidence insights should be prominently featured
        final_document = result.data["final_document"]
        assert "High Confidence Topic" in final_document
        
        # Should handle low confidence appropriately
        assert "further investigation" in final_document.lower()


if __name__ == "__main__":
    pytest.main([__file__])