"""
Unit tests for AnalysisAgent.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any

from agent_scrivener.agents.analysis_agent import AnalysisAgent, Insight
from agent_scrivener.models.core import ExtractedArticle, AcademicPaper, Source, SourceType
from agent_scrivener.models.analysis import AnalysisResults, NamedEntity, TopicModel, StatisticalSummary
from agent_scrivener.models.errors import ProcessingError, ValidationError
from agent_scrivener.tools.code_interpreter_wrapper import CodeInterpreterWrapper


@pytest.fixture
def mock_code_interpreter():
    """Create mock code interpreter wrapper."""
    mock = AsyncMock(spec=CodeInterpreterWrapper)
    
    # Mock health check
    mock.health_check.return_value = {
        'status': 'healthy',
        'interpreter_available': True
    }
    
    # Mock NER analysis
    mock.perform_ner_analysis.return_value = [
        NamedEntity(
            text="Machine Learning",
            label="TECHNOLOGY",
            confidence_score=0.9,
            start_pos=0,
            end_pos=16
        ),
        NamedEntity(
            text="Research Institute",
            label="ORG",
            confidence_score=0.85,
            start_pos=20,
            end_pos=38
        )
    ]
    
    # Mock topic modeling
    mock.perform_topic_modeling.return_value = [
        TopicModel(
            topic_id=0,
            keywords=["machine", "learning", "algorithm"],
            weight=0.8,
            description="Machine Learning"
        ),
        TopicModel(
            topic_id=1,
            keywords=["research", "analysis", "data"],
            weight=0.6,
            description="Research Methods"
        )
    ]
    
    # Mock statistical analysis
    mock.perform_statistical_analysis.return_value = [
        StatisticalSummary(
            metric_name="mean_confidence",
            value=0.85,
            unit="score",
            sample_size=10
        ),
        StatisticalSummary(
            metric_name="std_deviation",
            value=0.12,
            unit="score",
            sample_size=10
        )
    ]
    
    # Mock sentiment analysis
    mock.perform_sentiment_analysis.return_value = {
        'overall': 0.7,
        'positive': 0.8,
        'negative': 0.1,
        'neutral': 0.1
    }
    
    # Mock comprehensive analysis
    mock.analyze_research_content.return_value = AnalysisResults(
        session_id="test_session",
        named_entities=[
            NamedEntity(
                text="Artificial Intelligence",
                label="TECHNOLOGY",
                confidence_score=0.95,
                start_pos=0,
                end_pos=22
            )
        ],
        topics=[
            TopicModel(
                topic_id=0,
                keywords=["ai", "machine", "learning"],
                weight=0.9,
                description="AI and Machine Learning"
            )
        ],
        statistical_summaries=[
            StatisticalSummary(
                metric_name="average_confidence",
                value=0.88,
                unit="score",
                sample_size=5
            )
        ],
        key_themes=["artificial intelligence", "machine learning"],
        sentiment_scores={'overall': 0.6},
        processed_sources=[],
        analysis_timestamp=datetime.now(),
        processing_time_seconds=1.5
    )
    
    return mock


@pytest.fixture
def analysis_agent(mock_code_interpreter):
    """Create AnalysisAgent instance with mocked dependencies."""
    return AnalysisAgent(mock_code_interpreter)


@pytest.fixture
def sample_articles():
    """Create sample ExtractedArticle objects for testing."""
    return [
        ExtractedArticle(
            source=Source(
                url="https://example.com/article1",
                title="Machine Learning Research",
                author="Dr. Smith",
                source_type=SourceType.WEB,
                metadata={'quality_score': 0.9, 'domain': 'example.com'}
            ),
            content="This article discusses machine learning algorithms and their applications in research.",
            key_findings=["ML algorithms are effective", "Research shows promising results"],
            confidence_score=0.85,
            extraction_timestamp=datetime.now()
        ),
        ExtractedArticle(
            source=Source(
                url="https://research.edu/article2",
                title="AI in Healthcare",
                author="Prof. Johnson",
                source_type=SourceType.WEB,
                metadata={'quality_score': 0.95, 'domain': 'research.edu'}
            ),
            content="Artificial intelligence applications in healthcare show significant potential for improving patient outcomes.",
            key_findings=["AI improves diagnostics", "Patient outcomes enhanced"],
            confidence_score=0.92,
            extraction_timestamp=datetime.now()
        )
    ]


@pytest.fixture
def sample_papers():
    """Create sample AcademicPaper objects for testing."""
    return [
        AcademicPaper(
            title="Deep Learning for Medical Diagnosis",
            authors=["Dr. Alice", "Dr. Bob"],
            abstract="This paper presents a deep learning approach for medical diagnosis with high accuracy.",
            publication_year=2023,
            doi="10.1234/example.2023.001",
            database_source="arXiv",
            citation_count=25,
            keywords=["deep learning", "medical", "diagnosis"],
            full_text_url="https://arxiv.org/abs/2023.001"
        ),
        AcademicPaper(
            title="Natural Language Processing in Research",
            authors=["Prof. Carol", "Dr. David"],
            abstract="An overview of NLP techniques applied to research data analysis and knowledge extraction.",
            publication_year=2024,
            doi="10.5678/nlp.2024.002",
            database_source="Semantic Scholar",
            citation_count=15,
            keywords=["nlp", "research", "analysis"],
            full_text_url="https://semanticscholar.org/paper/002"
        )
    ]


class TestAnalysisAgent:
    """Test cases for AnalysisAgent."""
    
    async def test_initialization(self, mock_code_interpreter):
        """Test AnalysisAgent initialization."""
        agent = AnalysisAgent(mock_code_interpreter)
        
        assert agent.name == "analysis_agent"
        assert agent.interpreter == mock_code_interpreter
        assert agent.min_confidence_threshold == 0.6
        assert agent.max_topics == 10
        assert 'enable_ner' in agent.analysis_config
        assert 'enable_topics' in agent.analysis_config
    
    async def test_initialization_with_custom_name(self, mock_code_interpreter):
        """Test AnalysisAgent initialization with custom name."""
        agent = AnalysisAgent(mock_code_interpreter, name="custom_analysis")
        
        assert agent.name == "custom_analysis"
    
    async def test_execute_success(self, analysis_agent, sample_articles, sample_papers):
        """Test successful execution of analysis agent."""
        result = await analysis_agent.execute(
            articles=sample_articles,
            papers=sample_papers,
            session_id="test_session"
        )
        
        assert result.success is True
        assert result.agent_name == "analysis_agent"
        assert result.data is not None
        assert isinstance(result.data, AnalysisResults)
        assert result.execution_time_ms is not None
    
    async def test_execute_with_minimal_input(self, analysis_agent, sample_articles):
        """Test execution with minimal required input."""
        result = await analysis_agent.execute(articles=sample_articles)
        
        assert result.success is True
        assert isinstance(result.data, AnalysisResults)
    
    async def test_execute_validation_error_empty_articles(self, analysis_agent):
        """Test execution with empty articles list."""
        result = await analysis_agent.execute(articles=[])
        
        assert result.success is False
        assert "at least one article" in result.error.lower()
    
    async def test_execute_validation_error_missing_articles(self, analysis_agent):
        """Test execution without articles parameter."""
        result = await analysis_agent.execute()
        
        assert result.success is False
        assert "missing 1 required positional argument: 'articles'" in result.error
    
    async def test_perform_named_entity_recognition(self, analysis_agent):
        """Test named entity recognition functionality."""
        texts = [
            "Machine learning is transforming healthcare research.",
            "The Research Institute published groundbreaking findings."
        ]
        
        entities = await analysis_agent.perform_named_entity_recognition(texts)
        
        assert len(entities) > 0
        assert all(isinstance(entity, NamedEntity) for entity in entities)
        assert all(entity.confidence_score >= 0.5 for entity in entities)
        
        # Verify mock was called correctly
        analysis_agent.interpreter.perform_ner_analysis.assert_called_once_with(
            texts, 'en_core_web_sm'
        )
    
    async def test_perform_named_entity_recognition_custom_model(self, analysis_agent):
        """Test NER with custom model."""
        texts = ["Test text for NER analysis."]
        
        await analysis_agent.perform_named_entity_recognition(texts, model_name='en_core_web_lg')
        
        analysis_agent.interpreter.perform_ner_analysis.assert_called_once_with(
            texts, 'en_core_web_lg'
        )
    
    async def test_perform_named_entity_recognition_error(self, analysis_agent):
        """Test NER error handling."""
        analysis_agent.interpreter.perform_ner_analysis.side_effect = Exception("NER failed")
        
        with pytest.raises(ProcessingError, match="Named entity recognition failed"):
            await analysis_agent.perform_named_entity_recognition(["test text"])
    
    async def test_perform_topic_modeling(self, analysis_agent):
        """Test topic modeling functionality."""
        texts = [
            "Machine learning algorithms are used in data analysis.",
            "Research methodology involves statistical analysis.",
            "Artificial intelligence applications in healthcare."
        ]
        
        topics = await analysis_agent.perform_topic_modeling(texts, num_topics=3)
        
        assert len(topics) > 0
        assert all(isinstance(topic, TopicModel) for topic in topics)
        assert all(topic.weight > 0 for topic in topics)
        assert all(len(topic.keywords) > 0 for topic in topics)
        
        # Verify mock was called correctly (should adjust to 2 topics for 3 texts)
        analysis_agent.interpreter.perform_topic_modeling.assert_called_once_with(
            texts, 2, 'lda'
        )
    
    async def test_perform_topic_modeling_adjusted_topics(self, analysis_agent):
        """Test topic modeling with topic count adjustment."""
        texts = ["Single text for analysis."]
        
        await analysis_agent.perform_topic_modeling(texts, num_topics=10)
        
        # Should adjust to minimum of 2 topics for single text
        analysis_agent.interpreter.perform_topic_modeling.assert_called_once_with(
            texts, 2, 'lda'
        )
    
    async def test_perform_topic_modeling_nmf_method(self, analysis_agent):
        """Test topic modeling with NMF method."""
        texts = ["Test text for topic modeling."]
        
        await analysis_agent.perform_topic_modeling(texts, method='nmf')
        
        analysis_agent.interpreter.perform_topic_modeling.assert_called_once_with(
            texts, 2, 'nmf'
        )
    
    async def test_perform_topic_modeling_error(self, analysis_agent):
        """Test topic modeling error handling."""
        analysis_agent.interpreter.perform_topic_modeling.side_effect = Exception("Topic modeling failed")
        
        with pytest.raises(ProcessingError, match="Topic modeling failed"):
            await analysis_agent.perform_topic_modeling(["test text"])
    
    async def test_perform_statistical_analysis(self, analysis_agent, sample_articles, sample_papers):
        """Test statistical analysis functionality."""
        stats = await analysis_agent.perform_statistical_analysis(sample_articles, sample_papers)
        
        assert len(stats) > 0
        assert all(isinstance(stat, StatisticalSummary) for stat in stats)
        
        # Should include both interpreter results and custom statistics
        analysis_agent.interpreter.perform_statistical_analysis.assert_called_once()
    
    async def test_perform_statistical_analysis_no_papers(self, analysis_agent, sample_articles):
        """Test statistical analysis with only articles."""
        stats = await analysis_agent.perform_statistical_analysis(sample_articles, [])
        
        assert isinstance(stats, list)
        # Should still work with just articles
    
    async def test_perform_statistical_analysis_error(self, analysis_agent, sample_articles):
        """Test statistical analysis error handling."""
        analysis_agent.interpreter.perform_statistical_analysis.side_effect = Exception("Stats failed")
        
        with pytest.raises(ProcessingError, match="Statistical analysis failed"):
            await analysis_agent.perform_statistical_analysis(sample_articles, [])
    
    async def test_comprehensive_analysis_full_workflow(self, analysis_agent, sample_articles, sample_papers):
        """Test complete comprehensive analysis workflow."""
        result = await analysis_agent._perform_comprehensive_analysis(
            articles=sample_articles,
            papers=sample_papers,
            session_id="test_comprehensive",
            analysis_config={'enable_ner': True, 'enable_topics': True}
        )
        
        assert isinstance(result, AnalysisResults)
        assert result.session_id == "test_comprehensive"
        assert len(result.named_entities) > 0
        assert len(result.topics) > 0
        assert len(result.key_themes) > 0
        assert result.processing_time_seconds is not None
        
        # Verify interpreter was called
        analysis_agent.interpreter.analyze_research_content.assert_called_once()
    
    async def test_comprehensive_analysis_with_config_override(self, analysis_agent, sample_articles):
        """Test comprehensive analysis with configuration override."""
        custom_config = {
            'enable_ner': False,
            'enable_topics': True,
            'num_topics': 3
        }
        
        await analysis_agent._perform_comprehensive_analysis(
            articles=sample_articles,
            analysis_config=custom_config
        )
        
        # Verify config was merged and passed
        call_args = analysis_agent.interpreter.analyze_research_content.call_args
        passed_config = call_args[1]['analysis_config']
        assert passed_config['enable_ner'] is False
        assert passed_config['enable_topics'] is True
        assert passed_config['num_topics'] == 3
    
    async def test_comprehensive_analysis_error_handling(self, analysis_agent, sample_articles):
        """Test comprehensive analysis error handling."""
        analysis_agent.interpreter.analyze_research_content.side_effect = Exception("Analysis failed")
        
        with pytest.raises(ProcessingError, match="Analysis failed"):
            await analysis_agent._perform_comprehensive_analysis(articles=sample_articles)
    
    def test_filter_and_enhance_entities(self, analysis_agent):
        """Test entity filtering and enhancement."""
        entities = [
            NamedEntity(text="Machine Learning", label="TECHNOLOGY", confidence_score=0.9, start_pos=0, end_pos=16),
            NamedEntity(text="machine learning", label="TECHNOLOGY", confidence_score=0.8, start_pos=20, end_pos=36),  # Duplicate
            NamedEntity(text="Test", label="MISC", confidence_score=0.3, start_pos=40, end_pos=44),  # Low confidence
            NamedEntity(text="Research", label="CONCEPT", confidence_score=0.7, start_pos=50, end_pos=58),
        ]
        
        filtered = analysis_agent._filter_and_enhance_entities(entities)
        
        # Should remove duplicates and low confidence entities
        assert len(filtered) == 2
        assert all(entity.confidence_score >= 0.5 for entity in filtered)
        assert all(entity.label in analysis_agent.priority_entity_types for entity in filtered)
        
        # Should keep the higher confidence duplicate
        ml_entities = [e for e in filtered if e.text.lower() == "machine learning"]
        assert len(ml_entities) == 1
        assert ml_entities[0].confidence_score == 0.9
    
    def test_enhance_topics(self, analysis_agent):
        """Test topic enhancement."""
        topics = [
            TopicModel(topic_id=0, keywords=["machine", "learning", "ai"], weight=0.7),
            TopicModel(topic_id=1, keywords=["x", "y"], weight=0.5),  # Short keywords
        ]
        
        enhanced = analysis_agent._enhance_topics(topics, ["test text"])
        
        assert len(enhanced) == 2
        assert enhanced[0].description is not None
        assert "machine, learning, ai" in enhanced[0].description
        
        # First topic should get quality bonus for good keywords
        assert enhanced[0].weight > 0.7
    
    def test_prepare_statistical_data(self, analysis_agent, sample_articles, sample_papers):
        """Test statistical data preparation."""
        data = analysis_agent._prepare_statistical_data(sample_articles, sample_papers)
        
        # Should include article metrics
        assert 'article_confidence_scores' in data
        assert 'article_content_lengths' in data
        assert 'article_key_findings_count' in data
        
        # Should include paper metrics
        assert 'paper_citation_counts' in data
        assert 'paper_publication_years' in data
        assert 'paper_abstract_lengths' in data
        assert 'paper_author_counts' in data
        
        # Verify data types and lengths
        assert len(data['article_confidence_scores']) == len(sample_articles)
        assert len(data['paper_citation_counts']) == len(sample_papers)
        assert all(isinstance(score, float) for score in data['article_confidence_scores'])
    
    def test_calculate_custom_statistics(self, analysis_agent, sample_articles, sample_papers):
        """Test custom statistics calculation."""
        stats = analysis_agent._calculate_custom_statistics(sample_articles, sample_papers)
        
        assert len(stats) > 0
        assert all(isinstance(stat, StatisticalSummary) for stat in stats)
        
        # Should include source diversity
        diversity_stats = [s for s in stats if s.metric_name == "source_diversity"]
        assert len(diversity_stats) == 1
        assert diversity_stats[0].value > 0
        
        # Should include average confidence
        confidence_stats = [s for s in stats if s.metric_name == "average_confidence"]
        assert len(confidence_stats) == 1
        assert 0 <= confidence_stats[0].value <= 1
    
    def test_enhance_key_themes(self, analysis_agent, sample_articles, sample_papers):
        """Test key themes enhancement."""
        themes = ["machine learning", "artificial intelligence", "research", "nonexistent"]
        
        enhanced = analysis_agent._enhance_key_themes(themes, sample_articles, sample_papers)
        
        assert len(enhanced) <= 10
        assert isinstance(enhanced, list)
        assert all(isinstance(theme, str) for theme in enhanced)
        
        # Themes that appear in content should be ranked higher
        assert "machine learning" in enhanced  # Appears in sample data
        assert "artificial intelligence" in enhanced  # Appears in sample data
    
    def test_generate_topic_insights(self, analysis_agent, sample_articles, sample_papers):
        """Test topic insight generation."""
        topics = [
            TopicModel(
                topic_id=0,
                keywords=["machine", "learning", "algorithm"],
                weight=0.8,
                description="Machine Learning Algorithms"
            )
        ]
        
        insights = analysis_agent._generate_topic_insights(topics, sample_articles, sample_papers)
        
        assert len(insights) > 0
        assert all(isinstance(insight, Insight) for insight in insights)
        assert all(insight.confidence_score > 0 for insight in insights)
        assert all(len(insight.summary) > 0 for insight in insights)
        assert all(len(insight.related_sources) > 0 for insight in insights)
    
    def test_generate_entity_insights(self, analysis_agent, sample_articles, sample_papers):
        """Test entity insight generation."""
        entities = [
            NamedEntity(text="Machine Learning", label="TECHNOLOGY", confidence_score=0.9, start_pos=0, end_pos=16),
            NamedEntity(text="Deep Learning", label="TECHNOLOGY", confidence_score=0.85, start_pos=20, end_pos=33),
            NamedEntity(text="Neural Networks", label="TECHNOLOGY", confidence_score=0.8, start_pos=40, end_pos=55),
        ]
        
        insights = analysis_agent._generate_entity_insights(entities, sample_articles, sample_papers)
        
        assert len(insights) > 0
        assert all(isinstance(insight, Insight) for insight in insights)
        
        # Should group by entity type
        tech_insights = [i for i in insights if "TECHNOLOGY" in i.topic]
        assert len(tech_insights) > 0
    
    def test_generate_sentiment_insights(self, analysis_agent, sample_articles, sample_papers):
        """Test sentiment insight generation."""
        sentiment_scores = {
            'overall': 0.7,
            'positive': 0.8,
            'negative': 0.1,
            'neutral': 0.1
        }
        
        insights = analysis_agent._generate_sentiment_insights(sentiment_scores, sample_articles, sample_papers)
        
        assert len(insights) > 0
        assert all(isinstance(insight, Insight) for insight in insights)
        
        # Should identify positive sentiment
        sentiment_insight = insights[0]
        assert "positive" in sentiment_insight.summary.lower()
        assert sentiment_insight.confidence_score > 0
    
    def test_generate_sentiment_insights_neutral(self, analysis_agent, sample_articles, sample_papers):
        """Test sentiment insight generation with neutral sentiment."""
        sentiment_scores = {'overall': 0.05}  # Very neutral
        
        insights = analysis_agent._generate_sentiment_insights(sentiment_scores, sample_articles, sample_papers)
        
        # Should not generate insights for very neutral sentiment
        assert len(insights) == 0
    
    def test_generate_statistical_insights(self, analysis_agent, sample_articles, sample_papers):
        """Test statistical insight generation."""
        statistics = [
            StatisticalSummary(metric_name="average_confidence", value=0.85, unit="score", sample_size=10),
            StatisticalSummary(metric_name="source_diversity", value=7.0, unit="unique_domains", sample_size=10),
            StatisticalSummary(metric_name="other_metric", value=0.5, unit="value", sample_size=5),
        ]
        
        insights = analysis_agent._generate_statistical_insights(statistics, sample_articles, sample_papers)
        
        assert len(insights) > 0
        assert all(isinstance(insight, Insight) for insight in insights)
        
        # Should generate insights for high confidence and diversity
        topics = [insight.topic for insight in insights]
        assert any("Quality" in topic for topic in topics)
        assert any("Diversity" in topic for topic in topics)
    
    async def test_health_check_success(self, analysis_agent):
        """Test successful health check."""
        health = await analysis_agent.health_check()
        
        assert health['status'] == 'healthy'
        assert health['agent_name'] == 'analysis_agent'
        assert 'code_interpreter_status' in health
        assert 'test_ner_successful' in health
        assert 'analysis_config' in health
        assert 'timestamp' in health
        
        # Verify interpreter health check was called
        analysis_agent.interpreter.health_check.assert_called_once()
    
    async def test_health_check_failure(self, analysis_agent):
        """Test health check with failure."""
        analysis_agent.interpreter.health_check.side_effect = Exception("Health check failed")
        
        health = await analysis_agent.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['agent_name'] == 'analysis_agent'
        assert 'error' in health
        assert 'timestamp' in health
    
    async def test_concurrent_analysis_operations(self, analysis_agent, sample_articles):
        """Test concurrent analysis operations."""
        # Test that multiple analysis operations can run concurrently
        tasks = [
            analysis_agent.perform_named_entity_recognition(["Test text 1"]),
            analysis_agent.perform_topic_modeling(["Test text 2"]),
            analysis_agent.perform_statistical_analysis(sample_articles, [])
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        assert len(results) == 3
        assert all(not isinstance(result, Exception) for result in results)
    
    def test_insight_class(self):
        """Test Insight class functionality."""
        sources = [Source(url="https://example.com", title="Test", source_type=SourceType.WEB)]
        
        insight = Insight(
            topic="Test Topic",
            summary="Test summary",
            supporting_evidence=["Evidence 1", "Evidence 2"],
            confidence_score=0.8,
            related_sources=sources
        )
        
        assert insight.topic == "Test Topic"
        assert insight.summary == "Test summary"
        assert len(insight.supporting_evidence) == 2
        assert insight.confidence_score == 0.8
        assert len(insight.related_sources) == 1
    
    async def test_analysis_with_minimal_content(self, analysis_agent):
        """Test analysis with articles containing minimal content."""
        minimal_articles = [
            ExtractedArticle(
                source=Source(url="https://example.com", title="Minimal", source_type=SourceType.WEB),
                content="Short content.",
                key_findings=[],
                confidence_score=0.1,
                extraction_timestamp=datetime.now()
            )
        ]
        
        # Should handle minimal content gracefully
        result = await analysis_agent.execute(articles=minimal_articles)
        assert result.success is True
    
    async def test_analysis_config_validation(self, analysis_agent, sample_articles):
        """Test analysis configuration validation and application."""
        custom_config = {
            'enable_ner': False,
            'enable_topics': True,
            'enable_sentiment': False,
            'num_topics': 3,
            'ner_model': 'en_core_web_lg'
        }
        
        result = await analysis_agent.execute(
            articles=sample_articles,
            analysis_config=custom_config
        )
        
        assert result.success is True
        
        # Verify config was passed to interpreter
        call_args = analysis_agent.interpreter.analyze_research_content.call_args
        passed_config = call_args[1]['analysis_config']
        assert passed_config['enable_ner'] is False
        assert passed_config['num_topics'] == 3
        assert passed_config['ner_model'] == 'en_core_web_lg'


@pytest.mark.asyncio
class TestAnalysisAgentIntegration:
    """Integration tests for AnalysisAgent."""
    
    async def test_full_analysis_pipeline(self, analysis_agent, sample_articles, sample_papers):
        """Test complete analysis pipeline integration."""
        result = await analysis_agent.execute(
            articles=sample_articles,
            papers=sample_papers,
            session_id="integration_test",
            analysis_config={
                'enable_ner': True,
                'enable_topics': True,
                'enable_sentiment': True,
                'enable_statistics': True
            }
        )
        
        assert result.success is True
        analysis_results = result.data
        
        # Verify all analysis components are present
        assert isinstance(analysis_results, AnalysisResults)
        assert analysis_results.session_id == "integration_test"
        assert len(analysis_results.named_entities) > 0
        assert len(analysis_results.topics) > 0
        assert len(analysis_results.key_themes) > 0
        assert analysis_results.sentiment_scores
        assert analysis_results.processing_time_seconds is not None
        
        # Verify data quality
        assert all(entity.confidence_score > 0 for entity in analysis_results.named_entities)
        assert all(topic.weight > 0 for topic in analysis_results.topics)
        assert all(isinstance(theme, str) for theme in analysis_results.key_themes)
    
    async def test_error_recovery_and_partial_results(self, analysis_agent, sample_articles):
        """Test error recovery and partial results handling."""
        # Mock partial failure in one analysis component
        analysis_agent.interpreter.perform_ner_analysis.side_effect = Exception("NER failed")
        
        # Should still complete other analysis components
        result = await analysis_agent.execute(articles=sample_articles)
        
        # The comprehensive analysis should handle individual component failures
        # and still return results from successful components
        assert result.success is True or "NER" in result.error