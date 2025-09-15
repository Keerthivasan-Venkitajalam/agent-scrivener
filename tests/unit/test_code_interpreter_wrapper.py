"""
Unit tests for CodeInterpreterWrapper.
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agent_scrivener.tools.code_interpreter_wrapper import CodeInterpreterWrapper
from agent_scrivener.models.core import ExtractedArticle, AcademicPaper, Source, SourceType
from agent_scrivener.models.analysis import AnalysisResults, NamedEntity, TopicModel, StatisticalSummary
from agent_scrivener.models.errors import ProcessingError


class TestCodeInterpreterWrapper:
    """Test cases for CodeInterpreterWrapper."""
    
    @pytest.fixture
    def mock_code_interpreter(self):
        """Mock AgentCore code interpreter."""
        mock_interpreter = AsyncMock()
        mock_interpreter.execute.return_value = {
            'output': 'Execution completed successfully',
            'variables': {'result': 42},
            'execution_time': 0.5,
            'memory_used': 50
        }
        return mock_interpreter
    
    @pytest.fixture
    def code_wrapper(self, mock_code_interpreter):
        """Code interpreter wrapper with mocked dependencies."""
        return CodeInterpreterWrapper(mock_code_interpreter)
    
    @pytest.fixture
    def code_wrapper_no_interpreter(self):
        """Code interpreter wrapper without interpreter for testing fallbacks."""
        return CodeInterpreterWrapper()
    
    @pytest.fixture
    def sample_articles(self):
        """Sample extracted articles for testing."""
        return [
            ExtractedArticle(
                source=Source(
                    url="https://example.com/article1",
                    title="Machine Learning Research",
                    source_type=SourceType.WEB
                ),
                content="Machine learning is a powerful technology for data analysis. Research shows significant improvements in accuracy.",
                confidence_score=0.9,
                key_findings=["ML improves accuracy"]
            ),
            ExtractedArticle(
                source=Source(
                    url="https://example.com/article2",
                    title="Deep Learning Applications",
                    source_type=SourceType.WEB
                ),
                content="Deep learning algorithms have revolutionized computer vision and natural language processing.",
                confidence_score=0.85,
                key_findings=["DL revolutionizes CV and NLP"]
            )
        ]
    
    @pytest.fixture
    def sample_papers(self):
        """Sample academic papers for testing."""
        return [
            AcademicPaper(
                title="Neural Networks in Healthcare",
                authors=["Dr. Smith", "Dr. Johnson"],
                abstract="This paper explores the application of neural networks in healthcare diagnostics.",
                publication_year=2023,
                database_source="arXiv",
                doi="10.1234/example.2023.001"
            ),
            AcademicPaper(
                title="Transformer Models for NLP",
                authors=["Dr. Brown"],
                abstract="A comprehensive study of transformer architectures for natural language processing tasks.",
                publication_year=2023,
                database_source="Semantic Scholar",
                citation_count=25
            )
        ]
    
    @pytest.mark.asyncio
    async def test_execute_secure_code_success(self, code_wrapper, mock_code_interpreter):
        """Test successful secure code execution."""
        code = "import numpy as np\nresult = np.array([1, 2, 3]).mean()"
        context_data = {"test": "data"}
        
        result = await code_wrapper.execute_secure_code(code, context_data)
        
        assert result['success'] is True
        assert 'output' in result
        assert 'variables' in result
        assert 'execution_time' in result
        mock_code_interpreter.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_secure_code_security_violation(self, code_wrapper):
        """Test security validation for dangerous code."""
        dangerous_code = "import os\nos.system('rm -rf /')"
        
        with pytest.raises(ProcessingError) as exc_info:
            await code_wrapper.execute_secure_code(dangerous_code)
        
        assert "Security violation" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_secure_code_unauthorized_import(self, code_wrapper):
        """Test security validation for unauthorized imports."""
        unauthorized_code = "import subprocess\nsubprocess.call(['ls'])"
        
        with pytest.raises(ProcessingError) as exc_info:
            await code_wrapper.execute_secure_code(unauthorized_code)
        
        assert "Security violation" in str(exc_info.value)
        # The dangerous operation check catches subprocess before import validation
        assert "dangerous operation detected" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_secure_code_timeout(self, code_wrapper, mock_code_interpreter):
        """Test code execution timeout handling."""
        code = "import numpy as np\nresult = np.array([1, 2, 3]).mean()"  # Use allowed import
        mock_code_interpreter.execute.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(ProcessingError) as exc_info:
            await code_wrapper.execute_secure_code(code, timeout=1)
        
        assert "timeout" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_execute_secure_code_no_interpreter(self, code_wrapper_no_interpreter):
        """Test code execution without interpreter (mock mode)."""
        code = "import numpy as np\nresult = np.array([1, 2, 3]).mean()"
        
        result = await code_wrapper_no_interpreter.execute_secure_code(code)
        
        assert result['success'] is True
        assert 'Mock execution completed' in result['output']
    
    @pytest.mark.asyncio
    async def test_perform_ner_analysis(self, code_wrapper, mock_code_interpreter):
        """Test Named Entity Recognition analysis."""
        texts = ["Apple Inc. is located in Cupertino, California.", "John Smith works at Google."]
        
        # Mock NER results
        mock_code_interpreter.execute.return_value = {
            'output': 'NER analysis completed',
            'variables': {
                'entities': [
                    {
                        'text': 'Apple Inc.',
                        'label': 'ORG',
                        'confidence': 0.95,
                        'start': 0,
                        'end': 10
                    },
                    {
                        'text': 'Cupertino',
                        'label': 'GPE',
                        'confidence': 0.9,
                        'start': 25,
                        'end': 34
                    }
                ]
            }
        }
        
        entities = await code_wrapper.perform_ner_analysis(texts)
        
        assert len(entities) == 2
        assert all(isinstance(entity, NamedEntity) for entity in entities)
        assert entities[0].text == 'Apple Inc.'
        assert entities[0].label == 'ORG'
        assert entities[0].confidence_score == 0.95
    
    @pytest.mark.asyncio
    async def test_perform_ner_analysis_no_results(self, code_wrapper, mock_code_interpreter):
        """Test NER analysis when no entities are found."""
        texts = ["This is a simple sentence with no named entities."]
        
        mock_code_interpreter.execute.return_value = {
            'output': 'NER analysis completed',
            'variables': {}  # No entities found
        }
        
        entities = await code_wrapper.perform_ner_analysis(texts)
        
        assert len(entities) == 0
    
    @pytest.mark.asyncio
    async def test_perform_topic_modeling(self, code_wrapper, mock_code_interpreter):
        """Test topic modeling analysis."""
        texts = ["Machine learning algorithms", "Deep learning neural networks", "Data science analytics"]
        
        # Mock topic modeling results
        mock_code_interpreter.execute.return_value = {
            'output': 'Topic modeling completed',
            'variables': {
                'topics': [
                    {
                        'id': 0,
                        'keywords': ['machine', 'learning', 'algorithms'],
                        'weight': 0.8,
                        'description': 'Machine Learning'
                    },
                    {
                        'id': 1,
                        'keywords': ['deep', 'neural', 'networks'],
                        'weight': 0.6,
                        'description': 'Deep Learning'
                    }
                ]
            }
        }
        
        topics = await code_wrapper.perform_topic_modeling(texts, num_topics=2)
        
        assert len(topics) == 2
        assert all(isinstance(topic, TopicModel) for topic in topics)
        assert topics[0].topic_id == 0
        assert topics[0].keywords == ['machine', 'learning', 'algorithms']
        assert topics[0].weight == 0.8
    
    @pytest.mark.asyncio
    async def test_perform_statistical_analysis(self, code_wrapper, mock_code_interpreter):
        """Test statistical analysis."""
        data = {
            'accuracy': [0.85, 0.87, 0.89, 0.91, 0.88],
            'precision': [0.82, 0.84, 0.86, 0.88, 0.85]
        }
        
        # Mock statistical results
        mock_code_interpreter.execute.return_value = {
            'output': 'Statistical analysis completed',
            'variables': {
                'statistics': [
                    {
                        'metric': 'accuracy_mean',
                        'value': 0.88,
                        'unit': 'score',
                        'sample_size': 5
                    },
                    {
                        'metric': 'precision_std',
                        'value': 0.02,
                        'unit': 'score',
                        'sample_size': 5
                    }
                ]
            }
        }
        
        stats = await code_wrapper.perform_statistical_analysis(data)
        
        assert len(stats) == 2
        assert all(isinstance(stat, StatisticalSummary) for stat in stats)
        assert stats[0].metric_name == 'accuracy_mean'
        assert stats[0].value == 0.88
        assert stats[0].sample_size == 5
    
    @pytest.mark.asyncio
    async def test_perform_sentiment_analysis(self, code_wrapper, mock_code_interpreter):
        """Test sentiment analysis."""
        texts = ["This is a great product!", "I hate this service.", "It's okay, nothing special."]
        
        # Mock sentiment results
        mock_code_interpreter.execute.return_value = {
            'output': 'Sentiment analysis completed',
            'variables': {
                'sentiment_scores': {
                    'overall': 0.1,
                    'positive': 0.8,
                    'negative': -0.7,
                    'neutral': 0.0,
                    'subjectivity': 0.6
                }
            }
        }
        
        sentiment = await code_wrapper.perform_sentiment_analysis(texts)
        
        assert isinstance(sentiment, dict)
        assert 'overall' in sentiment
        assert 'positive' in sentiment
        assert 'negative' in sentiment
        assert sentiment['overall'] == 0.1
        assert sentiment['positive'] == 0.8
    
    @pytest.mark.asyncio
    async def test_generate_visualization(self, code_wrapper, mock_code_interpreter):
        """Test visualization generation."""
        data = {'x': [1, 2, 3, 4], 'y': [2, 4, 6, 8]}
        
        # Mock visualization results
        mock_code_interpreter.execute.return_value = {
            'output': 'Visualization generated',
            'variables': {
                'chart_data': {
                    'data': data,
                    'base64_image': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                    'metadata': {'width': 800, 'height': 600}
                }
            }
        }
        
        chart = await code_wrapper.generate_visualization(data, 'line', 'Test Chart')
        
        assert chart['chart_type'] == 'line'
        assert chart['title'] == 'Test Chart'
        assert 'base64_image' in chart
        assert chart['data'] == data
        assert chart['metadata']['width'] == 800
    
    @pytest.mark.asyncio
    async def test_analyze_research_content_comprehensive(self, code_wrapper, sample_articles, sample_papers, mock_code_interpreter):
        """Test comprehensive research content analysis."""
        # Mock all analysis results
        call_count = 0
        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:  # NER
                return {
                    'variables': {
                        'entities': [
                            {'text': 'Machine Learning', 'label': 'TECHNOLOGY', 'confidence': 0.9, 'start': 0, 'end': 16}
                        ]
                    }
                }
            elif call_count == 2:  # Topic modeling
                return {
                    'variables': {
                        'topics': [
                            {'id': 0, 'keywords': ['machine', 'learning'], 'weight': 0.8, 'description': 'ML Topic'}
                        ]
                    }
                }
            elif call_count == 3:  # Sentiment
                return {
                    'variables': {
                        'sentiment_scores': {'overall': 0.7, 'positive': 0.8}
                    }
                }
            return {'variables': {}}
        
        mock_code_interpreter.execute.side_effect = mock_execute
        
        analysis_config = {
            'session_id': 'test_session',
            'enable_ner': True,
            'enable_topics': True,
            'enable_sentiment': True,
            'num_topics': 3
        }
        
        results = await code_wrapper.analyze_research_content(sample_articles, sample_papers, analysis_config)
        
        assert isinstance(results, AnalysisResults)
        assert results.session_id == 'test_session'
        assert len(results.named_entities) == 1
        assert len(results.topics) == 1
        assert len(results.sentiment_scores) == 2
        assert len(results.processed_sources) == 4  # 2 articles + 2 papers
        assert results.processing_time_seconds is not None
        assert results.processing_time_seconds >= 0  # Allow zero for very fast mock execution
    
    @pytest.mark.asyncio
    async def test_analyze_research_content_empty_input(self, code_wrapper):
        """Test analysis with empty input."""
        results = await code_wrapper.analyze_research_content([], [])
        
        assert isinstance(results, AnalysisResults)
        assert results.session_id == "empty"
        assert len(results.named_entities) == 0
        assert len(results.topics) == 0
        assert len(results.processed_sources) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_research_content_partial_config(self, code_wrapper, sample_articles, mock_code_interpreter):
        """Test analysis with partial configuration (some analyses disabled)."""
        # Mock only sentiment analysis (NER and topics disabled)
        mock_code_interpreter.execute.return_value = {
            'variables': {
                'sentiment_scores': {'overall': 0.5}
            }
        }
        
        analysis_config = {
            'enable_ner': False,
            'enable_topics': False,
            'enable_sentiment': True
        }
        
        results = await code_wrapper.analyze_research_content(sample_articles, [], analysis_config)
        
        assert len(results.named_entities) == 0  # Disabled
        assert len(results.topics) == 0  # Disabled
        assert len(results.sentiment_scores) == 1  # Enabled
    
    @pytest.mark.asyncio
    async def test_analyze_research_content_with_exceptions(self, code_wrapper, sample_articles, mock_code_interpreter):
        """Test analysis handling of partial failures."""
        # Mock one analysis to fail, others to succeed
        call_count = 0
        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:  # NER fails
                raise Exception("NER analysis failed")
            elif call_count == 2:  # Topic modeling succeeds
                return {
                    'variables': {
                        'topics': [
                            {'id': 0, 'keywords': ['test'], 'weight': 0.5}
                        ]
                    }
                }
            elif call_count == 3:  # Sentiment succeeds
                return {
                    'variables': {
                        'sentiment_scores': {'overall': 0.6}
                    }
                }
            return {'variables': {}}
        
        mock_code_interpreter.execute.side_effect = mock_execute
        
        results = await code_wrapper.analyze_research_content(sample_articles, [])
        
        # Should handle partial failures gracefully
        assert len(results.named_entities) == 0  # Failed
        assert len(results.topics) == 1  # Succeeded
        assert len(results.sentiment_scores) == 1  # Succeeded
    
    def test_validate_code_security_safe_code(self, code_wrapper):
        """Test security validation for safe code."""
        safe_code = """
import numpy as np
import pandas as pd
data = np.array([1, 2, 3])
result = data.mean()
"""
        
        # Should not raise any exception
        code_wrapper._validate_code_security(safe_code)
    
    def test_validate_code_security_dangerous_operations(self, code_wrapper):
        """Test security validation catches dangerous operations."""
        dangerous_codes = [
            "import os\nos.system('rm -rf /')",
            "import subprocess\nsubprocess.call(['ls'])",
            "eval('malicious_code')",
            "exec('dangerous_code')",
            "open('/etc/passwd', 'r')",
            "__import__('os').system('ls')"
        ]
        
        for code in dangerous_codes:
            with pytest.raises(ProcessingError) as exc_info:
                code_wrapper._validate_code_security(code)
            assert "Security violation" in str(exc_info.value)
    
    def test_validate_code_security_unauthorized_imports(self, code_wrapper):
        """Test security validation for unauthorized imports."""
        unauthorized_codes = [
            "import requests",
            "import urllib",
            "import socket",
            "import threading"
        ]
        
        for code in unauthorized_codes:
            with pytest.raises(ProcessingError) as exc_info:
                code_wrapper._validate_code_security(code)
            assert "unauthorized import" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_mock_code_execution_ner(self, code_wrapper_no_interpreter):
        """Test mock execution for NER code."""
        code = "entities = perform_ner_analysis(texts)"
        
        result = await code_wrapper_no_interpreter.execute_secure_code(code)
        
        assert result['success'] is True
        assert 'entities' in result['variables']
        assert len(result['variables']['entities']) > 0
        assert result['variables']['entities'][0]['text'] == 'Machine Learning'
    
    @pytest.mark.asyncio
    async def test_mock_code_execution_topics(self, code_wrapper_no_interpreter):
        """Test mock execution for topic modeling code."""
        code = "topics = perform_topic_modeling(texts)"
        
        result = await code_wrapper_no_interpreter.execute_secure_code(code)
        
        assert result['success'] is True
        assert 'topics' in result['variables']
        assert len(result['variables']['topics']) > 0
        assert result['variables']['topics'][0]['keywords'] == ['machine', 'learning', 'algorithm']
    
    @pytest.mark.asyncio
    async def test_mock_code_execution_sentiment(self, code_wrapper_no_interpreter):
        """Test mock execution for sentiment analysis code."""
        code = "sentiment_scores = analyze_sentiment(texts)"
        
        result = await code_wrapper_no_interpreter.execute_secure_code(code)
        
        assert result['success'] is True
        assert 'sentiment_scores' in result['variables']
        assert 'overall' in result['variables']['sentiment_scores']
        assert result['variables']['sentiment_scores']['overall'] == 0.7
    
    @pytest.mark.asyncio
    async def test_mock_code_execution_statistics(self, code_wrapper_no_interpreter):
        """Test mock execution for statistical analysis code."""
        code = "statistics = calculate_statistics(data)"
        
        result = await code_wrapper_no_interpreter.execute_secure_code(code)
        
        assert result['success'] is True
        assert 'statistics' in result['variables']
        assert len(result['variables']['statistics']) > 0
        assert result['variables']['statistics'][0]['metric'] == 'mean_confidence'
    
    @pytest.mark.asyncio
    async def test_mock_code_execution_visualization(self, code_wrapper_no_interpreter):
        """Test mock execution for visualization code."""
        code = "chart_data = generate_chart(data)"
        
        result = await code_wrapper_no_interpreter.execute_secure_code(code)
        
        assert result['success'] is True
        assert 'chart_data' in result['variables']
        assert 'base64_image' in result['variables']['chart_data']
        assert 'metadata' in result['variables']['chart_data']
    
    def test_analysis_templates_exist(self, code_wrapper):
        """Test that all analysis templates are available."""
        expected_templates = ['ner', 'topic_modeling', 'statistical_analysis', 'sentiment_analysis', 'visualization']
        
        for template_name in expected_templates:
            assert template_name in code_wrapper.analysis_templates
            assert isinstance(code_wrapper.analysis_templates[template_name], str)
            assert len(code_wrapper.analysis_templates[template_name]) > 0
    
    def test_analysis_templates_formatting(self, code_wrapper):
        """Test that analysis templates can be formatted with parameters."""
        # Test NER template
        ner_code = code_wrapper.analysis_templates['ner'].format(
            model_name='en_core_web_sm',
            texts='["test text"]'
        )
        assert 'en_core_web_sm' in ner_code
        assert '["test text"]' in ner_code
        
        # Test topic modeling template
        topic_code = code_wrapper.analysis_templates['topic_modeling'].format(
            texts='["test"]',
            num_topics=3,
            method='lda'
        )
        assert 'num_topics = 3' in topic_code
        assert 'method = "lda"' in topic_code
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, code_wrapper, mock_code_interpreter):
        """Test health check when system is healthy."""
        mock_code_interpreter.execute.return_value = {
            'output': 'Test successful',
            'variables': {'result': 2.0},
            'execution_time': 0.1,
            'memory_used': 10
        }
        
        health = await code_wrapper.health_check()
        
        assert health['status'] == 'healthy'
        assert health['interpreter_available'] is True
        assert health['test_execution_successful'] is True
        assert health['execution_time'] == 0.1
        assert health['memory_limit_mb'] == 512
        assert 'numpy' in health['allowed_imports']
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, code_wrapper, mock_code_interpreter):
        """Test health check when system is unhealthy."""
        mock_code_interpreter.execute.side_effect = Exception("Interpreter failed")
        
        health = await code_wrapper.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['test_execution_successful'] is False
        assert 'Interpreter failed' in health['error']
    
    @pytest.mark.asyncio
    async def test_health_check_no_interpreter(self, code_wrapper_no_interpreter):
        """Test health check without interpreter."""
        health = await code_wrapper_no_interpreter.health_check()
        
        assert health['interpreter_available'] is False
        # Should still work with mock execution
        assert health['test_execution_successful'] is True
    
    @pytest.mark.asyncio
    async def test_cleanup_temp_files(self, code_wrapper):
        """Test temporary file cleanup."""
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_file_path = tmp.name
            tmp.write(b"test content")
        
        # Verify file exists
        assert os.path.exists(temp_file_path)
        
        # Clean up
        await code_wrapper._cleanup_temp_files([temp_file_path])
        
        # Verify file is removed
        assert not os.path.exists(temp_file_path)
    
    @pytest.mark.asyncio
    async def test_cleanup_temp_files_nonexistent(self, code_wrapper):
        """Test cleanup of non-existent files doesn't raise errors."""
        # Should not raise any exception
        await code_wrapper._cleanup_temp_files(["/nonexistent/file.txt"])
    
    def test_allowed_imports_configuration(self, code_wrapper):
        """Test that allowed imports are properly configured."""
        expected_imports = {
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn',
            'nltk', 'spacy', 'textblob', 'wordcloud', 'plotly', 'json',
            'csv', 're', 'collections', 'statistics', 'math', 'datetime',
            'base64', 'io', 'time'  # Added missing imports
        }
        
        assert code_wrapper.allowed_imports == expected_imports
    
    def test_execution_limits_configuration(self, code_wrapper):
        """Test that execution limits are properly configured."""
        assert code_wrapper.execution_timeout == 300  # 5 minutes
        assert code_wrapper.max_memory_mb == 512
        assert isinstance(code_wrapper.allowed_imports, set)
        assert len(code_wrapper.allowed_imports) > 0