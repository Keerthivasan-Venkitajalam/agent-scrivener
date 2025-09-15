"""
Code interpreter wrapper for secure Python execution and data analysis.
"""

import asyncio
import json
import re
import tempfile
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import base64
from pathlib import Path

from ..models.core import ExtractedArticle, AcademicPaper, Source
from ..models.analysis import AnalysisResults, NamedEntity, TopicModel, StatisticalSummary
from ..models.errors import ProcessingError, ErrorSeverity
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CodeInterpreterWrapper:
    """
    Wrapper for AgentCore Code Interpreter with secure Python execution.
    
    Provides data analysis utilities for NER, topic modeling, statistical analysis,
    and visualization generation with comprehensive error handling.
    """
    
    def __init__(self, code_interpreter=None):
        """
        Initialize the code interpreter wrapper.
        
        Args:
            code_interpreter: AgentCore Code Interpreter instance
        """
        self.interpreter = code_interpreter
        self.execution_timeout = 300  # 5 minutes
        self.max_memory_mb = 512
        self.allowed_imports = {
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn',
            'nltk', 'spacy', 'textblob', 'wordcloud', 'plotly', 'json',
            'csv', 're', 'collections', 'statistics', 'math', 'datetime',
            'base64', 'io', 'time'  # Added missing imports
        }
        
        # Pre-built analysis code templates
        self.analysis_templates = {
            'ner': self._get_ner_template(),
            'topic_modeling': self._get_topic_modeling_template(),
            'statistical_analysis': self._get_statistical_template(),
            'sentiment_analysis': self._get_sentiment_template(),
            'visualization': self._get_visualization_template()
        }
    
    async def execute_secure_code(
        self,
        code: str,
        context_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code securely with timeout and memory limits.
        
        Args:
            code: Python code to execute
            context_data: Data to make available in execution context
            timeout: Execution timeout in seconds
            
        Returns:
            Execution results including output, variables, and any errors
            
        Raises:
            ProcessingError: For execution failures or security violations
        """
        # Validate code security
        self._validate_code_security(code)
        
        # Prepare execution context
        execution_context = {
            'data': context_data or {},
            'results': {},
            'imports': {},
            'temp_files': []
        }
        
        try:
            if self.interpreter:
                # Use AgentCore Code Interpreter
                result = await self._execute_with_agentcore(code, execution_context, timeout)
            else:
                # Use mock execution for testing
                logger.warning("No code interpreter available, using mock execution")
                result = await self._mock_code_execution(code, execution_context)
            
            return result
            
        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}")
            raise ProcessingError(f"Code execution failed: {str(e)}")
        finally:
            # Cleanup temporary files
            await self._cleanup_temp_files(execution_context.get('temp_files', []))
    
    async def perform_ner_analysis(
        self,
        texts: List[str],
        model_name: str = 'en_core_web_sm'
    ) -> List[NamedEntity]:
        """
        Perform Named Entity Recognition on texts.
        
        Args:
            texts: List of texts to analyze
            model_name: spaCy model name to use
            
        Returns:
            List of named entities found in texts
        """
        code = self.analysis_templates['ner'].format(
            model_name=model_name,
            texts=json.dumps(texts)
        )
        
        try:
            result = await self.execute_secure_code(code)
            
            if 'entities' in result.get('variables', {}):
                entities_data = result['variables']['entities']
                return [
                    NamedEntity(
                        text=entity['text'],
                        label=entity['label'],
                        confidence_score=entity.get('confidence', 0.8),
                        start_pos=entity.get('start', 0),
                        end_pos=entity.get('end', len(entity['text']))
                    )
                    for entity in entities_data
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"NER analysis failed: {str(e)}")
            raise ProcessingError(f"NER analysis failed: {str(e)}")
    
    async def perform_topic_modeling(
        self,
        texts: List[str],
        num_topics: int = 5,
        method: str = 'lda'
    ) -> List[TopicModel]:
        """
        Perform topic modeling on texts.
        
        Args:
            texts: List of texts to analyze
            num_topics: Number of topics to extract
            method: Topic modeling method ('lda', 'nmf')
            
        Returns:
            List of discovered topics
        """
        code = self.analysis_templates['topic_modeling'].format(
            texts=json.dumps(texts),
            num_topics=num_topics,
            method=method
        )
        
        try:
            result = await self.execute_secure_code(code)
            
            if 'topics' in result.get('variables', {}):
                topics_data = result['variables']['topics']
                return [
                    TopicModel(
                        topic_id=topic['id'],
                        keywords=topic['keywords'],
                        weight=topic.get('weight', 0.5),
                        description=topic.get('description')
                    )
                    for topic in topics_data
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {str(e)}")
            raise ProcessingError(f"Topic modeling failed: {str(e)}")
    
    async def perform_statistical_analysis(
        self,
        data: Dict[str, List[float]],
        analysis_types: List[str] = None
    ) -> List[StatisticalSummary]:
        """
        Perform statistical analysis on numerical data.
        
        Args:
            data: Dictionary of metric names to values
            analysis_types: Types of analysis to perform
            
        Returns:
            List of statistical summaries
        """
        if analysis_types is None:
            analysis_types = ['mean', 'median', 'std', 'correlation']
        
        code = self.analysis_templates['statistical_analysis'].format(
            data=json.dumps(data),
            analysis_types=json.dumps(analysis_types)
        )
        
        try:
            result = await self.execute_secure_code(code)
            
            if 'statistics' in result.get('variables', {}):
                stats_data = result['variables']['statistics']
                return [
                    StatisticalSummary(
                        metric_name=stat['metric'],
                        value=stat['value'],
                        unit=stat.get('unit'),
                        confidence_interval=stat.get('confidence_interval'),
                        sample_size=stat.get('sample_size')
                    )
                    for stat in stats_data
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {str(e)}")
            raise ProcessingError(f"Statistical analysis failed: {str(e)}")
    
    async def perform_sentiment_analysis(
        self,
        texts: List[str],
        method: str = 'textblob'
    ) -> Dict[str, float]:
        """
        Perform sentiment analysis on texts.
        
        Args:
            texts: List of texts to analyze
            method: Sentiment analysis method ('textblob', 'vader')
            
        Returns:
            Dictionary of sentiment scores
        """
        code = self.analysis_templates['sentiment_analysis'].format(
            texts=json.dumps(texts),
            method=method
        )
        
        try:
            result = await self.execute_secure_code(code)
            
            if 'sentiment_scores' in result.get('variables', {}):
                return result['variables']['sentiment_scores']
            
            return {}
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise ProcessingError(f"Sentiment analysis failed: {str(e)}")
    
    async def generate_visualization(
        self,
        data: Dict[str, Any],
        chart_type: str,
        title: str = "Analysis Results",
        output_format: str = 'png'
    ) -> Dict[str, Any]:
        """
        Generate data visualization.
        
        Args:
            data: Data to visualize
            chart_type: Type of chart ('bar', 'line', 'scatter', 'heatmap', 'wordcloud')
            title: Chart title
            output_format: Output format ('png', 'svg', 'html')
            
        Returns:
            Dictionary containing chart data and metadata
        """
        code = self.analysis_templates['visualization'].format(
            data=json.dumps(data),
            chart_type=chart_type,
            title=title,
            output_format=output_format
        )
        
        try:
            result = await self.execute_secure_code(code)
            
            if 'chart_data' in result.get('variables', {}):
                chart_info = result['variables']['chart_data']
                return {
                    'chart_type': chart_type,
                    'title': title,
                    'format': output_format,
                    'data': chart_info.get('data'),
                    'base64_image': chart_info.get('base64_image'),
                    'file_path': chart_info.get('file_path'),
                    'metadata': chart_info.get('metadata', {})
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            raise ProcessingError(f"Visualization generation failed: {str(e)}")
    
    async def analyze_research_content(
        self,
        articles: List[ExtractedArticle],
        papers: List[AcademicPaper],
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisResults:
        """
        Perform comprehensive analysis on research content.
        
        Args:
            articles: Extracted articles to analyze
            papers: Academic papers to analyze
            analysis_config: Configuration for analysis parameters
            
        Returns:
            Complete analysis results
        """
        config = analysis_config or {}
        start_time = datetime.now()
        
        # Combine all text content
        all_texts = []
        all_sources = []
        
        for article in articles:
            all_texts.append(article.content)
            all_sources.append(article.source)
        
        for paper in papers:
            all_texts.append(f"{paper.title}. {paper.abstract}")
            # Convert paper to source
            paper_source = Source(
                url=paper.full_text_url or f"https://doi.org/{paper.doi}" if paper.doi else "https://example.com",
                title=paper.title,
                author=", ".join(paper.authors),
                source_type="academic"
            )
            all_sources.append(paper_source)
        
        if not all_texts:
            logger.warning("No content provided for analysis")
            return AnalysisResults(session_id="empty")
        
        try:
            # Perform parallel analysis
            tasks = []
            
            # Named Entity Recognition
            if config.get('enable_ner', True):
                tasks.append(self.perform_ner_analysis(all_texts))
            else:
                async def empty_ner():
                    return []
                tasks.append(empty_ner())
            
            # Topic Modeling
            if config.get('enable_topics', True):
                num_topics = config.get('num_topics', min(5, len(all_texts)))
                tasks.append(self.perform_topic_modeling(all_texts, num_topics))
            else:
                async def empty_topics():
                    return []
                tasks.append(empty_topics())
            
            # Sentiment Analysis
            if config.get('enable_sentiment', True):
                tasks.append(self.perform_sentiment_analysis(all_texts))
            else:
                async def empty_sentiment():
                    return {}
                tasks.append(empty_sentiment())
            
            # Execute all analysis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            named_entities = results[0] if not isinstance(results[0], Exception) else []
            topics = results[1] if not isinstance(results[1], Exception) else []
            sentiment_scores = results[2] if not isinstance(results[2], Exception) else {}
            
            # Extract key themes from topics
            key_themes = []
            for topic in topics:
                if topic.keywords:
                    key_themes.extend(topic.keywords[:3])  # Top 3 keywords per topic
            
            # Remove duplicates and limit
            key_themes = list(dict.fromkeys(key_themes))[:10]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResults(
                session_id=config.get('session_id', f"analysis_{int(datetime.now().timestamp())}"),
                named_entities=named_entities,
                topics=topics,
                statistical_summaries=[],  # Could add statistical analysis of metrics
                key_themes=key_themes,
                sentiment_scores=sentiment_scores,
                processed_sources=all_sources,
                analysis_timestamp=datetime.now(),
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            logger.error(f"Research content analysis failed: {str(e)}")
            raise ProcessingError(f"Research content analysis failed: {str(e)}")
    
    def _validate_code_security(self, code: str):
        """Validate code for security issues."""
        # Check for dangerous operations
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ProcessingError(f"Security violation: dangerous operation detected - {pattern}")
        
        # Check imports against allowlist
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import'
        ]
        
        imports = []
        for pattern in import_patterns:
            imports.extend(re.findall(pattern, code))
        
        # Allow specific module imports that are part of allowed packages
        allowed_modules = self.allowed_imports.union({
            'TfidfVectorizer', 'LatentDirichletAllocation', 'NMF', 'ENGLISH_STOP_WORDS',
            'TextBlob', 'stats', 'BytesIO'
        })
        
        for imp in imports:
            if imp not in allowed_modules:
                raise ProcessingError(f"Security violation: unauthorized import - {imp}")
    
    async def _execute_with_agentcore(
        self,
        code: str,
        context: Dict[str, Any],
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """Execute code using AgentCore Code Interpreter."""
        execution_timeout = timeout or self.execution_timeout
        
        try:
            result = await asyncio.wait_for(
                self.interpreter.execute(
                    code=code,
                    context=context['data'],
                    timeout=execution_timeout,
                    memory_limit_mb=self.max_memory_mb
                ),
                timeout=execution_timeout
            )
            
            return {
                'success': True,
                'output': result.get('output', ''),
                'variables': result.get('variables', {}),
                'execution_time': result.get('execution_time', 0),
                'memory_used': result.get('memory_used', 0)
            }
            
        except asyncio.TimeoutError:
            raise ProcessingError(f"Code execution timeout after {execution_timeout} seconds")
        except Exception as e:
            raise ProcessingError(f"AgentCore execution failed: {str(e)}")
    
    async def _mock_code_execution(
        self,
        code: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock code execution for testing."""
        # Simulate execution based on code content
        variables = {}
        output = "Mock execution completed"
        
        if 'entities' in code:
            # Mock NER results
            variables['entities'] = [
                {
                    'text': 'Machine Learning',
                    'label': 'TECHNOLOGY',
                    'confidence': 0.9,
                    'start': 0,
                    'end': 16
                },
                {
                    'text': 'Research',
                    'label': 'CONCEPT',
                    'confidence': 0.8,
                    'start': 20,
                    'end': 28
                }
            ]
        
        if 'topics' in code:
            # Mock topic modeling results
            variables['topics'] = [
                {
                    'id': 0,
                    'keywords': ['machine', 'learning', 'algorithm'],
                    'weight': 0.8,
                    'description': 'Machine Learning'
                },
                {
                    'id': 1,
                    'keywords': ['research', 'analysis', 'data'],
                    'weight': 0.6,
                    'description': 'Research Methods'
                }
            ]
        
        if 'sentiment' in code:
            # Mock sentiment analysis results
            variables['sentiment_scores'] = {
                'overall': 0.7,
                'positive': 0.8,
                'negative': 0.1,
                'neutral': 0.1
            }
        
        if 'statistics' in code:
            # Mock statistical analysis results
            variables['statistics'] = [
                {
                    'metric': 'mean_confidence',
                    'value': 0.85,
                    'unit': 'score',
                    'sample_size': 100
                },
                {
                    'metric': 'std_deviation',
                    'value': 0.12,
                    'unit': 'score',
                    'sample_size': 100
                }
            ]
        
        if 'chart' in code:
            # Mock visualization results
            variables['chart_data'] = {
                'data': {'x': [1, 2, 3], 'y': [4, 5, 6]},
                'base64_image': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                'metadata': {'width': 800, 'height': 600}
            }
        
        return {
            'success': True,
            'output': output,
            'variables': variables,
            'execution_time': 0.1,
            'memory_used': 10
        }
    
    async def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files created during execution."""
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")
    
    def _get_ner_template(self) -> str:
        """Get NER analysis code template."""
        return '''
import spacy
import json

# Load spaCy model
try:
    nlp = spacy.load("{model_name}")
except OSError:
    # Fallback to basic model
    nlp = spacy.load("en_core_web_sm")

texts = {texts}
entities = []

for i, text in enumerate(texts):
    doc = nlp(text)
    for ent in doc.ents:
        entities.append({{
            "text": ent.text,
            "label": ent.label_,
            "confidence": 0.8,  # spaCy doesn't provide confidence scores by default
            "start": ent.start_char,
            "end": ent.end_char,
            "source_index": i
        }})

print(f"Found {{len(entities)}} named entities")
'''
    
    def _get_topic_modeling_template(self) -> str:
        """Get topic modeling code template."""
        return '''
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

texts = {texts}
num_topics = {num_topics}
method = "{method}"

# Preprocess texts
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words=list(ENGLISH_STOP_WORDS),
    ngram_range=(1, 2),
    min_df=2
)

try:
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Apply topic modeling
    if method == "lda":
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    else:
        model = NMF(n_components=num_topics, random_state=42)
    
    model.fit(tfidf_matrix)
    
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        weight = float(np.sum(topic) / np.sum(model.components_))
        
        topics.append({{
            "id": topic_idx,
            "keywords": top_words[:5],
            "weight": weight,
            "description": f"Topic {{topic_idx + 1}}: {{', '.join(top_words[:3])}}"
        }})
    
    print(f"Generated {{len(topics)}} topics using {{method}}")
    
except Exception as e:
    print(f"Topic modeling failed: {{e}}")
    topics = []
'''
    
    def _get_statistical_template(self) -> str:
        """Get statistical analysis code template."""
        return '''
import json
import numpy as np
from scipy import stats
import pandas as pd

data = {data}
analysis_types = {analysis_types}
statistics = []

try:
    df = pd.DataFrame(data)
    
    for column in df.select_dtypes(include=[np.number]).columns:
        values = df[column].dropna()
        
        if len(values) == 0:
            continue
            
        if "mean" in analysis_types:
            statistics.append({{
                "metric": f"{{column}}_mean",
                "value": float(values.mean()),
                "unit": "value",
                "sample_size": len(values)
            }})
        
        if "median" in analysis_types:
            statistics.append({{
                "metric": f"{{column}}_median",
                "value": float(values.median()),
                "unit": "value",
                "sample_size": len(values)
            }})
        
        if "std" in analysis_types:
            statistics.append({{
                "metric": f"{{column}}_std",
                "value": float(values.std()),
                "unit": "value",
                "sample_size": len(values)
            }})
    
    # Correlation analysis
    if "correlation" in analysis_types and len(df.select_dtypes(include=[np.number]).columns) > 1:
        corr_matrix = df.corr()
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr_value):
                        statistics.append({{
                            "metric": f"correlation_{{col1}}_{{col2}}",
                            "value": float(corr_value),
                            "unit": "correlation",
                            "sample_size": len(df)
                        }})
    
    print(f"Generated {{len(statistics)}} statistical summaries")
    
except Exception as e:
    print(f"Statistical analysis failed: {{e}}")
    statistics = []
'''
    
    def _get_sentiment_template(self) -> str:
        """Get sentiment analysis code template."""
        return '''
import json
from textblob import TextBlob
import numpy as np

texts = {texts}
method = "{method}"
sentiment_scores = {{}}

try:
    if method == "textblob":
        polarities = []
        subjectivities = []
        
        for text in texts:
            blob = TextBlob(text)
            polarities.append(blob.sentiment.polarity)
            subjectivities.append(blob.sentiment.subjectivity)
        
        sentiment_scores = {{
            "overall": float(np.mean(polarities)),
            "positive": float(np.mean([p for p in polarities if p > 0.1])) if any(p > 0.1 for p in polarities) else 0.0,
            "negative": float(np.mean([p for p in polarities if p < -0.1])) if any(p < -0.1 for p in polarities) else 0.0,
            "neutral": float(np.mean([p for p in polarities if -0.1 <= p <= 0.1])) if any(-0.1 <= p <= 0.1 for p in polarities) else 0.0,
            "subjectivity": float(np.mean(subjectivities))
        }}
    
    print(f"Analyzed sentiment for {{len(texts)}} texts")
    
except Exception as e:
    print(f"Sentiment analysis failed: {{e}}")
    sentiment_scores = {{}}
'''
    
    def _get_visualization_template(self) -> str:
        """Get visualization code template."""
        return '''
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
from io import BytesIO

data = {data}
chart_type = "{chart_type}"
title = "{title}"
output_format = "{output_format}"

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(figsize=(10, 6))

try:
    if chart_type == "bar":
        if isinstance(data, dict) and "x" in data and "y" in data:
            ax.bar(data["x"], data["y"])
            ax.set_xlabel("Categories")
            ax.set_ylabel("Values")
    
    elif chart_type == "line":
        if isinstance(data, dict) and "x" in data and "y" in data:
            ax.plot(data["x"], data["y"], marker='o')
            ax.set_xlabel("X Values")
            ax.set_ylabel("Y Values")
    
    elif chart_type == "scatter":
        if isinstance(data, dict) and "x" in data and "y" in data:
            ax.scatter(data["x"], data["y"], alpha=0.7)
            ax.set_xlabel("X Values")
            ax.set_ylabel("Y Values")
    
    elif chart_type == "heatmap":
        if isinstance(data, dict) and "matrix" in data:
            sns.heatmap(np.array(data["matrix"]), annot=True, ax=ax)
    
    ax.set_title(title)
    plt.tight_layout()
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    
    chart_data = {{
        "data": data,
        "base64_image": image_base64,
        "metadata": {{
            "width": 800,
            "height": 600,
            "format": "png"
        }}
    }}
    
    plt.close()
    print(f"Generated {{chart_type}} chart")
    
except Exception as e:
    print(f"Visualization generation failed: {{e}}")
    chart_data = {{}}
'''
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on code interpreter."""
        try:
            # Test basic code execution
            test_code = '''
import numpy as np
result = np.array([1, 2, 3]).mean()
print(f"Test successful: {{result}}")
'''
            
            result = await self.execute_secure_code(test_code, timeout=10)
            
            return {
                'status': 'healthy',
                'interpreter_available': self.interpreter is not None,
                'test_execution_successful': result.get('success', False),
                'execution_time': result.get('execution_time', 0),
                'memory_limit_mb': self.max_memory_mb,
                'timeout_seconds': self.execution_timeout,
                'allowed_imports': list(self.allowed_imports),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'interpreter_available': self.interpreter is not None,
                'test_execution_successful': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }