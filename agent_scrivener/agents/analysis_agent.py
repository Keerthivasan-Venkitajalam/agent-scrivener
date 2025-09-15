"""
Analysis Agent for NLP capabilities and insight generation.

Implements named entity recognition, topic modeling, statistical analysis,
and insight generation for Agent Scrivener research content.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import statistics
from collections import Counter, defaultdict

from .base import BaseAgent, AgentResult
from ..models.core import ExtractedArticle, AcademicPaper, Source
from ..models.analysis import AnalysisResults, NamedEntity, TopicModel, StatisticalSummary
from ..models.errors import ProcessingError, ValidationError
from ..tools.code_interpreter_wrapper import CodeInterpreterWrapper
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Insight:
    """Represents a research insight generated from analysis."""
    
    def __init__(self, topic: str, summary: str, supporting_evidence: List[str], 
                 confidence_score: float, related_sources: List[Source]):
        self.topic = topic
        self.summary = summary
        self.supporting_evidence = supporting_evidence
        self.confidence_score = confidence_score
        self.related_sources = related_sources


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent for comprehensive NLP analysis and insight generation.
    
    Performs named entity recognition, topic modeling, statistical analysis,
    and generates structured insights from research content.
    """
    
    def __init__(self, code_interpreter_wrapper: CodeInterpreterWrapper, name: str = "analysis_agent"):
        """
        Initialize the Analysis Agent.
        
        Args:
            code_interpreter_wrapper: CodeInterpreterWrapper instance for NLP processing
            name: Agent name for identification
        """
        super().__init__(name)
        self.interpreter = code_interpreter_wrapper
        self.min_confidence_threshold = 0.6
        self.max_topics = 10
        self.max_entities_per_text = 50
        
        # Analysis configuration
        self.analysis_config = {
            'enable_ner': True,
            'enable_topics': True,
            'enable_sentiment': True,
            'enable_statistics': True,
            'ner_model': 'en_core_web_sm',
            'topic_method': 'lda',
            'sentiment_method': 'textblob'
        }
        
        # Entity type priorities for filtering
        self.priority_entity_types = [
            'PERSON', 'ORG', 'GPE', 'TECHNOLOGY', 'CONCEPT', 
            'METHOD', 'PRODUCT', 'EVENT', 'LAW', 'MONEY'
        ]
    
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute analysis agent with comprehensive NLP analysis.
        
        Args:
            articles: List of ExtractedArticle objects to analyze
            papers: List of AcademicPaper objects to analyze (optional)
            session_id: Session identifier for tracking (optional)
            analysis_config: Configuration overrides (optional)
            
        Returns:
            AgentResult with AnalysisResults
        """
        return await self._execute_with_timing(self._perform_comprehensive_analysis, **kwargs)
    
    async def _perform_comprehensive_analysis(
        self, 
        articles: List[ExtractedArticle], 
        papers: Optional[List[AcademicPaper]] = None,
        session_id: Optional[str] = None,
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisResults:
        """
        Perform comprehensive analysis on research content.
        
        Args:
            articles: List of extracted articles to analyze
            papers: List of academic papers to analyze
            session_id: Session identifier
            analysis_config: Configuration overrides
            
        Returns:
            Complete analysis results
        """
        # Validate input
        self.validate_input(
            {"articles": articles}, 
            ["articles"]
        )
        
        if not articles:
            raise ValidationError("At least one article must be provided for analysis")
        
        papers = papers or []
        session_id = session_id or f"analysis_{int(datetime.now().timestamp())}"
        config = {**self.analysis_config, **(analysis_config or {})}
        
        logger.info(f"Starting comprehensive analysis for session {session_id}")
        logger.info(f"Analyzing {len(articles)} articles and {len(papers)} papers")
        
        start_time = datetime.now()
        
        try:
            # Use code interpreter for comprehensive analysis
            analysis_results = await self.interpreter.analyze_research_content(
                articles=articles,
                papers=papers,
                analysis_config=config
            )
            
            # Enhance results with additional insights
            enhanced_results = await self._enhance_analysis_results(
                analysis_results, articles, papers, config
            )
            
            # Generate structured insights
            insights = await self._generate_insights(enhanced_results, articles, papers)
            
            # Add insights to results (store as metadata since AnalysisResults doesn't have insights field)
            enhanced_results.key_themes.extend([insight.topic for insight in insights[:5]])
            enhanced_results.key_themes = list(dict.fromkeys(enhanced_results.key_themes))[:10]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            enhanced_results.processing_time_seconds = processing_time
            enhanced_results.session_id = session_id
            
            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            logger.info(f"Generated {len(enhanced_results.named_entities)} entities, "
                       f"{len(enhanced_results.topics)} topics, {len(insights)} insights")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise ProcessingError(f"Analysis failed: {str(e)}")
    
    async def perform_named_entity_recognition(
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
            List of named entities with confidence scores
        """
        logger.info(f"Performing NER on {len(texts)} texts using {model_name}")
        
        try:
            # Use code interpreter for NER
            entities = await self.interpreter.perform_ner_analysis(texts, model_name)
            
            # Filter and enhance entities
            filtered_entities = self._filter_and_enhance_entities(entities)
            
            logger.info(f"NER completed: {len(filtered_entities)} entities extracted")
            return filtered_entities
            
        except Exception as e:
            logger.error(f"NER analysis failed: {str(e)}")
            raise ProcessingError(f"Named entity recognition failed: {str(e)}")
    
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
            List of discovered topics with keywords and weights
        """
        logger.info(f"Performing topic modeling on {len(texts)} texts using {method}")
        
        try:
            # Adjust number of topics based on content volume
            adjusted_topics = min(num_topics, max(2, len(texts) // 2))
            
            # Use code interpreter for topic modeling
            topics = await self.interpreter.perform_topic_modeling(
                texts, adjusted_topics, method
            )
            
            # Enhance topics with descriptions and quality scores
            enhanced_topics = self._enhance_topics(topics, texts)
            
            logger.info(f"Topic modeling completed: {len(enhanced_topics)} topics generated")
            return enhanced_topics
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {str(e)}")
            raise ProcessingError(f"Topic modeling failed: {str(e)}")
    
    async def perform_statistical_analysis(
        self, 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[StatisticalSummary]:
        """
        Perform statistical analysis on research content metrics.
        
        Args:
            articles: List of extracted articles
            papers: List of academic papers
            
        Returns:
            List of statistical summaries
        """
        logger.info(f"Performing statistical analysis on {len(articles)} articles and {len(papers)} papers")
        
        try:
            # Prepare numerical data for analysis
            data = self._prepare_statistical_data(articles, papers)
            
            if not data:
                logger.warning("No numerical data available for statistical analysis")
                return []
            
            # Use code interpreter for statistical analysis
            statistics_results = await self.interpreter.perform_statistical_analysis(
                data, ['mean', 'median', 'std', 'correlation']
            )
            
            # Add custom statistical summaries
            custom_stats = self._calculate_custom_statistics(articles, papers)
            statistics_results.extend(custom_stats)
            
            logger.info(f"Statistical analysis completed: {len(statistics_results)} summaries generated")
            return statistics_results
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {str(e)}")
            raise ProcessingError(f"Statistical analysis failed: {str(e)}")
    
    async def _enhance_analysis_results(
        self, 
        results: AnalysisResults, 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper],
        config: Dict[str, Any]
    ) -> AnalysisResults:
        """
        Enhance analysis results with additional processing.
        
        Args:
            results: Initial analysis results
            articles: Source articles
            papers: Source papers
            config: Analysis configuration
            
        Returns:
            Enhanced analysis results
        """
        # Enhance named entities with context
        if results.named_entities:
            results.named_entities = self._add_entity_context(results.named_entities, articles, papers)
        
        # Enhance topics with quality scores
        if results.topics:
            results.topics = self._add_topic_quality_scores(results.topics, articles, papers)
        
        # Add statistical summaries if not present
        if not results.statistical_summaries and config.get('enable_statistics', True):
            results.statistical_summaries = await self.perform_statistical_analysis(articles, papers)
        
        # Enhance key themes with frequency analysis
        results.key_themes = self._enhance_key_themes(results.key_themes, articles, papers)
        
        return results
    
    async def _generate_insights(
        self, 
        analysis_results: AnalysisResults, 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[Insight]:
        """
        Generate structured insights from analysis results.
        
        Args:
            analysis_results: Complete analysis results
            articles: Source articles
            papers: Source papers
            
        Returns:
            List of generated insights
        """
        logger.info("Generating structured insights from analysis results")
        
        insights = []
        
        try:
            # Generate insights from topics
            topic_insights = self._generate_topic_insights(analysis_results.topics, articles, papers)
            insights.extend(topic_insights)
            
            # Generate insights from named entities
            entity_insights = self._generate_entity_insights(analysis_results.named_entities, articles, papers)
            insights.extend(entity_insights)
            
            # Generate insights from sentiment analysis
            if analysis_results.sentiment_scores:
                sentiment_insights = self._generate_sentiment_insights(
                    analysis_results.sentiment_scores, articles, papers
                )
                insights.extend(sentiment_insights)
            
            # Generate insights from statistical analysis
            if analysis_results.statistical_summaries:
                statistical_insights = self._generate_statistical_insights(
                    analysis_results.statistical_summaries, articles, papers
                )
                insights.extend(statistical_insights)
            
            # Rank and filter insights by confidence
            ranked_insights = sorted(insights, key=lambda x: x.confidence_score, reverse=True)
            filtered_insights = [i for i in ranked_insights if i.confidence_score >= self.min_confidence_threshold]
            
            logger.info(f"Generated {len(filtered_insights)} high-confidence insights")
            return filtered_insights[:20]  # Return top 20 insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            return []
    
    def _filter_and_enhance_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Filter and enhance named entities."""
        # Group entities by text to remove duplicates
        entity_groups = defaultdict(list)
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            entity_groups[key].append(entity)
        
        # Keep the entity with highest confidence from each group
        filtered_entities = []
        for group in entity_groups.values():
            best_entity = max(group, key=lambda x: x.confidence_score)
            
            # Only keep entities with reasonable confidence and priority types
            if (best_entity.confidence_score >= 0.5 and 
                best_entity.label in self.priority_entity_types):
                filtered_entities.append(best_entity)
        
        # Sort by confidence and limit count
        filtered_entities.sort(key=lambda x: x.confidence_score, reverse=True)
        return filtered_entities[:self.max_entities_per_text]
    
    def _enhance_topics(self, topics: List[TopicModel], texts: List[str]) -> List[TopicModel]:
        """Enhance topics with better descriptions and quality scores."""
        enhanced_topics = []
        
        for topic in topics:
            # Create better description
            if len(topic.keywords) >= 3:
                description = f"Topic focusing on {', '.join(topic.keywords[:3])}"
            else:
                description = f"Topic: {', '.join(topic.keywords)}"
            
            # Adjust weight based on keyword quality
            quality_bonus = 0.0
            for keyword in topic.keywords:
                if len(keyword) > 3 and keyword.isalpha():
                    quality_bonus += 0.05
            
            enhanced_topic = TopicModel(
                topic_id=topic.topic_id,
                keywords=topic.keywords,
                weight=min(topic.weight + quality_bonus, 1.0),
                description=description
            )
            enhanced_topics.append(enhanced_topic)
        
        return enhanced_topics
    
    def _prepare_statistical_data(
        self, 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> Dict[str, List[float]]:
        """Prepare numerical data for statistical analysis."""
        data = {}
        
        # Article metrics
        if articles:
            data['article_confidence_scores'] = [a.confidence_score for a in articles]
            data['article_content_lengths'] = [len(a.content) for a in articles]
            data['article_key_findings_count'] = [len(a.key_findings) for a in articles]
        
        # Paper metrics
        if papers:
            data['paper_citation_counts'] = [p.citation_count for p in papers]
            data['paper_publication_years'] = [float(p.publication_year) for p in papers]
            data['paper_abstract_lengths'] = [len(p.abstract) for p in papers]
            data['paper_author_counts'] = [len(p.authors) for p in papers]
        
        return data
    
    def _calculate_custom_statistics(
        self, 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[StatisticalSummary]:
        """Calculate custom statistical summaries."""
        summaries = []
        
        try:
            # Source diversity
            if articles:
                domains = [article.source.metadata.get('domain', '') for article in articles]
                unique_domains = len(set(filter(None, domains)))
                summaries.append(StatisticalSummary(
                    metric_name="source_diversity",
                    value=float(unique_domains),
                    unit="unique_domains",
                    sample_size=len(articles)
                ))
            
            # Average confidence
            if articles:
                avg_confidence = statistics.mean([a.confidence_score for a in articles])
                summaries.append(StatisticalSummary(
                    metric_name="average_confidence",
                    value=avg_confidence,
                    unit="score",
                    sample_size=len(articles)
                ))
            
            # Citation distribution
            if papers:
                citations = [p.citation_count for p in papers]
                if citations:
                    summaries.append(StatisticalSummary(
                        metric_name="median_citations",
                        value=float(statistics.median(citations)),
                        unit="citations",
                        sample_size=len(papers)
                    ))
            
            # Temporal distribution
            if papers:
                years = [p.publication_year for p in papers]
                if years:
                    year_range = max(years) - min(years)
                    summaries.append(StatisticalSummary(
                        metric_name="publication_year_range",
                        value=float(year_range),
                        unit="years",
                        sample_size=len(papers)
                    ))
        
        except Exception as e:
            logger.warning(f"Custom statistics calculation failed: {str(e)}")
        
        return summaries
    
    def _add_entity_context(
        self, 
        entities: List[NamedEntity], 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[NamedEntity]:
        """Add context information to named entities."""
        # For now, return entities as-is
        # In a more sophisticated implementation, we could add source tracking
        return entities
    
    def _add_topic_quality_scores(
        self, 
        topics: List[TopicModel], 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[TopicModel]:
        """Add quality scores to topics based on source quality."""
        # Calculate average source quality
        source_qualities = []
        
        for article in articles:
            quality = article.source.metadata.get('quality_score', 0.5)
            source_qualities.append(quality)
        
        avg_source_quality = statistics.mean(source_qualities) if source_qualities else 0.5
        
        # Adjust topic weights based on source quality
        enhanced_topics = []
        for topic in topics:
            adjusted_weight = topic.weight * (0.7 + 0.3 * avg_source_quality)
            enhanced_topic = TopicModel(
                topic_id=topic.topic_id,
                keywords=topic.keywords,
                weight=min(adjusted_weight, 1.0),
                description=topic.description
            )
            enhanced_topics.append(enhanced_topic)
        
        return enhanced_topics
    
    def _enhance_key_themes(
        self, 
        themes: List[str], 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[str]:
        """Enhance key themes with frequency analysis."""
        # Count theme frequency across all content
        theme_counts = Counter()
        
        # Count in article content
        for article in articles:
            content_lower = article.content.lower()
            for theme in themes:
                if theme.lower() in content_lower:
                    theme_counts[theme] += 1
        
        # Count in paper abstracts
        for paper in papers:
            abstract_lower = paper.abstract.lower()
            for theme in themes:
                if theme.lower() in abstract_lower:
                    theme_counts[theme] += 1
        
        # Sort themes by frequency and return top themes
        sorted_themes = [theme for theme, count in theme_counts.most_common()]
        
        # Add any themes that weren't counted but were in original list
        for theme in themes:
            if theme not in sorted_themes:
                sorted_themes.append(theme)
        
        return sorted_themes[:10]
    
    def _generate_topic_insights(
        self, 
        topics: List[TopicModel], 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[Insight]:
        """Generate insights from topic modeling results."""
        insights = []
        
        for topic in topics[:5]:  # Top 5 topics
            # Create insight summary
            summary = f"Analysis reveals significant focus on {topic.description.lower()}. "
            summary += f"Key concepts include: {', '.join(topic.keywords[:3])}."
            
            # Find supporting evidence
            evidence = []
            for article in articles[:3]:  # Sample from top articles
                if any(keyword.lower() in article.content.lower() for keyword in topic.keywords):
                    evidence.append(f"Referenced in: {article.source.title}")
            
            # Related sources
            related_sources = []
            for article in articles:
                if any(keyword.lower() in article.content.lower() for keyword in topic.keywords):
                    related_sources.append(article.source)
            
            insight = Insight(
                topic=f"Topic: {topic.description}",
                summary=summary,
                supporting_evidence=evidence[:5],
                confidence_score=topic.weight,
                related_sources=related_sources[:5]
            )
            insights.append(insight)
        
        return insights
    
    def _generate_entity_insights(
        self, 
        entities: List[NamedEntity], 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[Insight]:
        """Generate insights from named entity analysis."""
        insights = []
        
        # Group entities by type
        entity_groups = defaultdict(list)
        for entity in entities:
            entity_groups[entity.label].append(entity)
        
        # Generate insights for each entity type
        for entity_type, type_entities in entity_groups.items():
            if len(type_entities) >= 3:  # Only for types with multiple entities
                top_entities = sorted(type_entities, key=lambda x: x.confidence_score, reverse=True)[:3]
                
                summary = f"Analysis identifies key {entity_type.lower()} entities: "
                summary += f"{', '.join([e.text for e in top_entities])}. "
                summary += f"These appear frequently across the research content."
                
                evidence = [f"Entity '{entity.text}' found with {entity.confidence_score:.2f} confidence" 
                           for entity in top_entities]
                
                insight = Insight(
                    topic=f"Key {entity_type} Entities",
                    summary=summary,
                    supporting_evidence=evidence,
                    confidence_score=statistics.mean([e.confidence_score for e in top_entities]),
                    related_sources=[article.source for article in articles[:3]]
                )
                insights.append(insight)
        
        return insights
    
    def _generate_sentiment_insights(
        self, 
        sentiment_scores: Dict[str, float], 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[Insight]:
        """Generate insights from sentiment analysis."""
        insights = []
        
        overall_sentiment = sentiment_scores.get('overall', 0.0)
        
        if abs(overall_sentiment) > 0.1:  # Only if sentiment is notable
            if overall_sentiment > 0.1:
                sentiment_desc = "positive"
            elif overall_sentiment < -0.1:
                sentiment_desc = "negative"
            else:
                sentiment_desc = "neutral"
            
            summary = f"Sentiment analysis reveals a {sentiment_desc} tone across the research content "
            summary += f"(score: {overall_sentiment:.2f}). "
            
            if sentiment_desc == "positive":
                summary += "This suggests optimistic findings and promising developments in the field."
            elif sentiment_desc == "negative":
                summary += "This may indicate challenges or critical perspectives in the research."
            
            evidence = [f"Overall sentiment score: {overall_sentiment:.2f}"]
            for key, value in sentiment_scores.items():
                if key != 'overall':
                    evidence.append(f"{key.title()} sentiment: {value:.2f}")
            
            insight = Insight(
                topic="Research Sentiment Analysis",
                summary=summary,
                supporting_evidence=evidence[:5],
                confidence_score=min(abs(overall_sentiment) * 2, 1.0),
                related_sources=[article.source for article in articles[:3]]
            )
            insights.append(insight)
        
        return insights
    
    def _generate_statistical_insights(
        self, 
        statistics: List[StatisticalSummary], 
        articles: List[ExtractedArticle], 
        papers: List[AcademicPaper]
    ) -> List[Insight]:
        """Generate insights from statistical analysis."""
        insights = []
        
        # Find notable statistical patterns
        for stat in statistics:
            if stat.metric_name == "average_confidence" and stat.value > 0.8:
                summary = f"Statistical analysis shows high average confidence ({stat.value:.2f}) "
                summary += "across extracted sources, indicating reliable research quality."
                
                insight = Insight(
                    topic="Source Quality Assessment",
                    summary=summary,
                    supporting_evidence=[f"Average confidence: {stat.value:.2f} from {stat.sample_size} sources"],
                    confidence_score=0.8,
                    related_sources=[article.source for article in articles[:3]]
                )
                insights.append(insight)
            
            elif stat.metric_name == "source_diversity" and stat.value >= 5:
                summary = f"Analysis draws from {int(stat.value)} diverse sources, "
                summary += "providing comprehensive coverage of the research topic."
                
                insight = Insight(
                    topic="Source Diversity Analysis",
                    summary=summary,
                    supporting_evidence=[f"Unique domains: {int(stat.value)}"],
                    confidence_score=0.7,
                    related_sources=[article.source for article in articles[:3]]
                )
                insights.append(insight)
        
        return insights
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on analysis agent.
        
        Returns:
            Health check results
        """
        try:
            # Test basic functionality
            interpreter_health = await self.interpreter.health_check()
            
            # Test with minimal data
            test_texts = ["This is a test sentence about machine learning research."]
            test_entities = await self.interpreter.perform_ner_analysis(test_texts)
            
            return {
                'status': 'healthy',
                'agent_name': self.name,
                'code_interpreter_status': interpreter_health.get('status', 'unknown'),
                'test_ner_successful': len(test_entities) >= 0,
                'analysis_config': self.analysis_config,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'agent_name': self.name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }