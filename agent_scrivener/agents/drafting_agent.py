"""
Drafting Agent for content synthesis and document generation.

Implements content synthesis, section generation, Markdown formatting,
and table of contents generation for Agent Scrivener research documents.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .base import BaseAgent, AgentResult
from ..models.core import Insight, DocumentSections, DocumentSection, Source
from ..models.errors import ProcessingError, ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DraftingAgent(BaseAgent):
    """
    Drafting Agent for comprehensive content synthesis and document generation.
    
    Synthesizes analyzed research insights into coherent prose, generates
    document sections, and formats the final research document with proper
    Markdown structure and table of contents.
    """
    
    def __init__(self, name: str = "drafting_agent"):
        """
        Initialize the Drafting Agent.
        
        Args:
            name: Agent name for identification
        """
        super().__init__(name)
        self.logger = get_logger(f"agent_scrivener.agents.{name}")
    
    async def execute(self, insights: List[Insight], session_id: str, 
                     original_query: str, **kwargs) -> AgentResult:
        """
        Execute the drafting process to generate a complete research document.
        
        Args:
            insights: List of structured insights from analysis
            session_id: Research session identifier
            original_query: Original research query
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Result containing DocumentSections and final document
        """
        return await self._execute_with_timing(
            self._generate_document,
            insights=insights,
            session_id=session_id,
            original_query=original_query,
            **kwargs
        )
    
    async def _generate_document(self, insights: List[Insight], session_id: str,
                               original_query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate complete document from insights.
        
        Args:
            insights: List of structured insights
            session_id: Session identifier
            original_query: Original research query
            **kwargs: Additional parameters
            
        Returns:
            Dict containing document sections and final document
        """
        try:
            # Validate inputs
            self.validate_input(
                {"insights": insights, "session_id": session_id, "original_query": original_query},
                ["insights", "session_id", "original_query"]
            )
            
            # Additional validation for None values
            if session_id is None:
                raise ValidationError("session_id cannot be None")
            if original_query is None:
                raise ValidationError("original_query cannot be None")
            
            if not insights:
                raise ValidationError("No insights provided for document generation")
            
            self.logger.info(f"Generating document from {len(insights)} insights")
            
            # Generate document sections
            document_sections = await self._generate_document_sections(
                insights, original_query, session_id
            )
            
            # Generate table of contents
            table_of_contents = self._generate_table_of_contents(document_sections)
            document_sections.table_of_contents = table_of_contents
            
            # Format final document
            final_document = await self._format_final_document(document_sections)
            
            self.logger.info("Document generation completed successfully")
            
            return {
                "document_sections": document_sections,
                "final_document": final_document,
                "word_count": len(final_document.split()),
                "section_count": len(document_sections.get_all_sections())
            }
            
        except Exception as e:
            self.logger.error(f"Document generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate document: {str(e)}")
    
    async def _generate_document_sections(self, insights: List[Insight], 
                                        original_query: str, session_id: str) -> DocumentSections:
        """
        Generate structured document sections from insights.
        
        Args:
            insights: List of insights to synthesize
            original_query: Original research query
            session_id: Session identifier
            
        Returns:
            DocumentSections: Complete document structure
        """
        # Group insights by topic for better organization
        topic_groups = self._group_insights_by_topic(insights)
        
        # Generate core sections
        introduction = await self._generate_introduction_section(original_query, insights)
        methodology = await self._generate_methodology_section(insights, session_id)
        findings = await self._generate_findings_section(topic_groups, insights)
        conclusion = await self._generate_conclusion_section(insights, original_query)
        
        # Generate additional sections if needed
        additional_sections = await self._generate_additional_sections(topic_groups)
        
        return DocumentSections(
            introduction=introduction,
            methodology=methodology,
            findings=findings,
            conclusion=conclusion,
            sections=additional_sections
        )
    
    def _group_insights_by_topic(self, insights: List[Insight]) -> Dict[str, List[Insight]]:
        """
        Group insights by topic for better organization.
        
        Args:
            insights: List of insights to group
            
        Returns:
            Dict mapping topics to lists of insights
        """
        topic_groups = {}
        
        for insight in insights:
            topic = insight.topic.lower().strip()
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(insight)
        
        # Sort groups by average confidence score
        for topic in topic_groups:
            topic_groups[topic].sort(key=lambda x: x.confidence_score, reverse=True)
        
        return topic_groups
    
    async def _generate_introduction_section(self, original_query: str, 
                                           insights: List[Insight]) -> DocumentSection:
        """
        Generate introduction section.
        
        Args:
            original_query: Original research query
            insights: List of insights for context
            
        Returns:
            DocumentSection: Introduction section
        """
        # Extract key themes from insights
        key_themes = list(set([insight.topic for insight in insights[:5]]))
        
        # Generate introduction content
        content = f"""## Introduction

This research document presents a comprehensive analysis of "{original_query}". 
The investigation encompasses multiple dimensions of this topic, drawing from 
diverse sources to provide a thorough understanding of the current state of knowledge.

The analysis focuses on several key areas: {', '.join(key_themes[:3])}{'...' if len(key_themes) > 3 else ''}. 
Through systematic examination of available literature and data sources, this document 
aims to synthesize current understanding and identify significant patterns and insights.

The research methodology employed a multi-agent approach, combining web-based research, 
academic database queries, and advanced analytical techniques to ensure comprehensive 
coverage of the topic. The findings presented here represent a synthesis of 
{len(insights)} distinct insights derived from rigorous analysis.
"""
        
        return DocumentSection(
            title="Introduction",
            content=content.strip(),
            section_type="introduction",
            order=0,
            citations=[]
        )
    
    async def _generate_methodology_section(self, insights: List[Insight], 
                                          session_id: str) -> DocumentSection:
        """
        Generate methodology section.
        
        Args:
            insights: List of insights for context
            session_id: Session identifier
            
        Returns:
            DocumentSection: Methodology section
        """
        # Count unique sources
        all_sources = []
        for insight in insights:
            all_sources.extend(insight.related_sources)
        unique_sources = len(set(source.url for source in all_sources))
        
        content = f"""## Methodology

This research employed a systematic multi-agent approach to data collection and analysis. 
The methodology consisted of several integrated phases designed to ensure comprehensive 
coverage and rigorous analysis of the research topic.

### Data Collection

The research process utilized multiple specialized agents:

- **Research Agent**: Conducted web-based searches and content extraction from {unique_sources} unique sources
- **API Agent**: Queried academic databases including arXiv, PubMed, and Semantic Scholar
- **Analysis Agent**: Performed named entity recognition, topic modeling, and statistical analysis

### Analysis Framework

The analytical framework employed advanced natural language processing techniques:

1. **Named Entity Recognition**: Identification of key concepts, organizations, and individuals
2. **Topic Modeling**: Discovery of latent themes and patterns in the collected data
3. **Statistical Analysis**: Quantitative assessment of trends and relationships
4. **Insight Synthesis**: Integration of findings into coherent analytical insights

### Quality Assurance

All insights were evaluated based on confidence scores, with an average confidence level of 
{sum(insight.confidence_score for insight in insights) / len(insights):.2f}. 
Sources were validated for accessibility and relevance to ensure research integrity.

Session ID: `{session_id}`
"""
        
        return DocumentSection(
            title="Methodology",
            content=content.strip(),
            section_type="methodology",
            order=1,
            citations=[]
        )
    
    async def _generate_findings_section(self, topic_groups: Dict[str, List[Insight]], 
                                       insights: List[Insight]) -> DocumentSection:
        """
        Generate findings section from grouped insights.
        
        Args:
            topic_groups: Insights grouped by topic
            insights: All insights for reference
            
        Returns:
            DocumentSection: Findings section
        """
        content_parts = ["## Findings\n"]
        content_parts.append("The analysis revealed several significant findings across multiple dimensions of the research topic.\n")
        
        # Generate subsections for each topic group
        for i, (topic, topic_insights) in enumerate(sorted(topic_groups.items()), 1):
            content_parts.append(f"### {i}. {topic.title()}\n")
            
            # Synthesize insights for this topic
            high_confidence_insights = [
                insight for insight in topic_insights 
                if insight.confidence_score >= 0.7
            ]
            
            if high_confidence_insights:
                # Create narrative from high-confidence insights
                topic_summary = self._synthesize_topic_insights(high_confidence_insights)
                content_parts.append(f"{topic_summary}\n")
                
                # Add supporting evidence
                if high_confidence_insights[0].supporting_evidence:
                    content_parts.append("**Key Evidence:**\n")
                    for evidence in high_confidence_insights[0].supporting_evidence[:3]:
                        content_parts.append(f"- {evidence}\n")
                    content_parts.append("")
            else:
                content_parts.append(f"Analysis of {topic} revealed preliminary findings that require further investigation.\n")
        
        # Add summary of overall findings
        content_parts.append("### Summary of Key Findings\n")
        top_insights = sorted(insights, key=lambda x: x.confidence_score, reverse=True)[:3]
        
        for i, insight in enumerate(top_insights, 1):
            content_parts.append(f"{i}. **{insight.topic}**: {insight.summary}\n")
        
        return DocumentSection(
            title="Findings",
            content="\n".join(content_parts).strip(),
            section_type="findings",
            order=2,
            citations=[]
        )
    
    def _synthesize_topic_insights(self, insights: List[Insight]) -> str:
        """
        Synthesize multiple insights into coherent narrative.
        
        Args:
            insights: List of insights for the same topic
            
        Returns:
            str: Synthesized narrative
        """
        if not insights:
            return "No significant insights available for this topic."
        
        # Use the highest confidence insight as the primary narrative
        primary_insight = insights[0]
        narrative_parts = [primary_insight.summary]
        
        # Add supporting details from other insights
        if len(insights) > 1:
            supporting_points = []
            for insight in insights[1:3]:  # Limit to avoid redundancy
                if insight.summary != primary_insight.summary:
                    supporting_points.append(insight.summary)
            
            if supporting_points:
                narrative_parts.append(f" Additionally, {' '.join(supporting_points)}")
        
        return " ".join(narrative_parts)
    
    async def _generate_conclusion_section(self, insights: List[Insight], 
                                         original_query: str) -> DocumentSection:
        """
        Generate conclusion section.
        
        Args:
            insights: List of insights to conclude from
            original_query: Original research query
            
        Returns:
            DocumentSection: Conclusion section
        """
        # Identify top themes and insights
        top_insights = sorted(insights, key=lambda x: x.confidence_score, reverse=True)[:5]
        unique_topics = list(set(insight.topic for insight in top_insights))
        
        content = f"""## Conclusion

This comprehensive analysis of "{original_query}" has yielded significant insights across 
{len(unique_topics)} major thematic areas. The research demonstrates the complexity and 
multifaceted nature of the topic, revealing important patterns and relationships.

### Key Contributions

The analysis has made several important contributions to understanding:

"""
        
        # Add key contributions from top insights
        for i, insight in enumerate(top_insights[:3], 1):
            content += f"{i}. **{insight.topic}**: {insight.summary[:100]}{'...' if len(insight.summary) > 100 else ''}\n"
        
        content += f"""
### Research Implications

The findings suggest several important implications for future research and practice. 
The high confidence levels achieved (average: {sum(insight.confidence_score for insight in insights) / len(insights):.2f}) 
indicate robust analytical foundations for the conclusions presented.

### Future Directions

Based on the analysis, several areas warrant further investigation:

- Deeper exploration of the relationships between identified themes
- Longitudinal studies to track developments over time  
- Cross-domain analysis to identify broader patterns

This research provides a solid foundation for understanding the current state of knowledge 
while highlighting opportunities for continued investigation and development.
"""
        
        return DocumentSection(
            title="Conclusion",
            content=content.strip(),
            section_type="conclusion",
            order=3,
            citations=[]
        )
    
    async def _generate_additional_sections(self, topic_groups: Dict[str, List[Insight]]) -> List[DocumentSection]:
        """
        Generate additional sections if needed.
        
        Args:
            topic_groups: Insights grouped by topic
            
        Returns:
            List of additional DocumentSection objects
        """
        additional_sections = []
        
        # If there are many topic groups, create detailed sections for top topics
        if len(topic_groups) > 5:
            sorted_topics = sorted(
                topic_groups.items(), 
                key=lambda x: sum(insight.confidence_score for insight in x[1]), 
                reverse=True
            )
            
            for i, (topic, topic_insights) in enumerate(sorted_topics[:3], 4):
                if len(topic_insights) >= 2:  # Only create section if substantial content
                    section_content = f"## Detailed Analysis: {topic.title()}\n\n"
                    section_content += self._create_detailed_topic_analysis(topic_insights)
                    
                    additional_sections.append(DocumentSection(
                        title=f"Detailed Analysis: {topic.title()}",
                        content=section_content.strip(),
                        section_type="detailed_analysis",
                        order=i,
                        citations=[]
                    ))
        
        return additional_sections
    
    def _create_detailed_topic_analysis(self, insights: List[Insight]) -> str:
        """
        Create detailed analysis for a specific topic.
        
        Args:
            insights: List of insights for the topic
            
        Returns:
            str: Detailed analysis content
        """
        content_parts = []
        
        # Overview
        avg_confidence = sum(insight.confidence_score for insight in insights) / len(insights)
        content_parts.append(f"This section provides detailed analysis based on {len(insights)} insights "
                           f"with an average confidence score of {avg_confidence:.2f}.\n")
        
        # Key findings
        content_parts.append("### Key Findings\n")
        for i, insight in enumerate(insights[:3], 1):
            content_parts.append(f"**Finding {i}**: {insight.summary}\n")
            if insight.supporting_evidence:
                content_parts.append("*Supporting Evidence*:")
                for evidence in insight.supporting_evidence[:2]:
                    content_parts.append(f"- {evidence}")
                content_parts.append("")
        
        # Synthesis
        content_parts.append("### Analysis Synthesis\n")
        content_parts.append(self._synthesize_topic_insights(insights))
        
        return "\n".join(content_parts)
    
    def _generate_table_of_contents(self, document_sections: DocumentSections) -> str:
        """
        Generate table of contents with proper section linking.
        
        Args:
            document_sections: Complete document structure
            
        Returns:
            str: Formatted table of contents
        """
        toc_parts = ["# Table of Contents\n"]
        
        # Get all sections in order
        all_sections = document_sections.get_all_sections()
        
        for section in all_sections:
            # Create anchor link from title
            anchor = self._create_anchor_link(section.title)
            toc_parts.append(f"{section.order + 1}. [{section.title}](#{anchor})")
        
        return "\n".join(toc_parts)
    
    def _create_anchor_link(self, title: str) -> str:
        """
        Create markdown anchor link from section title.
        
        Args:
            title: Section title
            
        Returns:
            str: Anchor link
        """
        # Convert to lowercase, replace spaces with hyphens, remove special chars
        anchor = re.sub(r'[^\w\s-]', '', title.lower())
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')
    
    async def _format_final_document(self, document_sections: DocumentSections) -> str:
        """
        Format the complete document with proper Markdown structure.
        
        Args:
            document_sections: Complete document structure
            
        Returns:
            str: Final formatted document
        """
        document_parts = []
        
        # Add table of contents
        if document_sections.table_of_contents:
            document_parts.append(document_sections.table_of_contents)
            document_parts.append("\n---\n")
        
        # Add all sections in order
        all_sections = document_sections.get_all_sections()
        
        for section in all_sections:
            document_parts.append(section.content)
            document_parts.append("\n")
        
        # Add metadata footer
        document_parts.append("---")
        document_parts.append(f"*Document generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        document_parts.append(f"*Total sections: {len(all_sections)}*")
        
        return "\n".join(document_parts)
    
    async def generate_section_content(self, section_type: str, insights: List[Insight], 
                                     context: Dict[str, Any]) -> str:
        """
        Generate content for a specific section type.
        
        Args:
            section_type: Type of section to generate
            insights: Relevant insights
            context: Additional context information
            
        Returns:
            str: Generated section content
        """
        if section_type == "introduction":
            return (await self._generate_introduction_section(
                context.get("original_query", ""), insights
            )).content
        elif section_type == "methodology":
            return (await self._generate_methodology_section(
                insights, context.get("session_id", "")
            )).content
        elif section_type == "findings":
            topic_groups = self._group_insights_by_topic(insights)
            return (await self._generate_findings_section(topic_groups, insights)).content
        elif section_type == "conclusion":
            return (await self._generate_conclusion_section(
                insights, context.get("original_query", "")
            )).content
        else:
            raise ValueError(f"Unknown section type: {section_type}")
    
    def validate_document_structure(self, document_sections: DocumentSections) -> bool:
        """
        Validate that the document structure is complete and well-formed.
        
        Args:
            document_sections: Document structure to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValidationError: If structure is invalid
        """
        required_sections = ["introduction", "methodology", "findings", "conclusion"]
        
        # Check that all required sections exist
        section_types = {
            document_sections.introduction.section_type,
            document_sections.methodology.section_type,
            document_sections.findings.section_type,
            document_sections.conclusion.section_type
        }
        
        missing_sections = set(required_sections) - section_types
        if missing_sections:
            raise ValidationError(f"Missing required sections: {missing_sections}")
        
        # Check that all sections have content
        all_sections = document_sections.get_all_sections()
        empty_sections = [s.title for s in all_sections if not s.content.strip()]
        if empty_sections:
            raise ValidationError(f"Empty sections found: {empty_sections}")
        
        # Check table of contents exists
        if not document_sections.table_of_contents.strip():
            raise ValidationError("Table of contents is missing")
        
        return True