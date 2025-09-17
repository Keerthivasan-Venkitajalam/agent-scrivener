"""
Sample Research Queries for Agent Scrivener Demonstration

This module contains test queries that demonstrate the system's capabilities
across different research domains and complexity levels.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SampleQuery:
    """Represents a sample research query with expected outcomes"""
    title: str
    query: str
    domain: str
    complexity: str  # "simple", "medium", "complex"
    expected_sources: int
    expected_sections: List[str]
    description: str

# Technology and AI Research Queries
TECHNOLOGY_QUERIES = [
    SampleQuery(
        title="Large Language Models in Healthcare",
        query="Analyze the current applications and future potential of large language models in healthcare, including diagnostic assistance, medical record processing, and patient interaction. Examine both benefits and ethical concerns, with focus on recent developments in 2024.",
        domain="Technology/Healthcare",
        complexity="complex",
        expected_sources=15,
        expected_sections=["Introduction", "Current Applications", "Diagnostic Assistance", "Ethical Considerations", "Future Potential", "Conclusion"],
        description="Comprehensive analysis of LLMs in healthcare with ethical considerations"
    ),
    
    SampleQuery(
        title="Quantum Computing Progress 2024",
        query="What are the latest breakthroughs in quantum computing hardware and algorithms in 2024? Focus on IBM, Google, and other major players' achievements in quantum supremacy and practical applications.",
        domain="Technology",
        complexity="medium",
        expected_sources=12,
        expected_sections=["Introduction", "Hardware Breakthroughs", "Algorithm Advances", "Industry Leaders", "Practical Applications", "Conclusion"],
        description="Recent quantum computing developments and industry progress"
    ),
    
    SampleQuery(
        title="Edge AI Implementation",
        query="How is edge AI being implemented in IoT devices and what are the performance trade-offs compared to cloud-based AI processing?",
        domain="Technology",
        complexity="simple",
        expected_sources=8,
        expected_sections=["Introduction", "Edge AI Overview", "IoT Implementation", "Performance Analysis", "Conclusion"],
        description="Simple analysis of edge AI in IoT applications"
    )
]

# Climate and Environmental Research Queries
CLIMATE_QUERIES = [
    SampleQuery(
        title="Carbon Capture Technologies Assessment",
        query="Evaluate the effectiveness and scalability of current carbon capture, utilization, and storage (CCUS) technologies. Compare direct air capture, point-source capture, and ocean-based solutions with analysis of costs, energy requirements, and deployment challenges.",
        domain="Climate/Environment",
        complexity="complex",
        expected_sources=18,
        expected_sections=["Introduction", "Technology Overview", "Direct Air Capture", "Point-Source Capture", "Ocean-Based Solutions", "Cost Analysis", "Deployment Challenges", "Conclusion"],
        description="Comprehensive evaluation of carbon capture technologies"
    ),
    
    SampleQuery(
        title="Renewable Energy Grid Integration",
        query="What are the current challenges and solutions for integrating renewable energy sources into existing power grids? Focus on storage solutions, grid stability, and smart grid technologies.",
        domain="Energy",
        complexity="medium",
        expected_sources=14,
        expected_sections=["Introduction", "Integration Challenges", "Storage Solutions", "Grid Stability", "Smart Grid Technologies", "Conclusion"],
        description="Analysis of renewable energy grid integration challenges"
    )
]

# Business and Economics Research Queries
BUSINESS_QUERIES = [
    SampleQuery(
        title="Remote Work Impact on Productivity",
        query="Analyze the long-term impact of remote work on employee productivity, company culture, and business outcomes based on post-pandemic data and studies.",
        domain="Business",
        complexity="medium",
        expected_sources=16,
        expected_sections=["Introduction", "Productivity Metrics", "Cultural Impact", "Business Outcomes", "Long-term Trends", "Conclusion"],
        description="Post-pandemic remote work impact analysis"
    ),
    
    SampleQuery(
        title="Cryptocurrency Market Trends",
        query="What are the current trends in cryptocurrency adoption by institutional investors and the regulatory landscape changes in 2024?",
        domain="Finance",
        complexity="simple",
        expected_sources=10,
        expected_sections=["Introduction", "Institutional Adoption", "Regulatory Changes", "Market Trends", "Conclusion"],
        description="Current cryptocurrency market and regulatory analysis"
    )
]

# Medical and Health Research Queries
MEDICAL_QUERIES = [
    SampleQuery(
        title="Personalized Medicine Advances",
        query="Examine recent advances in personalized medicine, including genomic testing, targeted therapies, and AI-driven treatment recommendations. Analyze the current state of implementation in clinical practice and barriers to widespread adoption.",
        domain="Medicine",
        complexity="complex",
        expected_sources=20,
        expected_sections=["Introduction", "Genomic Testing", "Targeted Therapies", "AI-Driven Recommendations", "Clinical Implementation", "Adoption Barriers", "Future Outlook", "Conclusion"],
        description="Comprehensive review of personalized medicine developments"
    )
]

# All sample queries organized by complexity
ALL_QUERIES = {
    "simple": [q for q in TECHNOLOGY_QUERIES + CLIMATE_QUERIES + BUSINESS_QUERIES + MEDICAL_QUERIES if q.complexity == "simple"],
    "medium": [q for q in TECHNOLOGY_QUERIES + CLIMATE_QUERIES + BUSINESS_QUERIES + MEDICAL_QUERIES if q.complexity == "medium"],
    "complex": [q for q in TECHNOLOGY_QUERIES + CLIMATE_QUERIES + BUSINESS_QUERIES + MEDICAL_QUERIES if q.complexity == "complex"]
}

def get_queries_by_domain(domain: str) -> List[SampleQuery]:
    """Get all queries for a specific domain"""
    all_queries = TECHNOLOGY_QUERIES + CLIMATE_QUERIES + BUSINESS_QUERIES + MEDICAL_QUERIES
    return [q for q in all_queries if domain.lower() in q.domain.lower()]

def get_queries_by_complexity(complexity: str) -> List[SampleQuery]:
    """Get all queries of a specific complexity level"""
    return ALL_QUERIES.get(complexity, [])

def get_demo_query_sequence() -> List[SampleQuery]:
    """Get a sequence of queries for demonstration purposes"""
    return [
        # Start with simple query to show basic functionality
        next(q for q in ALL_QUERIES["simple"] if "Edge AI" in q.title),
        # Medium complexity to show analysis capabilities
        next(q for q in ALL_QUERIES["medium"] if "Remote Work" in q.title),
        # Complex query to show full system capabilities
        next(q for q in ALL_QUERIES["complex"] if "Large Language Models" in q.title)
    ]