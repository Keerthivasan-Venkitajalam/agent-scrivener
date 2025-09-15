"""
Test datasets for consistent integration testing.

Provides standardized test data for research queries, mock responses,
and expected outputs to ensure consistent test results.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from agent_scrivener.models.core import (
    Source, SourceType, ExtractedArticle, AcademicPaper, Insight
)


@dataclass
class TestDataset:
    """Container for a complete test dataset."""
    name: str
    description: str
    research_query: str
    expected_sources: List[Dict[str, Any]]
    expected_articles: List[Dict[str, Any]]
    expected_papers: List[Dict[str, Any]]
    expected_insights: List[Dict[str, Any]]
    expected_document_sections: Dict[str, str]
    performance_benchmarks: Dict[str, float]


class TestDatasets:
    """Collection of standardized test datasets."""
    
    @staticmethod
    def get_machine_learning_dataset() -> TestDataset:
        """Dataset for machine learning research queries."""
        return TestDataset(
            name="machine_learning_healthcare",
            description="Machine learning applications in healthcare research",
            research_query="Machine learning applications in healthcare diagnosis and treatment",
            expected_sources=[
                {
                    "url": "https://example-medical.com/ml-diagnosis",
                    "title": "Machine Learning in Medical Diagnosis: A Comprehensive Review",
                    "author": "Dr. Sarah Johnson",
                    "source_type": "web",
                    "domain": "example-medical.com",
                    "word_count": 2500,
                    "confidence_score": 0.92
                },
                {
                    "url": "https://healthcare-tech.org/ai-treatment",
                    "title": "AI-Powered Treatment Recommendations in Modern Healthcare",
                    "author": "Prof. Michael Chen",
                    "source_type": "web",
                    "domain": "healthcare-tech.org",
                    "word_count": 1800,
                    "confidence_score": 0.88
                },
                {
                    "url": "https://medical-ai.net/deep-learning-radiology",
                    "title": "Deep Learning Applications in Radiology and Medical Imaging",
                    "author": "Dr. Emily Rodriguez",
                    "source_type": "web",
                    "domain": "medical-ai.net",
                    "word_count": 3200,
                    "confidence_score": 0.95
                }
            ],
            expected_articles=[
                {
                    "content": "Machine learning has revolutionized healthcare by enabling more accurate diagnosis and personalized treatment plans. Recent advances in deep learning have shown remarkable success in medical image analysis, with convolutional neural networks achieving diagnostic accuracy comparable to expert radiologists. Natural language processing techniques are being applied to electronic health records to extract meaningful insights and predict patient outcomes. The integration of ML algorithms in clinical decision support systems has improved treatment recommendations and reduced medical errors.",
                    "key_findings": [
                        "ML algorithms achieve diagnostic accuracy comparable to expert radiologists",
                        "NLP techniques extract insights from electronic health records",
                        "Clinical decision support systems reduce medical errors",
                        "Personalized treatment plans improve patient outcomes"
                    ],
                    "confidence_score": 0.92
                }
            ],
            expected_papers=[
                {
                    "title": "Deep Learning for Medical Image Analysis: A Survey",
                    "authors": ["Zhang, L.", "Wang, H.", "Liu, M."],
                    "abstract": "This survey provides a comprehensive overview of deep learning techniques applied to medical image analysis. We review recent advances in convolutional neural networks, attention mechanisms, and transfer learning approaches for various medical imaging modalities including X-ray, CT, MRI, and ultrasound.",
                    "publication_year": 2023,
                    "doi": "10.1016/j.media.2023.102845",
                    "database_source": "pubmed",
                    "citation_count": 156
                },
                {
                    "title": "Machine Learning in Clinical Decision Support: Current Applications and Future Directions",
                    "authors": ["Smith, J.A.", "Brown, K.L.", "Davis, R.M."],
                    "abstract": "Clinical decision support systems powered by machine learning are transforming healthcare delivery. This paper reviews current applications in diagnosis, treatment recommendation, and risk prediction, while discussing challenges and future research directions.",
                    "publication_year": 2023,
                    "doi": "10.1038/s41591-023-02456-7",
                    "database_source": "arxiv",
                    "citation_count": 89
                }
            ],
            expected_insights=[
                {
                    "topic": "Diagnostic Accuracy",
                    "summary": "Machine learning models, particularly deep learning approaches, have demonstrated diagnostic accuracy comparable to or exceeding human experts in various medical domains.",
                    "confidence_score": 0.91,
                    "supporting_evidence": [
                        "CNN models achieve 94% accuracy in skin cancer detection",
                        "Radiologist-level performance in chest X-ray analysis",
                        "Superior performance in diabetic retinopathy screening"
                    ]
                },
                {
                    "topic": "Clinical Integration",
                    "summary": "Integration of ML systems into clinical workflows requires careful consideration of usability, interpretability, and regulatory compliance.",
                    "confidence_score": 0.85,
                    "supporting_evidence": [
                        "FDA approval required for diagnostic ML systems",
                        "Clinician acceptance depends on system transparency",
                        "Integration challenges with existing EHR systems"
                    ]
                }
            ],
            expected_document_sections={
                "introduction": "Machine learning (ML) has emerged as a transformative technology in healthcare, offering unprecedented opportunities to improve patient care through enhanced diagnostic accuracy, personalized treatment recommendations, and predictive analytics.",
                "methodology": "This research synthesizes findings from peer-reviewed literature, clinical studies, and industry reports to provide a comprehensive overview of ML applications in healthcare.",
                "findings": "Key findings indicate that ML algorithms achieve diagnostic accuracy comparable to expert clinicians in several domains, with deep learning models showing particular promise in medical imaging applications.",
                "conclusion": "Machine learning represents a paradigm shift in healthcare delivery, with the potential to significantly improve patient outcomes while reducing costs and medical errors."
            },
            performance_benchmarks={
                "expected_execution_time_ms": 2500,
                "expected_memory_usage_mb": 150,
                "expected_success_rate": 100.0,
                "expected_source_count": 8,
                "expected_insight_count": 5
            }
        )
    
    @staticmethod
    def get_climate_change_dataset() -> TestDataset:
        """Dataset for climate change research queries."""
        return TestDataset(
            name="climate_change_agriculture",
            description="Climate change impact on agricultural systems",
            research_query="Climate change impacts on global agricultural productivity and food security",
            expected_sources=[
                {
                    "url": "https://climate-research.org/agriculture-impacts",
                    "title": "Climate Change and Agricultural Productivity: Global Trends and Regional Variations",
                    "author": "Dr. Maria Gonzalez",
                    "source_type": "web",
                    "domain": "climate-research.org",
                    "word_count": 2800,
                    "confidence_score": 0.89
                },
                {
                    "url": "https://food-security.net/climate-adaptation",
                    "title": "Adaptation Strategies for Climate-Resilient Agriculture",
                    "author": "Prof. David Thompson",
                    "source_type": "web",
                    "domain": "food-security.net",
                    "word_count": 2200,
                    "confidence_score": 0.86
                }
            ],
            expected_articles=[
                {
                    "content": "Climate change poses significant challenges to global agricultural systems through altered precipitation patterns, increased temperature variability, and more frequent extreme weather events. Rising temperatures affect crop yields differently across regions, with some areas experiencing decreased productivity while others may see temporary improvements. Changes in rainfall patterns disrupt traditional farming practices and water management systems.",
                    "key_findings": [
                        "Temperature increases reduce yields of major crops",
                        "Precipitation changes disrupt farming practices",
                        "Extreme weather events increase crop losses",
                        "Regional variations in climate impacts"
                    ],
                    "confidence_score": 0.89
                }
            ],
            expected_papers=[
                {
                    "title": "Global Assessment of Climate Change Impacts on Crop Yields",
                    "authors": ["Anderson, P.K.", "Martinez, C.L.", "Wilson, R.J."],
                    "abstract": "This study presents a comprehensive global assessment of climate change impacts on major crop yields using multi-model ensemble projections and observational data from 1980-2020.",
                    "publication_year": 2023,
                    "doi": "10.1038/s41558-023-01678-2",
                    "database_source": "nature",
                    "citation_count": 234
                }
            ],
            expected_insights=[
                {
                    "topic": "Yield Impacts",
                    "summary": "Climate change is projected to reduce yields of major staple crops globally, with significant regional variations in impact severity.",
                    "confidence_score": 0.88,
                    "supporting_evidence": [
                        "Wheat yields declining by 6% per degree of warming",
                        "Rice productivity affected by night temperature increases",
                        "Maize yields show high sensitivity to heat stress"
                    ]
                }
            ],
            expected_document_sections={
                "introduction": "Climate change represents one of the most pressing challenges facing global agriculture in the 21st century.",
                "methodology": "This analysis synthesizes data from climate models, agricultural statistics, and field studies.",
                "findings": "Evidence indicates widespread negative impacts on crop productivity with significant regional variations.",
                "conclusion": "Urgent adaptation measures are needed to maintain food security under changing climate conditions."
            },
            performance_benchmarks={
                "expected_execution_time_ms": 2800,
                "expected_memory_usage_mb": 160,
                "expected_success_rate": 100.0,
                "expected_source_count": 7,
                "expected_insight_count": 4
            }
        )
    
    @staticmethod
    def get_quantum_computing_dataset() -> TestDataset:
        """Dataset for quantum computing research queries."""
        return TestDataset(
            name="quantum_computing_algorithms",
            description="Quantum computing algorithms and applications",
            research_query="Recent advances in quantum computing algorithms and their practical applications",
            expected_sources=[
                {
                    "url": "https://quantum-research.edu/algorithms",
                    "title": "Breakthrough Quantum Algorithms for Optimization Problems",
                    "author": "Dr. Alice Quantum",
                    "source_type": "web",
                    "domain": "quantum-research.edu",
                    "word_count": 3500,
                    "confidence_score": 0.94
                }
            ],
            expected_articles=[
                {
                    "content": "Quantum computing algorithms leverage quantum mechanical phenomena such as superposition and entanglement to solve certain computational problems exponentially faster than classical computers. Recent developments in variational quantum algorithms show promise for near-term quantum devices.",
                    "key_findings": [
                        "Variational algorithms suitable for NISQ devices",
                        "Quantum advantage demonstrated in specific problems",
                        "Error correction remains a major challenge"
                    ],
                    "confidence_score": 0.94
                }
            ],
            expected_papers=[
                {
                    "title": "Variational Quantum Algorithms for Combinatorial Optimization",
                    "authors": ["Quantum, A.", "Entangle, B.", "Superpose, C."],
                    "abstract": "We present novel variational quantum algorithms for solving combinatorial optimization problems on near-term quantum devices.",
                    "publication_year": 2023,
                    "doi": "10.1103/PhysRevA.107.032415",
                    "database_source": "arxiv",
                    "citation_count": 67
                }
            ],
            expected_insights=[
                {
                    "topic": "Algorithm Development",
                    "summary": "New quantum algorithms are being developed specifically for near-term quantum devices with limited coherence times.",
                    "confidence_score": 0.87,
                    "supporting_evidence": [
                        "QAOA shows promise for optimization problems",
                        "VQE algorithms for quantum chemistry",
                        "Hybrid classical-quantum approaches"
                    ]
                }
            ],
            expected_document_sections={
                "introduction": "Quantum computing represents a paradigm shift in computational capability.",
                "methodology": "Analysis of recent literature and experimental results in quantum algorithm development.",
                "findings": "Significant progress in variational algorithms and error mitigation techniques.",
                "conclusion": "Quantum computing is approaching practical utility for specific problem domains."
            },
            performance_benchmarks={
                "expected_execution_time_ms": 2200,
                "expected_memory_usage_mb": 140,
                "expected_success_rate": 100.0,
                "expected_source_count": 6,
                "expected_insight_count": 4
            }
        )
    
    @staticmethod
    def get_all_datasets() -> List[TestDataset]:
        """Get all available test datasets."""
        return [
            TestDatasets.get_machine_learning_dataset(),
            TestDatasets.get_climate_change_dataset(),
            TestDatasets.get_quantum_computing_dataset()
        ]
    
    @staticmethod
    def get_dataset_by_name(name: str) -> TestDataset:
        """Get a specific dataset by name."""
        datasets = {
            "machine_learning_healthcare": TestDatasets.get_machine_learning_dataset(),
            "climate_change_agriculture": TestDatasets.get_climate_change_dataset(),
            "quantum_computing_algorithms": TestDatasets.get_quantum_computing_dataset()
        }
        
        if name not in datasets:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(datasets.keys())}")
        
        return datasets[name]
    
    @staticmethod
    def save_dataset_to_file(dataset: TestDataset, filepath: str):
        """Save a dataset to a JSON file."""
        dataset_dict = asdict(dataset)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False, default=str)
    
    @staticmethod
    def load_dataset_from_file(filepath: str) -> TestDataset:
        """Load a dataset from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset_dict = json.load(f)
        
        return TestDataset(**dataset_dict)


class MockResponseGenerator:
    """Generate mock responses based on test datasets."""
    
    @staticmethod
    def generate_web_search_response(dataset: TestDataset, query: str) -> List[Dict[str, Any]]:
        """Generate mock web search response."""
        return [
            {
                "url": source["url"],
                "title": source["title"],
                "snippet": f"Relevant information about {query} from {source['domain']}",
                "domain": source["domain"]
            }
            for source in dataset.expected_sources
        ]
    
    @staticmethod
    def generate_content_extraction_response(dataset: TestDataset, url: str) -> Dict[str, Any]:
        """Generate mock content extraction response."""
        # Find matching source
        matching_source = next(
            (source for source in dataset.expected_sources if source["url"] == url),
            None
        )
        
        if not matching_source:
            return {
                "url": url,
                "title": "Generic Article Title",
                "content": "Generic article content for testing purposes.",
                "author": "Test Author",
                "word_count": 500,
                "confidence_score": 0.8
            }
        
        # Find matching article content
        article_content = "Default article content for testing."
        if dataset.expected_articles:
            article_content = dataset.expected_articles[0]["content"]
        
        return {
            "url": url,
            "title": matching_source["title"],
            "content": article_content,
            "author": matching_source["author"],
            "word_count": matching_source["word_count"],
            "confidence_score": matching_source["confidence_score"]
        }
    
    @staticmethod
    def generate_academic_search_response(dataset: TestDataset, query: str, database: str) -> List[Dict[str, Any]]:
        """Generate mock academic search response."""
        return [
            paper for paper in dataset.expected_papers
            if paper.get("database_source", "").lower() == database.lower()
        ]
    
    @staticmethod
    def generate_analysis_response(dataset: TestDataset, content: str) -> Dict[str, Any]:
        """Generate mock analysis response."""
        return {
            "entities": [
                {"text": "Machine Learning", "label": "TECHNOLOGY", "confidence": 0.9},
                {"text": "Healthcare", "label": "DOMAIN", "confidence": 0.85},
                {"text": "Diagnosis", "label": "APPLICATION", "confidence": 0.8}
            ],
            "topics": [
                {"topic_id": 0, "words": ["machine", "learning", "algorithm"], "weight": 0.4},
                {"topic_id": 1, "words": ["healthcare", "medical", "diagnosis"], "weight": 0.35},
                {"topic_id": 2, "words": ["treatment", "patient", "clinical"], "weight": 0.25}
            ],
            "insights": dataset.expected_insights,
            "sentiment": {"polarity": 0.1, "subjectivity": 0.3},
            "key_phrases": ["machine learning", "healthcare applications", "diagnostic accuracy"],
            "summary": f"Analysis of content related to {dataset.research_query}"
        }


# Export test data as JSON files for external use
def export_test_datasets():
    """Export all test datasets to JSON files."""
    import os
    
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(__file__)
    os.makedirs(data_dir, exist_ok=True)
    
    # Export each dataset
    for dataset in TestDatasets.get_all_datasets():
        filepath = os.path.join(data_dir, f"{dataset.name}.json")
        TestDatasets.save_dataset_to_file(dataset, filepath)
        print(f"Exported dataset: {filepath}")


if __name__ == "__main__":
    export_test_datasets()