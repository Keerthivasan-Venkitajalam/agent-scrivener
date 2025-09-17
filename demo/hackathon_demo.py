#!/usr/bin/env python3
"""
Agent Scrivener Hackathon Demonstration Script

This script provides an interactive demonstration of the Agent Scrivener system
for hackathon presentations and live demos.
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.sample_queries import get_demo_query_sequence, SampleQuery
from demo.sample_outputs import get_sample_output, get_sample_metadata
from agent_scrivener.api.models import ResearchRequest, ResearchResponse
from agent_scrivener.models.core import ResearchSession

class HackathonDemo:
    """Interactive demonstration class for hackathon presentations"""
    
    def __init__(self):
        self.demo_queries = get_demo_query_sequence()
        self.current_step = 0
        
    def print_banner(self):
        """Print the demo banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    AGENT SCRIVENER DEMO                      ║
║              Autonomous Research & Content Synthesis         ║
║                                                              ║
║  Transform any research query into a comprehensive,          ║
║  fully-cited research document using AI agent orchestration ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def print_system_overview(self):
        """Print system architecture overview"""
        print("\n🏗️  SYSTEM ARCHITECTURE")
        print("=" * 50)
        print("┌─ Planner Agent ──────────────────────────────────┐")
        print("│  • Query analysis & task decomposition          │")
        print("│  • DAG workflow creation                         │")
        print("└──────────────────────────────────────────────────┘")
        print("           │")
        print("           ▼")
        print("┌─ Research Agents ────────────────────────────────┐")
        print("│  Research Agent    │  API Agent                  │")
        print("│  • Web search      │  • arXiv, PubMed           │")
        print("│  • Content extract │  • Semantic Scholar        │")
        print("└────────────────────┴─────────────────────────────┘")
        print("           │")
        print("           ▼")
        print("┌─ Analysis Agent ─────────────────────────────────┐")
        print("│  • Named Entity Recognition                      │")
        print("│  • Topic modeling & statistical analysis        │")
        print("│  • Insight generation                           │")
        print("└──────────────────────────────────────────────────┘")
        print("           │")
        print("           ▼")
        print("┌─ Content Synthesis ──────────────────────────────┐")
        print("│  Drafting Agent    │  Citation Agent             │")
        print("│  • Section writing │  • Source tracking          │")
        print("│  • Markdown format │  • APA citations            │")
        print("└────────────────────┴─────────────────────────────┘")
        
    async def demonstrate_query_processing(self, query: SampleQuery):
        """Demonstrate processing of a single query"""
        print(f"\n🔍 PROCESSING QUERY: {query.title}")
        print("=" * 60)
        print(f"Domain: {query.domain}")
        print(f"Complexity: {query.complexity}")
        print(f"Query: {query.query[:100]}...")
        
        # Simulate real-time processing steps
        steps = [
            ("🧠 Planner Agent: Analyzing query...", 1.5),
            ("📊 Creating task execution graph...", 1.0),
            ("🌐 Research Agent: Searching web sources...", 2.0),
            ("📚 API Agent: Querying academic databases...", 2.5),
            ("🔬 Analysis Agent: Processing content...", 3.0),
            ("📝 Drafting Agent: Synthesizing sections...", 2.5),
            ("📖 Citation Agent: Formatting references...", 1.5),
            ("✅ Research complete!", 0.5)
        ]
        
        start_time = time.time()
        
        for step_desc, duration in steps:
            print(f"\n{step_desc}")
            await self.simulate_progress_bar(duration)
            
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total processing time: {elapsed_time:.1f} seconds")
        
    async def simulate_progress_bar(self, duration: float):
        """Simulate a progress bar for demo purposes"""
        steps = 20
        step_duration = duration / steps
        
        print("[", end="", flush=True)
        for i in range(steps):
            await asyncio.sleep(step_duration)
            print("█", end="", flush=True)
        print("] 100%")
        
    def show_sample_output(self, query_type: str):
        """Display sample output for a query"""
        print(f"\n📄 SAMPLE OUTPUT")
        print("=" * 50)
        
        # Show metadata first
        metadata = get_sample_metadata(query_type)
        if metadata:
            print("📊 Research Metrics:")
            for key, value in metadata.items():
                print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        print("\n📝 Generated Document Preview:")
        print("-" * 40)
        
        # Show first few lines of the output
        output = get_sample_output(query_type)
        lines = output.split('\n')[:25]  # Show first 25 lines
        for line in lines:
            print(line)
        
        print("\n... [Document continues with full analysis and citations] ...")
        
    def demonstrate_key_features(self):
        """Highlight key system features"""
        print("\n🌟 KEY FEATURES DEMONSTRATED")
        print("=" * 50)
        
        features = [
            "🤖 Multi-Agent Orchestration",
            "   • Specialized agents for different research tasks",
            "   • Intelligent task decomposition and coordination",
            "",
            "🔍 Comprehensive Source Coverage", 
            "   • Web search with content extraction",
            "   • Academic database integration (arXiv, PubMed)",
            "   • Automated source validation and quality scoring",
            "",
            "🧠 Advanced Analysis Capabilities",
            "   • Named Entity Recognition (NER)",
            "   • Topic modeling and statistical analysis", 
            "   • Insight generation with confidence scoring",
            "",
            "📝 Professional Document Generation",
            "   • Structured sections with logical flow",
            "   • Proper Markdown formatting",
            "   • Comprehensive citation management",
            "",
            "⚡ Real-time Progress Tracking",
            "   • WebSocket-based status updates",
            "   • Detailed processing metrics",
            "   • Error handling and recovery",
            "",
            "🏗️ Spec-Driven Development",
            "   • Built using Kiro's systematic approach",
            "   • Comprehensive testing and validation",
            "   • AWS AgentCore Runtime deployment"
        ]
        
        for feature in features:
            print(feature)
            
    def show_technical_highlights(self):
        """Show technical implementation highlights"""
        print("\n⚙️  TECHNICAL HIGHLIGHTS")
        print("=" * 50)
        
        highlights = [
            "🏛️  Architecture:",
            "   • AWS Bedrock AgentCore Runtime",
            "   • Strands SDK for workflow orchestration", 
            "   • FastAPI with WebSocket support",
            "   • Pydantic models for data validation",
            "",
            "🔧 Agent Tools:",
            "   • Nova Act SDK for precise web navigation",
            "   • AgentCore Gateway for API management",
            "   • Code Interpreter for data analysis",
            "   • Memory system for session persistence",
            "",
            "📊 Quality Assurance:",
            "   • Comprehensive unit and integration tests",
            "   • Performance benchmarking and load testing",
            "   • Error handling with circuit breaker patterns",
            "   • Real-time monitoring and alerting",
            "",
            "🚀 Deployment:",
            "   • Docker containerization",
            "   • Infrastructure as Code (CDK)",
            "   • Auto-scaling and load balancing",
            "   • Security best practices"
        ]
        
        for highlight in highlights:
            print(highlight)
            
    async def run_interactive_demo(self):
        """Run the complete interactive demonstration"""
        self.print_banner()
        
        print("\nWelcome to the Agent Scrivener demonstration!")
        print("This system transforms research queries into comprehensive documents.")
        
        # Show system overview
        input("\nPress Enter to see the system architecture...")
        self.print_system_overview()
        
        # Demonstrate each query type
        for i, query in enumerate(self.demo_queries):
            input(f"\nPress Enter to demonstrate {query.complexity} complexity query...")
            await self.demonstrate_query_processing(query)
            
            # Show sample output for first two queries
            if i < 2:
                input("\nPress Enter to see the generated output...")
                query_type = "edge_ai" if "Edge AI" in query.title else "remote_work"
                self.show_sample_output(query_type)
        
        # Show key features
        input("\nPress Enter to see key features...")
        self.demonstrate_key_features()
        
        # Show technical highlights
        input("\nPress Enter to see technical implementation...")
        self.show_technical_highlights()
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 50)
        print("Thank you for exploring Agent Scrivener!")
        print("Questions? Let's discuss the implementation details.")
        
    async def run_automated_demo(self):
        """Run automated demo for presentations"""
        self.print_banner()
        await asyncio.sleep(2)
        
        self.print_system_overview()
        await asyncio.sleep(3)
        
        # Quick demo of one query
        query = self.demo_queries[0]  # Simple query
        await self.demonstrate_query_processing(query)
        await asyncio.sleep(2)
        
        self.show_sample_output("edge_ai")
        await asyncio.sleep(3)
        
        self.demonstrate_key_features()
        await asyncio.sleep(2)
        
        print("\n🎉 Demo complete! Ready for questions.")

def main():
    """Main demo function"""
    demo = HackathonDemo()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Automated demo for presentations
        asyncio.run(demo.run_automated_demo())
    else:
        # Interactive demo
        asyncio.run(demo.run_interactive_demo())

if __name__ == "__main__":
    main()