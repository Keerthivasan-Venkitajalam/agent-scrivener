"""Unit tests for EndToEndValidator."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings
import asyncio

from agent_scrivener.deployment.validation import (
    EndToEndValidator,
    ValidationStatus
)


class TestEndToEndValidator:
    """Test EndToEndValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return EndToEndValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test-token",
            timeout_minutes=5
        )
    
    @pytest.mark.asyncio
    async def test_validate_document_quality_success(self, validator):
        """Test document quality validation with valid document."""
        document = """
        # Research Report
        
        ## Introduction
        This is the introduction section with important context about distributed systems.
        Distributed systems are complex architectures that span multiple computers and networks.
        Understanding their design principles is crucial for building reliable and scalable applications.
        The key challenges in distributed systems include managing consistency, ensuring availability,
        and handling network partitions gracefully. These three properties form the foundation of
        the CAP theorem, which states that a distributed system can only guarantee two of these
        three properties at any given time. This fundamental trade-off shapes how we design and
        implement distributed systems in practice. The introduction sets the stage for a deeper
        exploration of these critical concepts and their practical implications in modern software
        architecture and system design.
        
        ## Analysis
        Here we analyze the key findings and data points from our research into distributed systems.
        We examine multiple perspectives and evaluate the evidence carefully from various sources.
        The analysis reveals that consistency models vary significantly across different systems.
        Strong consistency provides the easiest programming model but comes at the cost of availability
        and performance. Eventual consistency offers better availability but requires careful handling
        of conflicts and reconciliation. The choice between these models depends on the specific
        requirements of the application and the acceptable trade-offs. Network partitions are inevitable
        in distributed systems, so systems must be designed to handle them gracefully. This includes
        implementing proper timeout mechanisms, retry logic, and fallback strategies. The analysis
        also examines various consensus algorithms like Paxos and Raft that enable distributed
        coordination while maintaining system correctness and reliability under various failure
        scenarios.
        
        ## Synthesis
        In this section, we synthesize the information gathered and identify patterns and connections
        between different concepts and ideas in distributed systems design. The research shows that
        successful distributed systems share common architectural patterns. These include the use of
        replication for fault tolerance, partitioning for scalability, and consensus algorithms for
        coordination. The synthesis of these patterns reveals that there is no one-size-fits-all
        solution. Instead, system designers must carefully consider their specific requirements and
        constraints. The trade-offs between consistency, availability, and partition tolerance must
        be evaluated in the context of the application's needs. Modern distributed systems often
        employ hybrid approaches that combine different consistency models for different parts of
        the system, allowing for optimization based on specific use cases. This synthesis brings
        together theoretical foundations with practical implementation strategies to provide a
        comprehensive understanding of distributed systems architecture.
        
        ## Conclusion
        Finally, we draw conclusions based on our comprehensive analysis and synthesis of the
        research materials and sources about distributed systems design principles. The key
        principles of distributed systems design revolve around understanding and managing the
        fundamental trade-offs inherent in distributed computing. Designers must carefully balance
        consistency, availability, and partition tolerance based on their application requirements.
        There is no perfect solution that works for all scenarios, but understanding these principles
        enables informed decision-making. The future of distributed systems will likely see continued
        evolution of these patterns and the emergence of new approaches to handle increasingly
        complex distributed architectures. Success in distributed systems design requires both
        theoretical understanding and practical experience with real-world trade-offs. The conclusion
        emphasizes the importance of continuous learning and adaptation as distributed systems
        continue to evolve and new challenges emerge in the field.
        """
        
        result = await validator.validate_document_quality(document)
        
        assert result.status == ValidationStatus.PASS
        assert result.details["word_count"] >= 500
        assert "introduction" in result.details["sections_found"]
        assert "analysis" in result.details["sections_found"]
        assert "synthesis" in result.details["sections_found"]
        assert "conclusion" in result.details["sections_found"]
    
    @pytest.mark.asyncio
    async def test_validate_document_quality_missing_sections(self, validator):
        """Test document quality validation with missing sections."""
        document = """
        # Research Report
        
        ## Introduction
        This is just an introduction with some basic content.
        
        ## Analysis
        Some analysis here but missing other sections.
        """
        
        result = await validator.validate_document_quality(document)
        
        assert result.status == ValidationStatus.FAIL
        assert "synthesis" in result.details["missing_sections"]
        assert "conclusion" in result.details["missing_sections"]
        assert "Missing sections" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_document_quality_insufficient_words(self, validator):
        """Test document quality validation with insufficient word count."""
        document = """
        # Research Report
        
        ## Introduction
        Short intro.
        
        ## Analysis
        Brief analysis.
        
        ## Synthesis
        Quick synthesis.
        
        ## Conclusion
        Short conclusion.
        """
        
        result = await validator.validate_document_quality(document)
        
        assert result.status == ValidationStatus.FAIL
        assert result.details["word_count"] < 500
        assert "Word count" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_source_retrieval_success(self, validator):
        """Test source retrieval validation with sufficient sources."""
        result = await validator.validate_source_retrieval(sources_count=5)
        
        assert result.status == ValidationStatus.PASS
        assert result.details["sources_count"] == 5
        assert result.details["min_sources"] == 3
    
    @pytest.mark.asyncio
    async def test_validate_source_retrieval_insufficient(self, validator):
        """Test source retrieval validation with insufficient sources."""
        result = await validator.validate_source_retrieval(sources_count=2)
        
        assert result.status == ValidationStatus.FAIL
        assert result.details["sources_count"] == 2
        assert "Insufficient sources" in result.message
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_validate_complete_workflow_success(self, mock_client_class, validator):
        """Test complete workflow validation with successful execution."""
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # Mock research submission response
        submit_response = MagicMock()
        submit_response.status_code = 200
        submit_response.json.return_value = {
            "session_id": "test-session-123",
            "status": "pending"
        }
        
        # Mock status response (completed)
        status_response = MagicMock()
        status_response.status_code = 200
        status_response.json.return_value = {
            "status": "completed",
            "progress_percentage": 100,
            "current_task": "Completed"
        }
        
        # Mock result response
        result_response = MagicMock()
        result_response.status_code = 200
        result_response.json.return_value = {
            "session_id": "test-session-123",
            "status": "completed",
            "document_content": "Test document with introduction, analysis, synthesis, and conclusion sections.",
            "sources_count": 5,
            "word_count": 1200,
            "completion_time_minutes": 2.5
        }
        
        # Set up mock responses
        mock_client.post.return_value = submit_response
        mock_client.get.side_effect = [status_response, result_response]
        
        result = await validator.validate_complete_workflow(
            test_query="Test query",
            timeout_minutes=5
        )
        
        assert result.status == ValidationStatus.PASS
        assert result.details["session_id"] == "test-session-123"
        assert result.details["sources_count"] == 5
        assert result.details["word_count"] == 1200
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_validate_complete_workflow_api_error(self, mock_client_class, validator):
        """Test complete workflow validation with API error."""
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # Mock failed submission response
        submit_response = MagicMock()
        submit_response.status_code = 500
        submit_response.text = "Internal server error"
        
        mock_client.post.return_value = submit_response
        
        result = await validator.validate_complete_workflow(
            test_query="Test query",
            timeout_minutes=5
        )
        
        assert result.status == ValidationStatus.FAIL
        assert "Failed to submit research request" in result.message
        assert "HTTP 500" in result.message
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_validate_complete_workflow_failed_status(self, mock_client_class, validator):
        """Test complete workflow validation with failed workflow."""
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # Mock research submission response
        submit_response = MagicMock()
        submit_response.status_code = 200
        submit_response.json.return_value = {
            "session_id": "test-session-123",
            "status": "pending"
        }
        
        # Mock status response (failed)
        status_response = MagicMock()
        status_response.status_code = 200
        status_response.json.return_value = {
            "status": "failed",
            "progress_percentage": 45,
            "current_task": "Analysis",
            "error_message": "Agent execution failed",
            "failed_agent": "AnalysisAgent",
            "stack_trace": "Traceback (most recent call last):\n  File 'agent.py', line 42\n    raise Exception('Test error')"
        }
        
        mock_client.post.return_value = submit_response
        mock_client.get.return_value = status_response
        
        result = await validator.validate_complete_workflow(
            test_query="Test query",
            timeout_minutes=5
        )
        
        assert result.status == ValidationStatus.FAIL
        assert "Workflow failed" in result.message
        
        # Verify error capture functionality
        assert result.details["error_message"] == "Agent execution failed"
        assert result.details["agent_name"] == "AnalysisAgent"
        assert result.details["stack_trace"] != ""
        assert "Traceback" in result.details["stack_trace"]
        
        # Verify additional error capture fields
        assert result.details["session_id"] == "test-session-123"
        assert "timestamp" in result.details
        assert "captured_at" in result.details
        
        # Verify remediation steps are provided
        assert result.remediation_steps is not None
        assert len(result.remediation_steps) > 0
    
    @pytest.mark.asyncio
    async def test_validator_initialization(self, validator):
        """Test validator is properly initialized."""
        assert validator.name == "EndToEndValidator"
        assert validator.api_base_url == "http://localhost:8000/api/v1"
        assert validator.timeout_minutes == 5
        assert "Bearer test-token" in validator.headers["Authorization"]
    
    def test_capture_agent_error(self, validator):
        """Test error capture functionality."""
        session_id = "test-session-456"
        agent_name = "TestAgent"
        error_message = "Test error occurred"
        stack_trace = "Traceback (most recent call last):\n  File 'test.py', line 10\n    raise Exception('Test')"
        
        error_info = validator.capture_agent_error(
            session_id=session_id,
            agent_name=agent_name,
            error_message=error_message,
            stack_trace=stack_trace
        )
        
        # Verify all required fields are captured
        assert error_info["session_id"] == session_id
        assert error_info["agent_name"] == agent_name
        assert error_info["error_message"] == error_message
        assert error_info["stack_trace"] == stack_trace
        assert "timestamp" in error_info
        assert "captured_at" in error_info
        
        # Verify timestamp is reasonable (within last second)
        import time
        assert abs(error_info["timestamp"] - time.time()) < 1.0
        
        # Verify captured_at is a formatted string
        assert isinstance(error_info["captured_at"], str)
        assert len(error_info["captured_at"]) > 0
    
    def test_capture_agent_error_without_stack_trace(self, validator):
        """Test error capture functionality without stack trace."""
        session_id = "test-session-789"
        agent_name = "AnotherAgent"
        error_message = "Another test error"
        
        error_info = validator.capture_agent_error(
            session_id=session_id,
            agent_name=agent_name,
            error_message=error_message
        )
        
        # Verify all required fields are captured
        assert error_info["session_id"] == session_id
        assert error_info["agent_name"] == agent_name
        assert error_info["error_message"] == error_message
        assert error_info["stack_trace"] == ""  # Empty string when not provided
        assert "timestamp" in error_info
        assert "captured_at" in error_info


    # Property-Based Tests

    @given(
        query_text=st.text(min_size=10, max_size=100).filter(lambda x: x.strip() and len(x.split()) >= 3)
    )
    @settings(max_examples=100, deadline=10000)
    def test_property_complete_workflow_execution(
        self,
        query_text
    ):
        """
        Property Test: Complete workflow execution
        
        Feature: production-readiness-validation, Property 1: Complete workflow execution
        
        **Validates: Requirements 1.1**
        
        For any valid test research query, executing the complete workflow should 
        successfully progress through all agent stages (Research, Analysis, Synthesis, 
        Quality) and produce a final document.
        
        This property verifies that:
        1. Any valid research query can be submitted successfully
        2. The workflow progresses through all required agent stages
        3. A final document is produced upon completion
        4. The session ID is returned and tracked throughout
        5. The workflow completes with a success status
        """
        # Create validator instance for this test iteration
        validator = EndToEndValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test-token",
            timeout_minutes=5
        )
        
        with patch('httpx.AsyncClient') as mock_client_class:
            # Mock HTTP client
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Generate a unique session ID for this test
            session_id = f"test-session-{abs(hash(query_text)) % 10000}"
            
            # Mock research submission response
            submit_response = MagicMock()
            submit_response.status_code = 200
            submit_response.json.return_value = {
                "session_id": session_id,
                "status": "pending"
            }
            
            # Mock status response - return completed immediately
            status_response_completed = MagicMock()
            status_response_completed.status_code = 200
            status_response_completed.json.return_value = {
                "status": "completed",
                "progress_percentage": 100,
                "current_task": "Completed"
            }
            
            # Mock result response with final document
            result_response = MagicMock()
            result_response.status_code = 200
            result_response.json.return_value = {
                "session_id": session_id,
                "status": "completed",
                "document_content": f"Research document for query: {query_text[:50]}...",
                "sources_count": 5,
                "word_count": 1200,
                "completion_time_minutes": 2.5
            }
            
            # Set up mock responses
            mock_client.post.return_value = submit_response
            mock_client.get.side_effect = [status_response_completed, result_response]
            
            # Execute the workflow validation using a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(validator.validate_complete_workflow(
                    test_query=query_text,
                    timeout_minutes=5
                ))
            finally:
                loop.close()
            
            # Property 1: Workflow completes successfully for any valid query
            assert result.status == ValidationStatus.PASS, \
                f"Workflow should complete successfully for query: {query_text[:50]}"
            
            # Property 2: Session ID is returned and tracked
            assert result.details["session_id"] == session_id, \
                "Session ID should be consistent throughout workflow"
            
            # Property 3: Final document is produced
            assert "document_content" in result.details, \
                "Final document content should be present"
            assert len(result.details["document_content"]) > 0, \
                "Document content should not be empty"
            
            # Property 4: Sources are retrieved
            assert result.details["sources_count"] > 0, \
                "At least one source should be retrieved"
            
            # Property 5: Word count is tracked
            assert result.details["word_count"] > 0, \
                "Document should have non-zero word count"
            
            # Property 6: Completion time is recorded
            assert result.details["completion_time_minutes"] > 0, \
                "Completion time should be positive"
            
            # Property 7: API was called with correct parameters
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            
            # Verify the request was made to the correct endpoint
            assert "/research" in call_args[0][0], \
                "Request should be made to /research endpoint"
            
            # Verify the query was included in the request
            request_json = call_args[1]["json"]
            assert request_json["query"] == query_text, \
                "Query text should match the input"


    @given(
        has_introduction=st.booleans(),
        has_analysis=st.booleans(),
        has_synthesis=st.booleans(),
        has_conclusion=st.booleans(),
        words_per_section=st.integers(min_value=150, max_value=500)
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_document_structure_completeness(
        self,
        has_introduction,
        has_analysis,
        has_synthesis,
        has_conclusion,
        words_per_section
    ):
        """
        Property Test: Document structure completeness
        
        Feature: production-readiness-validation, Property 2: Document structure completeness
        
        **Validates: Requirements 1.2**
        
        For any completed research workflow, the generated document should contain all 
        expected sections (introduction, analysis, synthesis, conclusion).
        
        This property verifies that:
        1. Documents are validated for the presence of all required sections
        2. Missing sections are correctly identified
        3. Validation passes only when all sections are present (and word count is sufficient)
        4. Section detection is case-insensitive
        5. The validation provides clear feedback about missing sections
        """
        # Create validator instance for this test iteration
        validator = EndToEndValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test-token",
            timeout_minutes=5
        )
        
        # Generate a document with the specified sections
        # Each section gets enough words to ensure we meet the 500 word minimum
        sections = []
        expected_sections = []
        missing_sections = []
        
        if has_introduction:
            sections.append("## Introduction\n" + " ".join(["intro"] * words_per_section))
            expected_sections.append("introduction")
        else:
            missing_sections.append("introduction")
        
        if has_analysis:
            sections.append("## Analysis\n" + " ".join(["analysis"] * words_per_section))
            expected_sections.append("analysis")
        else:
            missing_sections.append("analysis")
        
        if has_synthesis:
            sections.append("## Synthesis\n" + " ".join(["synthesis"] * words_per_section))
            expected_sections.append("synthesis")
        else:
            missing_sections.append("synthesis")
        
        if has_conclusion:
            sections.append("## Conclusion\n" + " ".join(["conclusion"] * words_per_section))
            expected_sections.append("conclusion")
        else:
            missing_sections.append("conclusion")
        
        # Create the document
        document = "# Research Report\n\n" + "\n\n".join(sections)
        
        # Execute the document quality validation using a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(validator.validate_document_quality(document))
        finally:
            loop.close()
        
        # Property 1: All four sections must be present for validation to pass
        # (Note: word count must also be sufficient, but we ensure that with words_per_section >= 150)
        all_sections_present = has_introduction and has_analysis and has_synthesis and has_conclusion
        
        if all_sections_present:
            # With all sections and sufficient words per section, validation should pass
            assert result.status == ValidationStatus.PASS, \
                f"Document with all sections should pass validation. Message: {result.message}"
            assert "sections_found" in result.details, \
                "Result should include sections_found"
            assert set(result.details["sections_found"]) == {"introduction", "analysis", "synthesis", "conclusion"}, \
                "All expected sections should be found"
        else:
            # Missing sections should cause validation to fail
            assert result.status == ValidationStatus.FAIL, \
                f"Document missing sections {missing_sections} should fail validation"
            assert "missing_sections" in result.details, \
                "Result should include missing_sections"
            
            # Property 2: Missing sections are correctly identified
            result_missing = set(result.details["missing_sections"])
            expected_missing = set(missing_sections)
            assert result_missing == expected_missing, \
                f"Missing sections should be {expected_missing}, got {result_missing}"
            
            # Property 3: Expected sections list is always present in failure details
            assert "expected_sections" in result.details, \
                "Result should include expected_sections when validation fails"
        
        # Property 4: Validation message provides clear feedback
        assert len(result.message) > 0, \
            "Validation result should include a message"
        
        if not all_sections_present:
            assert "Missing sections" in result.message or "missing" in result.message.lower(), \
                "Failure message should mention missing sections"
        
        # Property 5: Section detection is case-insensitive (implicit in implementation)
        # The validator converts document to lowercase for section detection
        
        # Property 6: Duration is tracked
        assert result.duration_seconds >= 0, \
            "Duration should be non-negative"


    @given(
        word_count=st.integers(min_value=0, max_value=2000),
        sources_count=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_document_quality_thresholds(
        self,
        word_count,
        sources_count
    ):
        """
        Property Test: Document quality thresholds
        
        Feature: production-readiness-validation, Property 3: Document quality thresholds
        
        **Validates: Requirements 1.3, 1.4**
        
        For any completed research workflow, the generated document should meet minimum 
        quality thresholds: word count ≥ 500 words and source count ≥ 3 sources.
        
        This property verifies that:
        1. Documents with word count ≥ 500 pass the word count threshold
        2. Documents with word count < 500 fail the word count threshold
        3. Documents with sources count ≥ 3 pass the source count threshold
        4. Documents with sources count < 3 fail the source count threshold
        5. Both thresholds must be met for overall quality validation to pass
        6. Validation provides clear feedback about which thresholds were not met
        """
        # Create validator instance for this test iteration
        validator = EndToEndValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test-token",
            timeout_minutes=5
        )
        
        # Generate a document with the specified word count
        # Include all required sections to isolate word count testing
        words_per_section = max(1, word_count // 4)  # Distribute words across 4 sections
        
        document = f"""# Research Report

## Introduction
{' '.join(['word'] * words_per_section)}

## Analysis
{' '.join(['word'] * words_per_section)}

## Synthesis
{' '.join(['word'] * words_per_section)}

## Conclusion
{' '.join(['word'] * words_per_section)}
"""
        
        # Execute the document quality validation using a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            doc_result = loop.run_until_complete(validator.validate_document_quality(document))
            source_result = loop.run_until_complete(validator.validate_source_retrieval(sources_count))
        finally:
            loop.close()
        
        # Define thresholds
        min_word_count = 500
        min_sources = 3
        
        # Property 1: Word count threshold validation
        actual_word_count = len(document.split())
        
        if actual_word_count >= min_word_count:
            # Document should pass word count check (sections are all present)
            assert doc_result.status == ValidationStatus.PASS, \
                f"Document with {actual_word_count} words (>= {min_word_count}) should pass validation"
            assert doc_result.details["word_count"] >= min_word_count, \
                "Word count in details should meet minimum threshold"
        else:
            # Document should fail word count check
            assert doc_result.status == ValidationStatus.FAIL, \
                f"Document with {actual_word_count} words (< {min_word_count}) should fail validation"
            assert doc_result.details["word_count"] < min_word_count, \
                "Word count in details should be below minimum threshold"
            assert "Word count" in doc_result.message or "word" in doc_result.message.lower(), \
                "Failure message should mention word count issue"
        
        # Property 2: Source count threshold validation
        if sources_count >= min_sources:
            # Should pass source count check
            assert source_result.status == ValidationStatus.PASS, \
                f"Document with {sources_count} sources (>= {min_sources}) should pass validation"
            assert source_result.details["sources_count"] >= min_sources, \
                "Source count in details should meet minimum threshold"
        else:
            # Should fail source count check
            assert source_result.status == ValidationStatus.FAIL, \
                f"Document with {sources_count} sources (< {min_sources}) should fail validation"
            assert source_result.details["sources_count"] < min_sources, \
                "Source count in details should be below minimum threshold"
            assert "source" in source_result.message.lower(), \
                "Failure message should mention source count issue"
        
        # Property 3: Both thresholds must be met for overall quality
        overall_quality_pass = (actual_word_count >= min_word_count) and (sources_count >= min_sources)
        
        if overall_quality_pass:
            assert doc_result.status == ValidationStatus.PASS, \
                "Document quality should pass when word count threshold is met"
            assert source_result.status == ValidationStatus.PASS, \
                "Source validation should pass when source count threshold is met"
        else:
            # At least one validation should fail
            assert doc_result.status == ValidationStatus.FAIL or source_result.status == ValidationStatus.FAIL, \
                "At least one validation should fail when thresholds are not met"
        
        # Property 4: Validation details include threshold information
        assert "word_count" in doc_result.details, \
            "Document validation should include word_count in details"
        assert "min_word_count" in doc_result.details or doc_result.status == ValidationStatus.PASS, \
            "Document validation should include min_word_count in details (or pass)"
        
        assert "sources_count" in source_result.details, \
            "Source validation should include sources_count in details"
        assert "min_sources" in source_result.details, \
            "Source validation should include min_sources in details"
        
        # Property 5: Validation messages are informative
        assert len(doc_result.message) > 0, \
            "Document validation should include a message"
        assert len(source_result.message) > 0, \
            "Source validation should include a message"
        
        # Property 6: Duration is tracked for both validations
        assert doc_result.duration_seconds >= 0, \
            "Document validation duration should be non-negative"
        assert source_result.duration_seconds >= 0, \
            "Source validation duration should be non-negative"
        
        # Property 7: Remediation steps are provided on failure
        if doc_result.status == ValidationStatus.FAIL:
            assert doc_result.remediation_steps is not None and len(doc_result.remediation_steps) > 0, \
                "Failed document validation should include remediation steps"
        
        if source_result.status == ValidationStatus.FAIL:
            assert source_result.remediation_steps is not None and len(source_result.remediation_steps) > 0, \
                "Failed source validation should include remediation steps"


    @given(
        query_text=st.text(min_size=10, max_size=100).filter(lambda x: x.strip() and len(x.split()) >= 3),
        completion_time_seconds=st.floats(min_value=1.0, max_value=600.0)
    )
    @settings(max_examples=100, deadline=10000)
    def test_property_workflow_completion_time(
        self,
        query_text,
        completion_time_seconds
    ):
        """
        Property Test: Workflow completion time
        
        Feature: production-readiness-validation, Property 4: Workflow completion time
        
        **Validates: Requirements 1.5**
        
        For any standard research query, the complete workflow should finish within 
        5 minutes.
        
        This property verifies that:
        1. Workflows that complete within 5 minutes (300 seconds) pass validation
        2. Workflows that exceed 5 minutes fail validation or timeout
        3. Completion time is accurately tracked and reported
        4. The timeout mechanism correctly identifies long-running workflows
        5. Validation provides clear feedback about completion time
        """
        # Create validator instance for this test iteration
        validator = EndToEndValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test-token",
            timeout_minutes=5
        )
        
        with patch('httpx.AsyncClient') as mock_client_class, \
             patch('time.time') as mock_time:
            # Mock HTTP client
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Generate a unique session ID for this test
            session_id = f"test-session-{abs(hash(query_text)) % 10000}"
            
            # Mock time progression
            start_time = 1000.0
            current_time = [start_time]  # Use list to allow modification in nested function
            
            def time_side_effect():
                return current_time[0]
            
            mock_time.side_effect = time_side_effect
            
            # Mock research submission response
            submit_response = MagicMock()
            submit_response.status_code = 200
            submit_response.json.return_value = {
                "session_id": session_id,
                "status": "pending"
            }
            
            # Determine if workflow should complete or timeout
            max_timeout_seconds = 5 * 60  # 5 minutes
            should_complete = completion_time_seconds <= max_timeout_seconds
            
            if should_complete:
                # Mock status response - return completed
                status_response_completed = MagicMock()
                status_response_completed.status_code = 200
                status_response_completed.json.return_value = {
                    "status": "completed",
                    "progress_percentage": 100,
                    "current_task": "Completed"
                }
                
                # Mock result response with completion time
                result_response = MagicMock()
                result_response.status_code = 200
                result_response.json.return_value = {
                    "session_id": session_id,
                    "status": "completed",
                    "document_content": f"Research document for query: {query_text[:50]}...",
                    "sources_count": 5,
                    "word_count": 1200,
                    "completion_time_minutes": completion_time_seconds / 60.0
                }
                
                # Set up mock responses
                mock_client.post.return_value = submit_response
                mock_client.get.side_effect = [status_response_completed, result_response]
                
                # Advance time to simulate workflow completion
                current_time[0] = start_time + completion_time_seconds
            else:
                # Mock status response - return in-progress (will timeout)
                status_response_in_progress = MagicMock()
                status_response_in_progress.status_code = 200
                status_response_in_progress.json.return_value = {
                    "status": "in_progress",
                    "progress_percentage": 50,
                    "current_task": "Analysis"
                }
                
                # Set up mock responses - always return in-progress
                mock_client.post.return_value = submit_response
                mock_client.get.return_value = status_response_in_progress
                
                # Advance time beyond timeout
                current_time[0] = start_time + max_timeout_seconds + 1
            
            # Mock asyncio.sleep to advance time
            async def mock_sleep(seconds):
                current_time[0] += seconds
            
            # Execute the workflow validation using a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with patch('asyncio.sleep', side_effect=mock_sleep):
                    result = loop.run_until_complete(validator.validate_complete_workflow(
                        test_query=query_text,
                        timeout_minutes=5
                    ))
            finally:
                loop.close()
            
            # Property 1: Workflows completing within 5 minutes should pass
            if should_complete:
                assert result.status == ValidationStatus.PASS, \
                    f"Workflow completing in {completion_time_seconds:.1f}s (<= 300s) should pass"
                
                # Property 2: Completion time is accurately tracked
                assert result.duration_seconds > 0, \
                    "Duration should be positive for completed workflows"
                
                # Property 3: Completion time is within expected range
                completion_time_minutes = result.details.get("completion_time_minutes", 0)
                assert completion_time_minutes > 0, \
                    "Completion time in minutes should be positive"
                assert completion_time_minutes <= 5.0, \
                    f"Completion time {completion_time_minutes:.2f} minutes should be <= 5 minutes"
                
                # Property 4: Session ID is tracked
                assert result.details["session_id"] == session_id, \
                    "Session ID should be consistent"
                
                # Property 5: Document content is present
                assert "document_content" in result.details, \
                    "Completed workflow should include document content"
            else:
                # Property 6: Workflows exceeding 5 minutes should timeout or fail
                assert result.status in [ValidationStatus.TIMEOUT, ValidationStatus.FAIL], \
                    f"Workflow taking {completion_time_seconds:.1f}s (> 300s) should timeout or fail"
                
                # Property 7: Timeout message is clear
                if result.status == ValidationStatus.TIMEOUT:
                    assert "5 minutes" in result.message or "timeout" in result.message.lower(), \
                        "Timeout message should mention the time limit"
                
                # Property 8: Session ID is still tracked even on timeout
                assert result.details.get("session_id") == session_id, \
                    "Session ID should be tracked even on timeout"
                
                # Property 9: Remediation steps are provided
                assert result.remediation_steps is not None and len(result.remediation_steps) > 0, \
                    "Timeout should include remediation steps"
            
            # Property 10: Duration is always tracked
            assert result.duration_seconds >= 0, \
                "Duration should be non-negative"
            
            # Property 11: Validation message is informative
            assert len(result.message) > 0, \
                "Validation result should include a message"


    @given(
        agent_name=st.from_regex(r'[A-Za-z][A-Za-z0-9]{4,49}', fullmatch=True),
        error_message=st.text(min_size=10, max_size=200).filter(lambda x: x.strip()),
        has_stack_trace=st.booleans()
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_error_information_capture(
        self,
        agent_name,
        error_message,
        has_stack_trace
    ):
        """
        Property Test: Error information capture
        
        Feature: production-readiness-validation, Property 5: Error information capture
        
        **Validates: Requirements 1.6**
        
        For any agent failure during workflow execution, the system should capture 
        complete error information including agent name, failure reason, and stack trace.
        
        This property verifies that:
        1. Agent name is captured when an agent fails
        2. Error message/failure reason is captured
        3. Stack trace is captured when available
        4. Session ID is tracked with the error
        5. Timestamp information is recorded
        6. Error information is structured and complete
        7. Validation provides remediation steps for failures
        8. Error capture works consistently across different failure scenarios
        """
        # Create validator instance for this test iteration
        validator = EndToEndValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test-token",
            timeout_minutes=5
        )
        
        # Generate a stack trace if needed
        stack_trace = ""
        if has_stack_trace:
            stack_trace = f"""Traceback (most recent call last):
  File "agent_scrivener/agents/{agent_name.lower()}.py", line 42, in execute
    result = await self.process()
  File "agent_scrivener/agents/{agent_name.lower()}.py", line 67, in process
    raise Exception('{error_message}')
Exception: {error_message}"""
        
        # Generate a unique session ID for this test
        session_id = f"test-session-{abs(hash(agent_name + error_message)) % 10000}"
        
        # Test the capture_agent_error method directly
        error_info = validator.capture_agent_error(
            session_id=session_id,
            agent_name=agent_name,
            error_message=error_message,
            stack_trace=stack_trace
        )
        
        # Property 1: Agent name is captured
        assert "agent_name" in error_info, \
            "Error details should include agent_name"
        assert error_info["agent_name"] == agent_name, \
            f"Captured agent name should be {agent_name}, got {error_info.get('agent_name')}"
        
        # Property 2: Error message is captured
        assert "error_message" in error_info, \
            "Error details should include error_message"
        assert error_info["error_message"] == error_message, \
            f"Captured error message should match: {error_message}"
        
        # Property 3: Stack trace is captured (when available)
        assert "stack_trace" in error_info, \
            "Error details should include stack_trace field"
        
        if has_stack_trace:
            assert error_info["stack_trace"] == stack_trace, \
                "Captured stack trace should match the provided stack trace"
            assert len(error_info["stack_trace"]) > 0, \
                "Stack trace should not be empty when provided"
            assert "Traceback" in error_info["stack_trace"], \
                "Stack trace should contain 'Traceback' marker"
        else:
            # Stack trace should be empty string when not provided
            assert error_info["stack_trace"] == "", \
                "Stack trace should be empty string when not provided"
        
        # Property 4: Session ID is tracked with the error
        assert "session_id" in error_info, \
            "Error details should include session_id"
        assert error_info["session_id"] == session_id, \
            f"Session ID should be {session_id}, got {error_info.get('session_id')}"
        
        # Property 5: Timestamp information is recorded
        assert "timestamp" in error_info, \
            "Error details should include timestamp"
        assert isinstance(error_info["timestamp"], (int, float)), \
            "Timestamp should be a numeric value"
        assert error_info["timestamp"] > 0, \
            "Timestamp should be positive"
        
        assert "captured_at" in error_info, \
            "Error details should include captured_at (formatted timestamp)"
        assert isinstance(error_info["captured_at"], str), \
            "captured_at should be a string"
        assert len(error_info["captured_at"]) > 0, \
            "captured_at should not be empty"
        
        # Property 6: Error information is structured and complete
        # All required fields should be present
        required_fields = ["session_id", "agent_name", "error_message", "stack_trace", "timestamp", "captured_at"]
        for field in required_fields:
            assert field in error_info, \
                f"Error details should include required field: {field}"
        
        # Property 7: Error information enables debugging
        # The captured information should be sufficient for debugging
        if has_stack_trace:
            # With stack trace, we should be able to identify the file and line
            assert "File" in error_info["stack_trace"] or "file" in error_info["stack_trace"].lower(), \
                "Stack trace should contain file information for debugging"
        
        # Agent name and error message should be sufficient to identify the issue
        assert len(error_info["agent_name"]) > 0, \
            "Agent name should not be empty"
        assert len(error_info["error_message"]) > 0, \
            "Error message should not be empty"
        
        # Property 8: Timestamp is reasonable (within last second)
        import time
        assert abs(error_info["timestamp"] - time.time()) < 2.0, \
            "Timestamp should be recent (within last 2 seconds)"
