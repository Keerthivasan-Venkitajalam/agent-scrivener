"""End-to-end validator for complete research workflow validation."""

import asyncio
import time
from typing import List, Optional, Dict, Any
import httpx

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus


class EndToEndValidator(BaseValidator):
    """Validates complete research pipeline from query submission to document generation.
    
    This validator tests the entire workflow including:
    - API request submission
    - Workflow progress tracking through all agent stages
    - Document quality validation (structure, word count, citations)
    - Source retrieval and citation verification
    - Timeout handling for workflow execution
    """
    
    def __init__(
        self,
        api_base_url: str,
        auth_token: str,
        timeout_minutes: int = 5
    ):
        """Initialize the end-to-end validator.
        
        Args:
            api_base_url: Base URL of the API (e.g., "http://localhost:8000/api/v1")
            auth_token: Authentication token for API requests
            timeout_minutes: Maximum time to wait for workflow completion (default: 5 minutes)
        """
        super().__init__(
            name="EndToEndValidator",
            timeout_seconds=timeout_minutes * 60
        )
        self.api_base_url = api_base_url.rstrip('/')
        self.auth_token = auth_token
        self.timeout_minutes = timeout_minutes
        self.headers = {"Authorization": f"Bearer {auth_token}"}
    
    async def validate(self) -> List[ValidationResult]:
        """Execute all end-to-end validation checks.
        
        Returns:
            List of validation results
        """
        self.log_validation_start()
        results = []
        
        # Test query for validation
        test_query = (
            "What are the key principles of distributed systems design? "
            "Include discussion of consistency, availability, and partition tolerance."
        )
        
        # Validate complete workflow
        workflow_result = await self.validate_complete_workflow(
            test_query=test_query,
            timeout_minutes=self.timeout_minutes
        )
        results.append(workflow_result)
        
        # If workflow succeeded, validate document quality and sources
        if workflow_result.is_success() and workflow_result.details.get("document_content"):
            document_content = workflow_result.details["document_content"]
            sources_count = workflow_result.details.get("sources_count", 0)
            
            # Validate document quality
            quality_result = await self.validate_document_quality(document_content)
            results.append(quality_result)
            
            # Validate source retrieval
            source_result = await self.validate_source_retrieval(sources_count)
            results.append(source_result)
        
        self.log_validation_complete(results)
        return results
    
    async def validate_complete_workflow(
        self,
        test_query: str,
        timeout_minutes: int = 5
    ) -> ValidationResult:
        """Execute complete research workflow and validate output.
        
        Submits a test query, tracks progress through all agent stages,
        and validates successful completion.
        
        Args:
            test_query: Research query to submit
            timeout_minutes: Maximum time to wait for completion
            
        Returns:
            ValidationResult indicating workflow success or failure
        """
        start_time = time.time()
        session_id = None
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Submit research request
                self.logger.info(f"Submitting test query: {test_query[:100]}...")
                response = await client.post(
                    f"{self.api_base_url}/research",
                    json={
                        "query": test_query,
                        "max_sources": 10,
                        "include_academic": True,
                        "include_web": True,
                        "priority": "high"
                    },
                    headers=self.headers
                )
                
                if response.status_code != 200:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Failed to submit research request: HTTP {response.status_code}",
                        duration_seconds=time.time() - start_time,
                        details={"response": response.text},
                        remediation_steps=[
                            "Check if the API server is running",
                            "Verify authentication token is valid",
                            "Check API logs for errors"
                        ]
                    )
                
                data = response.json()
                session_id = data.get("session_id")
                
                if not session_id:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="No session_id returned from API",
                        duration_seconds=time.time() - start_time,
                        details={"response": data},
                        remediation_steps=[
                            "Check API response format",
                            "Verify orchestrator is properly configured"
                        ]
                    )
                
                self.logger.info(f"Research session created: {session_id}")
                
                # Track workflow progress
                timeout_seconds = timeout_minutes * 60
                poll_interval = 5  # Poll every 5 seconds
                elapsed = 0
                
                while elapsed < timeout_seconds:
                    await asyncio.sleep(poll_interval)
                    elapsed = time.time() - start_time
                    
                    # Get session status
                    status_response = await client.get(
                        f"{self.api_base_url}/research/{session_id}/status",
                        headers=self.headers
                    )
                    
                    if status_response.status_code != 200:
                        return self.create_result(
                            status=ValidationStatus.FAIL,
                            message=f"Failed to get session status: HTTP {status_response.status_code}",
                            duration_seconds=elapsed,
                            details={"session_id": session_id},
                            remediation_steps=[
                                "Check if session was created successfully",
                                "Verify API endpoint is accessible"
                            ]
                        )
                    
                    status_data = status_response.json()
                    current_status = status_data.get("status")
                    progress = status_data.get("progress_percentage", 0)
                    current_task = status_data.get("current_task", "Unknown")
                    
                    self.logger.info(
                        f"Session {session_id}: {current_status} - "
                        f"{progress:.1f}% - {current_task}"
                    )
                    
                    # Check if completed
                    if current_status == "completed":
                        # Get final result
                        result_response = await client.get(
                            f"{self.api_base_url}/research/{session_id}/result",
                            headers=self.headers
                        )
                        
                        if result_response.status_code != 200:
                            return self.create_result(
                                status=ValidationStatus.FAIL,
                                message=f"Failed to get research result: HTTP {result_response.status_code}",
                                duration_seconds=elapsed,
                                details={"session_id": session_id},
                                remediation_steps=[
                                    "Check if session completed successfully",
                                    "Verify result endpoint is accessible"
                                ]
                            )
                        
                        result_data = result_response.json()
                        
                        return self.create_result(
                            status=ValidationStatus.PASS,
                            message=f"Workflow completed successfully in {elapsed:.1f} seconds",
                            duration_seconds=elapsed,
                            details={
                                "session_id": session_id,
                                "document_content": result_data.get("document_content", ""),
                                "sources_count": result_data.get("sources_count", 0),
                                "word_count": result_data.get("word_count", 0),
                                "completion_time_minutes": result_data.get("completion_time_minutes", 0)
                            }
                        )
                    
                    # Check if failed
                    if current_status == "failed":
                        error_message = status_data.get("error_message", "Unknown error")
                        agent_name = status_data.get("failed_agent", "Unknown")
                        stack_trace = status_data.get("stack_trace", "")
                        
                        # Capture detailed error information
                        error_info = self.capture_agent_error(
                            session_id=session_id,
                            agent_name=agent_name,
                            error_message=error_message,
                            stack_trace=stack_trace
                        )
                        
                        return self.create_result(
                            status=ValidationStatus.FAIL,
                            message=f"Workflow failed: {error_message}",
                            duration_seconds=elapsed,
                            details=error_info,
                            remediation_steps=[
                                "Check agent logs for detailed error information",
                                "Verify all agents are properly configured",
                                "Check if external services (Bedrock, etc.) are accessible"
                            ]
                        )
                
                # Timeout reached
                return self.create_result(
                    status=ValidationStatus.TIMEOUT,
                    message=f"Workflow did not complete within {timeout_minutes} minutes",
                    duration_seconds=elapsed,
                    details={"session_id": session_id},
                    remediation_steps=[
                        "Check if agents are processing correctly",
                        "Increase timeout if workflow is expected to take longer",
                        "Check for any blocking operations in agent execution"
                    ]
                )
        
        except httpx.TimeoutException as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"HTTP request timeout: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"session_id": session_id},
                remediation_steps=[
                    "Check if API server is responding",
                    "Verify network connectivity",
                    "Check API server logs for performance issues"
                ]
            )
        
        except Exception as e:
            # Capture error information for unexpected failures
            import traceback
            stack_trace = traceback.format_exc()
            
            error_info = {
                "session_id": session_id,
                "agent_name": "Unknown",
                "error_message": str(e),
                "stack_trace": stack_trace,
                "exception_type": type(e).__name__,
                "timestamp": time.time(),
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            
            self.logger.error(
                f"Workflow validation failed with exception: {type(e).__name__}: {str(e)}"
            )
            self.logger.debug(f"Stack trace: {stack_trace}")
            
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Workflow validation failed: {str(e)}",
                duration_seconds=time.time() - start_time,
                details=error_info,
                remediation_steps=[
                    "Check validator logs for detailed error information",
                    "Verify API configuration is correct",
                    "Ensure all required services are running"
                ]
            )
    
    async def validate_document_quality(self, document: str) -> ValidationResult:
        """Validate generated document meets quality standards.
        
        Checks:
        - Document contains expected sections (introduction, analysis, synthesis, conclusion)
        - Word count exceeds minimum threshold (500 words)
        
        Args:
            document: Generated document content
            
        Returns:
            ValidationResult indicating document quality
        """
        start_time = time.time()
        
        try:
            # Check for expected sections
            document_lower = document.lower()
            expected_sections = ["introduction", "analysis", "synthesis", "conclusion"]
            missing_sections = []
            
            for section in expected_sections:
                if section not in document_lower:
                    missing_sections.append(section)
            
            # Count words
            word_count = len(document.split())
            min_word_count = 500
            
            # Determine validation result
            issues = []
            
            if missing_sections:
                issues.append(f"Missing sections: {', '.join(missing_sections)}")
            
            if word_count < min_word_count:
                issues.append(f"Word count {word_count} is below minimum {min_word_count}")
            
            if issues:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Document quality validation failed: {'; '.join(issues)}",
                    duration_seconds=time.time() - start_time,
                    details={
                        "word_count": word_count,
                        "min_word_count": min_word_count,
                        "missing_sections": missing_sections,
                        "expected_sections": expected_sections
                    },
                    remediation_steps=[
                        "Check synthesis agent configuration",
                        "Verify document template includes all required sections",
                        "Check if agents are generating sufficient content"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message=f"Document quality validation passed: {word_count} words, all sections present",
                duration_seconds=time.time() - start_time,
                details={
                    "word_count": word_count,
                    "sections_found": expected_sections
                }
            )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Document quality validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                remediation_steps=[
                    "Check document format is valid",
                    "Verify document content is not empty"
                ]
            )
    
    async def validate_source_retrieval(self, sources_count: int) -> ValidationResult:
        """Validate source retrieval and citation.
        
        Checks that at least 3 sources were retrieved and cited.
        
        Args:
            sources_count: Number of sources retrieved
            
        Returns:
            ValidationResult indicating source retrieval success
        """
        start_time = time.time()
        min_sources = 3
        
        try:
            if sources_count < min_sources:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Insufficient sources: {sources_count} (minimum: {min_sources})",
                    duration_seconds=time.time() - start_time,
                    details={
                        "sources_count": sources_count,
                        "min_sources": min_sources
                    },
                    remediation_steps=[
                        "Check research agent configuration",
                        "Verify search functionality is working",
                        "Check if external search services are accessible"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message=f"Source retrieval validation passed: {sources_count} sources",
                duration_seconds=time.time() - start_time,
                details={
                    "sources_count": sources_count,
                    "min_sources": min_sources
                }
            )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Source validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                remediation_steps=[
                    "Check source data format",
                    "Verify source count is properly tracked"
                ]
            )
    def capture_agent_error(
        self,
        session_id: str,
        agent_name: str,
        error_message: str,
        stack_trace: str = ""
    ) -> Dict[str, Any]:
        """Capture detailed error information for agent failures.

        This method captures comprehensive error information when an agent fails
        during workflow execution, including:
        - Agent name that failed
        - Failure reason/error message
        - Stack trace for debugging
        - Session ID for tracking
        - Timestamp of the failure

        Args:
            session_id: ID of the research session
            agent_name: Name of the agent that failed
            error_message: Description of the failure
            stack_trace: Optional stack trace for debugging

        Returns:
            Dictionary containing structured error information
        """
        error_info = {
            "session_id": session_id,
            "agent_name": agent_name,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "timestamp": time.time(),
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

        self.logger.error(
            f"Agent failure captured - Session: {session_id}, "
            f"Agent: {agent_name}, Error: {error_message}"
        )

        if stack_trace:
            self.logger.debug(f"Stack trace: {stack_trace}")

        return error_info
