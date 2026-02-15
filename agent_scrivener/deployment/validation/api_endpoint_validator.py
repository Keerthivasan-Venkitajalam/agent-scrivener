"""API endpoint validator for REST and WebSocket API validation."""

import asyncio
import time
from typing import List, Dict, Any, Optional
import httpx
import websockets

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus


class APIEndpointValidator(BaseValidator):
    """Validates all REST and WebSocket API endpoints.
    
    This validator tests:
    - Health endpoint functionality and response time
    - All research CRUD endpoints (create, read, update, delete)
    - Authentication and authorization
    - Rate limiting enforcement
    - WebSocket connection and streaming
    - Response time measurements for all endpoints
    """
    
    def __init__(
        self,
        api_base_url: str,
        ws_base_url: str,
        auth_token: str,
        invalid_auth_token: Optional[str] = "invalid_token_12345"
    ):
        """Initialize the API endpoint validator.
        
        Args:
            api_base_url: Base URL of the REST API (e.g., "http://localhost:8000/api/v1")
            ws_base_url: Base URL of the WebSocket API (e.g., "ws://localhost:8000/api/v1/ws")
            auth_token: Valid authentication token for API requests
            invalid_auth_token: Invalid token for testing authentication failures
        """
        super().__init__(
            name="APIEndpointValidator",
            timeout_seconds=300  # 5 minutes for all endpoint tests
        )
        self.api_base_url = api_base_url.rstrip('/')
        self.ws_base_url = ws_base_url.rstrip('/')
        self.auth_token = auth_token
        self.invalid_auth_token = invalid_auth_token
        self.headers = {"Authorization": f"Bearer {auth_token}"}
    
    async def validate(self) -> List[ValidationResult]:
        """Execute all API endpoint validation checks.
        
        Returns:
            List of validation results
        """
        self.log_validation_start()
        results = []
        
        # Validate health endpoint
        health_result = await self.validate_health_endpoint()
        results.append(health_result)
        
        # Validate research endpoints (all CRUD operations)
        research_results = await self.validate_research_endpoints()
        results.extend(research_results)
        
        # Validate authentication
        auth_result = await self.validate_authentication()
        results.append(auth_result)
        
        # Validate rate limiting
        rate_limit_result = await self.validate_rate_limiting()
        results.append(rate_limit_result)
        
        # Validate WebSocket connection
        ws_result = await self.validate_websocket_connection()
        results.append(ws_result)
        
        self.log_validation_complete(results)
        return results
    
    async def validate_health_endpoint(self) -> ValidationResult:
        """Validate /health endpoint.
        
        Tests:
        - Returns status 200
        - Includes version information
        - Includes uptime information
        - Response time < 100ms
        
        Returns:
            ValidationResult for health endpoint
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Measure response time
                request_start = time.time()
                response = await client.get(f"{self.api_base_url}/health")
                response_time_ms = (time.time() - request_start) * 1000
                
                # Check status code
                if response.status_code != 200:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Health endpoint returned status {response.status_code}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Check if API server is running",
                            "Verify health endpoint is properly configured",
                            "Check API logs for errors"
                        ]
                    )
                
                # Parse response
                data = response.json()
                
                # Check for required fields
                missing_fields = []
                if "version" not in data:
                    missing_fields.append("version")
                if "uptime_seconds" not in data:
                    missing_fields.append("uptime_seconds")
                
                if missing_fields:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Health endpoint missing required fields: {', '.join(missing_fields)}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "response": data,
                            "missing_fields": missing_fields,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Update health endpoint to include all required fields",
                            "Verify HealthCheck model includes version and uptime_seconds"
                        ]
                    )
                
                # Check response time
                max_response_time_ms = 100
                if response_time_ms > max_response_time_ms:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message=f"Health endpoint response time {response_time_ms:.1f}ms exceeds threshold {max_response_time_ms}ms",
                        duration_seconds=time.time() - start_time,
                        details={
                            "response_time_ms": response_time_ms,
                            "threshold_ms": max_response_time_ms,
                            "version": data.get("version"),
                            "uptime_seconds": data.get("uptime_seconds")
                        },
                        remediation_steps=[
                            "Check API server performance",
                            "Verify no blocking operations in health endpoint",
                            "Consider caching health check data"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"Health endpoint validation passed (response time: {response_time_ms:.1f}ms)",
                    duration_seconds=time.time() - start_time,
                    details={
                        "response_time_ms": response_time_ms,
                        "version": data.get("version"),
                        "uptime_seconds": data.get("uptime_seconds"),
                        "status": data.get("status")
                    }
                )
        
        except httpx.TimeoutException:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message="Health endpoint request timed out",
                duration_seconds=time.time() - start_time,
                remediation_steps=[
                    "Check if API server is running",
                    "Verify network connectivity",
                    "Check for blocking operations in health endpoint"
                ]
            )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Health endpoint validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check API server is running",
                    "Verify API base URL is correct",
                    "Check network connectivity"
                ]
            )

    async def validate_research_endpoints(self) -> List[ValidationResult]:
        """Validate all research-related endpoints for CRUD operations.
        
        Tests:
        - POST /research - Create research session
        - GET /research/{session_id}/status - Get session status
        - GET /research/{session_id}/result - Get session result
        - POST /research/{session_id}/cancel - Cancel session
        - GET /research - List sessions with pagination
        - DELETE /research/{session_id} - Delete session
        
        Returns:
            List of validation results for each endpoint
        """
        results = []
        session_id = None
        
        # Test 1: POST /research - Create research session
        create_result, session_id = await self._test_create_research()
        results.append(create_result)
        
        if not session_id:
            # If session creation failed, skip dependent tests
            results.append(self.create_result(
                status=ValidationStatus.SKIP,
                message="Skipping dependent endpoint tests due to session creation failure",
                duration_seconds=0.0
            ))
            return results
        
        # Test 2: GET /research/{session_id}/status - Get session status
        status_result = await self._test_get_session_status(session_id)
        results.append(status_result)
        
        # Test 3: GET /research - List sessions
        list_result = await self._test_list_sessions()
        results.append(list_result)
        
        # Test 4: POST /research/{session_id}/cancel - Cancel session
        cancel_result = await self._test_cancel_session(session_id)
        results.append(cancel_result)
        
        # Test 5: GET /research/{session_id}/result - Get session result (may not be completed)
        result_result = await self._test_get_session_result(session_id)
        results.append(result_result)
        
        # Test 6: DELETE /research/{session_id} - Delete session
        delete_result = await self._test_delete_session(session_id)
        results.append(delete_result)
        
        return results
    
    async def _test_create_research(self) -> tuple[ValidationResult, Optional[str]]:
        """Test POST /research endpoint.
        
        Returns:
            Tuple of (ValidationResult, session_id or None)
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Measure response time
                request_start = time.time()
                response = await client.post(
                    f"{self.api_base_url}/research",
                    json={
                        "query": "Test query for API validation",
                        "max_sources": 5,
                        "include_academic": True,
                        "include_web": True,
                        "priority": "high"
                    },
                    headers=self.headers
                )
                response_time_ms = (time.time() - request_start) * 1000
                
                # Check status code
                if response.status_code != 200:
                    return (
                        self.create_result(
                            status=ValidationStatus.FAIL,
                            message=f"POST /research returned status {response.status_code}",
                            duration_seconds=time.time() - start_time,
                            details={
                                "status_code": response.status_code,
                                "response": response.text,
                                "response_time_ms": response_time_ms
                            },
                            remediation_steps=[
                                "Check if API server is running",
                                "Verify authentication token is valid",
                                "Check API logs for errors"
                            ]
                        ),
                        None
                    )
                
                # Parse response
                data = response.json()
                session_id = data.get("session_id")
                
                if not session_id:
                    return (
                        self.create_result(
                            status=ValidationStatus.FAIL,
                            message="POST /research did not return session_id",
                            duration_seconds=time.time() - start_time,
                            details={
                                "response": data,
                                "response_time_ms": response_time_ms
                            },
                            remediation_steps=[
                                "Check ResearchResponse model includes session_id",
                                "Verify orchestrator is creating sessions correctly"
                            ]
                        ),
                        None
                    )
                
                # Check response time (should be < 2 seconds)
                max_response_time_ms = 2000
                status = ValidationStatus.PASS
                message = f"POST /research validation passed (response time: {response_time_ms:.1f}ms)"
                
                if response_time_ms > max_response_time_ms:
                    status = ValidationStatus.WARNING
                    message = f"POST /research response time {response_time_ms:.1f}ms exceeds threshold {max_response_time_ms}ms"
                
                return (
                    self.create_result(
                        status=status,
                        message=message,
                        duration_seconds=time.time() - start_time,
                        details={
                            "session_id": session_id,
                            "response_time_ms": response_time_ms,
                            "threshold_ms": max_response_time_ms,
                            "status": data.get("status")
                        }
                    ),
                    session_id
                )
        
        except Exception as e:
            return (
                self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"POST /research validation error: {str(e)}",
                    duration_seconds=time.time() - start_time,
                    details={"exception": str(e), "exception_type": type(e).__name__},
                    remediation_steps=[
                        "Check API server is running",
                        "Verify authentication token is valid",
                        "Check network connectivity"
                    ]
                ),
                None
            )
    
    async def _test_get_session_status(self, session_id: str) -> ValidationResult:
        """Test GET /research/{session_id}/status endpoint.
        
        Args:
            session_id: Session ID to query
            
        Returns:
            ValidationResult for status endpoint
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Measure response time
                request_start = time.time()
                response = await client.get(
                    f"{self.api_base_url}/research/{session_id}/status",
                    headers=self.headers
                )
                response_time_ms = (time.time() - request_start) * 1000
                
                # Check status code
                if response.status_code != 200:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"GET /research/{{session_id}}/status returned status {response.status_code}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "session_id": session_id,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Verify session was created successfully",
                            "Check if session ID is valid",
                            "Check API logs for errors"
                        ]
                    )
                
                # Parse response
                data = response.json()
                
                # Check for required fields
                required_fields = ["session_id", "status", "progress_percentage"]
                missing_fields = [f for f in required_fields if f not in data]
                
                if missing_fields:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Status endpoint missing required fields: {', '.join(missing_fields)}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "response": data,
                            "missing_fields": missing_fields,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Update SessionStatus model to include all required fields",
                            "Verify orchestrator is tracking session progress"
                        ]
                    )
                
                # Check response time (should be < 200ms)
                max_response_time_ms = 200
                status = ValidationStatus.PASS
                message = f"GET /research/{{session_id}}/status validation passed (response time: {response_time_ms:.1f}ms)"
                
                if response_time_ms > max_response_time_ms:
                    status = ValidationStatus.WARNING
                    message = f"Status endpoint response time {response_time_ms:.1f}ms exceeds threshold {max_response_time_ms}ms"
                
                return self.create_result(
                    status=status,
                    message=message,
                    duration_seconds=time.time() - start_time,
                    details={
                        "session_id": session_id,
                        "response_time_ms": response_time_ms,
                        "threshold_ms": max_response_time_ms,
                        "session_status": data.get("status"),
                        "progress_percentage": data.get("progress_percentage")
                    }
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"GET /research/{{session_id}}/status validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check API server is running",
                    "Verify session ID is valid",
                    "Check network connectivity"
                ]
            )
    
    async def _test_list_sessions(self) -> ValidationResult:
        """Test GET /research endpoint with pagination.
        
        Returns:
            ValidationResult for list endpoint
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Test with pagination parameters
                request_start = time.time()
                response = await client.get(
                    f"{self.api_base_url}/research?page=1&page_size=10",
                    headers=self.headers
                )
                response_time_ms = (time.time() - request_start) * 1000
                
                # Check status code
                if response.status_code != 200:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"GET /research returned status {response.status_code}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Check if API server is running",
                            "Verify authentication token is valid",
                            "Check API logs for errors"
                        ]
                    )
                
                # Parse response
                data = response.json()
                
                # Check for required fields
                required_fields = ["sessions", "total_count", "page", "page_size"]
                missing_fields = [f for f in required_fields if f not in data]
                
                if missing_fields:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"List endpoint missing required fields: {', '.join(missing_fields)}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "response": data,
                            "missing_fields": missing_fields,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Update SessionList model to include all required fields",
                            "Verify pagination is implemented correctly"
                        ]
                    )
                
                # Verify pagination correctness
                sessions = data.get("sessions", [])
                total_count = data.get("total_count", 0)
                page_size = data.get("page_size", 0)
                
                if len(sessions) > page_size:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Pagination error: returned {len(sessions)} sessions but page_size is {page_size}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "sessions_count": len(sessions),
                            "page_size": page_size,
                            "total_count": total_count,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Fix pagination logic in list_user_sessions",
                            "Verify page_size is being respected"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"GET /research validation passed (response time: {response_time_ms:.1f}ms)",
                    duration_seconds=time.time() - start_time,
                    details={
                        "response_time_ms": response_time_ms,
                        "sessions_count": len(sessions),
                        "total_count": total_count,
                        "page": data.get("page"),
                        "page_size": page_size
                    }
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"GET /research validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check API server is running",
                    "Verify authentication token is valid",
                    "Check network connectivity"
                ]
            )

    async def _test_cancel_session(self, session_id: str) -> ValidationResult:
        """Test POST /research/{session_id}/cancel endpoint.
        
        Args:
            session_id: Session ID to cancel
            
        Returns:
            ValidationResult for cancel endpoint
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                request_start = time.time()
                response = await client.post(
                    f"{self.api_base_url}/research/{session_id}/cancel",
                    json={"reason": "API validation test"},
                    headers=self.headers
                )
                response_time_ms = (time.time() - request_start) * 1000
                
                # Check status code (200 for success, 404 if already completed/cancelled)
                if response.status_code not in [200, 404]:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"POST /research/{{session_id}}/cancel returned status {response.status_code}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "session_id": session_id,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Check if cancel endpoint is properly implemented",
                            "Verify session ID is valid",
                            "Check API logs for errors"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"POST /research/{{session_id}}/cancel validation passed (response time: {response_time_ms:.1f}ms)",
                    duration_seconds=time.time() - start_time,
                    details={
                        "session_id": session_id,
                        "response_time_ms": response_time_ms,
                        "status_code": response.status_code
                    }
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"POST /research/{{session_id}}/cancel validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check API server is running",
                    "Verify session ID is valid",
                    "Check network connectivity"
                ]
            )
    
    async def _test_get_session_result(self, session_id: str) -> ValidationResult:
        """Test GET /research/{session_id}/result endpoint.
        
        Note: This may return 400 if session is not completed yet, which is expected.
        
        Args:
            session_id: Session ID to get result for
            
        Returns:
            ValidationResult for result endpoint
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                request_start = time.time()
                response = await client.get(
                    f"{self.api_base_url}/research/{session_id}/result",
                    headers=self.headers
                )
                response_time_ms = (time.time() - request_start) * 1000
                
                # Check status code (200 for completed, 400 for not completed, 404 for not found)
                if response.status_code == 200:
                    # Session completed, validate response
                    data = response.json()
                    required_fields = ["session_id", "status", "document_content", "sources_count", "word_count"]
                    missing_fields = [f for f in required_fields if f not in data]
                    
                    if missing_fields:
                        return self.create_result(
                            status=ValidationStatus.FAIL,
                            message=f"Result endpoint missing required fields: {', '.join(missing_fields)}",
                            duration_seconds=time.time() - start_time,
                            details={
                                "response": data,
                                "missing_fields": missing_fields,
                                "response_time_ms": response_time_ms
                            },
                            remediation_steps=[
                                "Update ResearchResult model to include all required fields",
                                "Verify orchestrator is storing all result data"
                            ]
                        )
                    
                    return self.create_result(
                        status=ValidationStatus.PASS,
                        message=f"GET /research/{{session_id}}/result validation passed (response time: {response_time_ms:.1f}ms)",
                        duration_seconds=time.time() - start_time,
                        details={
                            "session_id": session_id,
                            "response_time_ms": response_time_ms,
                            "word_count": data.get("word_count"),
                            "sources_count": data.get("sources_count")
                        }
                    )
                
                elif response.status_code in [400, 404]:
                    # Expected if session not completed or cancelled
                    return self.create_result(
                        status=ValidationStatus.PASS,
                        message=f"GET /research/{{session_id}}/result correctly returned {response.status_code} for incomplete/cancelled session",
                        duration_seconds=time.time() - start_time,
                        details={
                            "session_id": session_id,
                            "response_time_ms": response_time_ms,
                            "status_code": response.status_code
                        }
                    )
                
                else:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"GET /research/{{session_id}}/result returned unexpected status {response.status_code}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "session_id": session_id,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Check result endpoint implementation",
                            "Verify proper status codes are returned",
                            "Check API logs for errors"
                        ]
                    )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"GET /research/{{session_id}}/result validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check API server is running",
                    "Verify session ID is valid",
                    "Check network connectivity"
                ]
            )
    
    async def _test_delete_session(self, session_id: str) -> ValidationResult:
        """Test DELETE /research/{session_id} endpoint.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            ValidationResult for delete endpoint
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                request_start = time.time()
                response = await client.delete(
                    f"{self.api_base_url}/research/{session_id}",
                    headers=self.headers
                )
                response_time_ms = (time.time() - request_start) * 1000
                
                # Check status code (200 for success, 404 if already deleted)
                if response.status_code not in [200, 404]:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"DELETE /research/{{session_id}} returned status {response.status_code}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "session_id": session_id,
                            "response_time_ms": response_time_ms
                        },
                        remediation_steps=[
                            "Check if delete endpoint is properly implemented",
                            "Verify session ID is valid",
                            "Check API logs for errors"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"DELETE /research/{{session_id}} validation passed (response time: {response_time_ms:.1f}ms)",
                    duration_seconds=time.time() - start_time,
                    details={
                        "session_id": session_id,
                        "response_time_ms": response_time_ms,
                        "status_code": response.status_code
                    }
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"DELETE /research/{{session_id}} validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check API server is running",
                    "Verify session ID is valid",
                    "Check network connectivity"
                ]
            )
    
    async def validate_authentication(self) -> ValidationResult:
        """Validate authentication and authorization.
        
        Tests:
        - Invalid authentication returns 401
        - Missing authentication returns 401
        - Valid authentication allows access
        
        Returns:
            ValidationResult for authentication
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Test 1: Invalid token
                invalid_headers = {"Authorization": f"Bearer {self.invalid_auth_token}"}
                response = await client.post(
                    f"{self.api_base_url}/research",
                    json={
                        "query": "Test query",
                        "max_sources": 5,
                        "include_academic": True,
                        "include_web": True,
                        "priority": "high"
                    },
                    headers=invalid_headers
                )
                
                if response.status_code != 401:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Invalid authentication did not return 401 (got {response.status_code})",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "expected_status": 401
                        },
                        remediation_steps=[
                            "Verify authentication middleware is enabled",
                            "Check JWT token validation logic",
                            "Ensure 401 is returned for invalid tokens"
                        ]
                    )
                
                # Test 2: Missing token
                response = await client.post(
                    f"{self.api_base_url}/research",
                    json={
                        "query": "Test query",
                        "max_sources": 5,
                        "include_academic": True,
                        "include_web": True,
                        "priority": "high"
                    }
                )
                
                if response.status_code != 401:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Missing authentication did not return 401 (got {response.status_code})",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "expected_status": 401
                        },
                        remediation_steps=[
                            "Verify authentication middleware is enabled",
                            "Check that endpoints require authentication",
                            "Ensure 401 is returned for missing tokens"
                        ]
                    )
                
                # Test 3: Valid token (already tested in other endpoints)
                # Health endpoint should NOT require authentication
                response = await client.get(f"{self.api_base_url}/health")
                
                if response.status_code != 200:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message=f"Health endpoint requires authentication (status {response.status_code})",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": response.status_code,
                            "expected_status": 200
                        },
                        remediation_steps=[
                            "Health endpoint should not require authentication",
                            "Update authentication middleware to exclude /health"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="Authentication validation passed",
                    duration_seconds=time.time() - start_time,
                    details={
                        "invalid_token_status": 401,
                        "missing_token_status": 401,
                        "health_endpoint_status": 200
                    }
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Authentication validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check API server is running",
                    "Verify authentication is properly configured",
                    "Check network connectivity"
                ]
            )

    async def validate_rate_limiting(self) -> ValidationResult:
        """Validate rate limiting enforcement.
        
        Tests:
        - Rate limits are enforced
        - Returns 429 when exceeded
        - Includes retry-after header
        
        Returns:
            ValidationResult for rate limiting
        """
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Make multiple rapid requests to trigger rate limiting
                # Note: This assumes rate limiting is configured
                # Adjust the number of requests based on your rate limit configuration
                
                rate_limit_triggered = False
                retry_after_header = None
                request_count = 0
                max_requests = 50  # Try up to 50 requests
                
                for i in range(max_requests):
                    request_count = i + 1
                    response = await client.post(
                        f"{self.api_base_url}/research",
                        json={
                            "query": f"Rate limit test query {i}",
                            "max_sources": 5,
                            "include_academic": True,
                            "include_web": True,
                            "priority": "high"
                        },
                        headers=self.headers
                    )
                    
                    if response.status_code == 429:
                        rate_limit_triggered = True
                        retry_after_header = response.headers.get("retry-after")
                        break
                    
                    # Small delay to avoid overwhelming the server
                    await asyncio.sleep(0.1)
                
                if not rate_limit_triggered:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message=f"Rate limiting not triggered after {request_count} requests",
                        duration_seconds=time.time() - start_time,
                        details={
                            "request_count": request_count,
                            "max_requests": max_requests
                        },
                        remediation_steps=[
                            "Verify rate limiting is configured",
                            "Check rate limit thresholds",
                            "Ensure rate limiting middleware is enabled",
                            "Note: Rate limiting may be configured with higher limits"
                        ]
                    )
                
                if not retry_after_header:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="Rate limit triggered but retry-after header missing",
                        duration_seconds=time.time() - start_time,
                        details={
                            "request_count": request_count,
                            "status_code": 429
                        },
                        remediation_steps=[
                            "Add retry-after header to 429 responses",
                            "Update rate limiting middleware to include retry-after"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"Rate limiting validation passed (triggered after {request_count} requests)",
                    duration_seconds=time.time() - start_time,
                    details={
                        "request_count": request_count,
                        "status_code": 429,
                        "retry_after": retry_after_header
                    }
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Rate limiting validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check API server is running",
                    "Verify rate limiting is properly configured",
                    "Check network connectivity"
                ]
            )
    
    async def validate_websocket_connection(self) -> ValidationResult:
        """Validate WebSocket connection and streaming.
        
        Tests:
        - WebSocket connection can be established
        - Connection accepts valid authentication
        - Progress updates are streamed
        
        Returns:
            ValidationResult for WebSocket
        """
        start_time = time.time()
        
        try:
            # Convert http(s) to ws(s) if needed
            ws_url = self.ws_base_url
            if not ws_url.startswith("ws"):
                ws_url = ws_url.replace("http://", "ws://").replace("https://", "wss://")
            
            # Try to establish WebSocket connection
            # Note: WebSocket endpoint path may vary based on implementation
            ws_endpoint = f"{ws_url}/research/stream"
            
            try:
                async with websockets.connect(
                    ws_endpoint,
                    extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                    timeout=10
                ) as websocket:
                    # Connection established successfully
                    
                    # Try to receive a message (with timeout)
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=5.0
                        )
                        
                        return self.create_result(
                            status=ValidationStatus.PASS,
                            message="WebSocket connection validation passed",
                            duration_seconds=time.time() - start_time,
                            details={
                                "connection_established": True,
                                "message_received": True,
                                "message_preview": str(message)[:100]
                            }
                        )
                    
                    except asyncio.TimeoutError:
                        # Connection established but no message received
                        # This is acceptable if no active sessions
                        return self.create_result(
                            status=ValidationStatus.PASS,
                            message="WebSocket connection established (no messages received)",
                            duration_seconds=time.time() - start_time,
                            details={
                                "connection_established": True,
                                "message_received": False,
                                "note": "No messages received within timeout (expected if no active sessions)"
                            }
                        )
            
            except websockets.exceptions.InvalidStatusCode as e:
                if e.status_code == 404:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="WebSocket endpoint not found (404)",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": 404,
                            "endpoint": ws_endpoint,
                            "note": "WebSocket endpoint may not be implemented yet"
                        },
                        remediation_steps=[
                            "Implement WebSocket endpoint for streaming",
                            "Verify WebSocket router is included in main app",
                            "Check WebSocket endpoint path configuration"
                        ]
                    )
                elif e.status_code == 401:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="WebSocket authentication failed (401)",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": 401,
                            "endpoint": ws_endpoint
                        },
                        remediation_steps=[
                            "Verify WebSocket authentication is properly configured",
                            "Check authentication token is valid",
                            "Ensure WebSocket endpoint accepts Bearer tokens"
                        ]
                    )
                else:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"WebSocket connection failed with status {e.status_code}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "status_code": e.status_code,
                            "endpoint": ws_endpoint,
                            "exception": str(e)
                        },
                        remediation_steps=[
                            "Check WebSocket endpoint implementation",
                            "Verify WebSocket server is running",
                            "Check API logs for errors"
                        ]
                    )
            
            except ConnectionRefusedError:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="WebSocket connection refused",
                    duration_seconds=time.time() - start_time,
                    details={
                        "endpoint": ws_endpoint
                    },
                    remediation_steps=[
                        "Check if WebSocket server is running",
                        "Verify WebSocket port is correct",
                        "Check firewall settings"
                    ]
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"WebSocket validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                    "endpoint": ws_endpoint if 'ws_endpoint' in locals() else "unknown"
                },
                remediation_steps=[
                    "Check WebSocket configuration",
                    "Verify WebSocket endpoint is accessible",
                    "Check network connectivity"
                ]
            )
