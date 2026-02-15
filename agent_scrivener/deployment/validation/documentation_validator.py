"""Documentation completeness validator for production readiness.

This validator checks that all required documentation exists and includes
necessary sections and examples for production operations.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Set

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus


logger = logging.getLogger(__name__)


class DocumentationValidator(BaseValidator):
    """Validates completeness of documentation for production deployment.
    
    Checks for:
    - API reference documentation with all endpoints
    - Deployment guide with Docker and AWS instructions
    - User guide with example queries
    - Architecture documentation with component diagrams
    - Troubleshooting guide with common error scenarios
    - API examples with request/response samples
    """
    
    def __init__(self, docs_dir: Optional[Path] = None):
        """Initialize the documentation validator.
        
        Args:
            docs_dir: Path to documentation directory (defaults to ./docs)
        """
        super().__init__(name="DocumentationValidator")
        self.docs_dir = docs_dir or Path("docs")
        
    async def validate(self) -> List[ValidationResult]:
        """Execute all documentation validation checks.
        
        Returns:
            List of validation results
        """
        self.log_validation_start()
        
        results = []
        
        # Validate each documentation type
        results.append(await self.validate_api_documentation())
        results.append(await self.validate_deployment_documentation())
        results.append(await self.validate_user_documentation())
        results.append(await self.validate_architecture_documentation())
        results.append(await self.validate_troubleshooting_documentation())
        results.append(await self.validate_api_examples())
        
        self.log_validation_complete(results)
        return results
    
    async def validate_api_documentation(self) -> ValidationResult:
        """Validate API reference documentation exists and includes all endpoints.
        
        Checks for:
        - API reference file exists
        - Documents all REST endpoints
        - Documents WebSocket endpoints
        - Includes authentication/authorization documentation
        
        Returns:
            ValidationResult for API documentation
        """
        api_doc_path = self.docs_dir / "api_reference.md"
        
        if not api_doc_path.exists():
            return self.create_result(
                status=ValidationStatus.FAIL,
                message="API reference documentation not found",
                details={"expected_path": str(api_doc_path)},
                remediation_steps=[
                    f"Create API reference documentation at {api_doc_path}",
                    "Document all REST endpoints (POST /research, GET /research/{id}, etc.)",
                    "Document WebSocket endpoints and message formats",
                    "Include authentication and authorization requirements",
                    "Add request/response examples for each endpoint"
                ]
            )
        
        # Read the API documentation
        content = api_doc_path.read_text()
        
        # Check for required sections
        missing_sections = []
        required_sections = {
            "authentication": ["authentication", "auth"],
            "authorization": ["authorization", "access control"],
            "endpoints": ["endpoints", "api endpoints"],
        }
        
        for section_name, keywords in required_sections.items():
            if not any(keyword.lower() in content.lower() for keyword in keywords):
                missing_sections.append(section_name)
        
        # Check for common REST endpoints
        required_endpoints = [
            "/health",
            "/research",
            "POST",
            "GET"
        ]
        
        missing_endpoints = []
        for endpoint in required_endpoints:
            if endpoint not in content:
                missing_endpoints.append(endpoint)
        
        if missing_sections or missing_endpoints:
            details = {}
            if missing_sections:
                details["missing_sections"] = missing_sections
            if missing_endpoints:
                details["missing_endpoints"] = missing_endpoints
            
            remediation = []
            if missing_sections:
                remediation.append(f"Add missing sections: {', '.join(missing_sections)}")
            if missing_endpoints:
                remediation.append(f"Document missing endpoints: {', '.join(missing_endpoints)}")
            
            return self.create_result(
                status=ValidationStatus.WARNING,
                message="API documentation is incomplete",
                details=details,
                remediation_steps=remediation
            )
        
        return self.create_result(
            status=ValidationStatus.PASS,
            message="API documentation is complete",
            details={"path": str(api_doc_path)}
        )
    
    async def validate_deployment_documentation(self) -> ValidationResult:
        """Validate deployment guide exists and includes Docker and AWS instructions.
        
        Checks for:
        - Deployment guide file exists
        - Includes Docker deployment instructions
        - Includes AWS deployment instructions
        - Includes environment variable configuration
        
        Returns:
            ValidationResult for deployment documentation
        """
        # Check for deployment documentation in various possible locations
        possible_paths = [
            self.docs_dir / "deployment_guide.md",
            self.docs_dir / "deployment.md",
            self.docs_dir.parent / "README.md" if self.docs_dir.name == "docs" else None
        ]
        
        # Filter out None values
        possible_paths = [p for p in possible_paths if p is not None]
        
        deployment_doc_path = None
        for path in possible_paths:
            if path.exists():
                deployment_doc_path = path
                break
        
        if not deployment_doc_path:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message="Deployment guide documentation not found",
                details={"checked_paths": [str(p) for p in possible_paths]},
                remediation_steps=[
                    f"Create deployment guide at {self.docs_dir / 'deployment_guide.md'}",
                    "Include Docker deployment instructions (docker build, docker run)",
                    "Include Docker Compose instructions",
                    "Include AWS deployment instructions (CDK deploy, CloudFormation)",
                    "Document required environment variables",
                    "Include troubleshooting section for common deployment issues"
                ]
            )
        
        # Read the deployment documentation
        content = deployment_doc_path.read_text()
        
        # Check for required deployment topics
        missing_topics = []
        required_topics = {
            "docker": ["docker", "dockerfile", "container"],
            "aws": ["aws", "cloud", "cdk", "cloudformation"],
            "environment": ["environment", "env", "configuration"]
        }
        
        for topic_name, keywords in required_topics.items():
            if not any(keyword.lower() in content.lower() for keyword in keywords):
                missing_topics.append(topic_name)
        
        if missing_topics:
            return self.create_result(
                status=ValidationStatus.WARNING,
                message="Deployment documentation is incomplete",
                details={
                    "path": str(deployment_doc_path),
                    "missing_topics": missing_topics
                },
                remediation_steps=[
                    f"Add missing deployment topics: {', '.join(missing_topics)}",
                    "Ensure Docker deployment steps are clearly documented",
                    "Ensure AWS deployment steps are clearly documented",
                    "Document all required environment variables"
                ]
            )
        
        return self.create_result(
            status=ValidationStatus.PASS,
            message="Deployment documentation is complete",
            details={"path": str(deployment_doc_path)}
        )
    
    async def validate_user_documentation(self) -> ValidationResult:
        """Validate user guide exists and includes example queries and expected outputs.
        
        Checks for:
        - User guide file exists
        - Includes example queries
        - Includes expected outputs or results
        - Includes usage instructions
        
        Returns:
            ValidationResult for user documentation
        """
        user_doc_path = self.docs_dir / "user_guide.md"
        
        if not user_doc_path.exists():
            return self.create_result(
                status=ValidationStatus.FAIL,
                message="User guide documentation not found",
                details={"expected_path": str(user_doc_path)},
                remediation_steps=[
                    f"Create user guide at {user_doc_path}",
                    "Include example research queries",
                    "Show expected outputs and document structure",
                    "Document how to submit queries via API",
                    "Document how to check query status and retrieve results",
                    "Include common use cases and workflows"
                ]
            )
        
        # Read the user documentation
        content = user_doc_path.read_text()
        
        # Check for required user guide sections
        missing_sections = []
        required_sections = {
            "examples": ["example", "sample", "query"],
            "usage": ["usage", "how to", "getting started"],
            "output": ["output", "result", "response"]
        }
        
        for section_name, keywords in required_sections.items():
            if not any(keyword.lower() in content.lower() for keyword in keywords):
                missing_sections.append(section_name)
        
        if missing_sections:
            return self.create_result(
                status=ValidationStatus.WARNING,
                message="User guide is incomplete",
                details={
                    "path": str(user_doc_path),
                    "missing_sections": missing_sections
                },
                remediation_steps=[
                    f"Add missing sections: {', '.join(missing_sections)}",
                    "Include concrete example queries users can try",
                    "Show what the expected output looks like",
                    "Document the complete user workflow from query to result"
                ]
            )
        
        return self.create_result(
            status=ValidationStatus.PASS,
            message="User guide is complete",
            details={"path": str(user_doc_path)}
        )
    
    async def validate_architecture_documentation(self) -> ValidationResult:
        """Validate architecture documentation exists and includes component diagrams.
        
        Checks for:
        - Architecture documentation file exists
        - Includes component descriptions
        - Includes system architecture information
        
        Returns:
            ValidationResult for architecture documentation
        """
        # Check for architecture documentation in various possible locations
        possible_paths = [
            self.docs_dir / "architecture.md",
            self.docs_dir / "design.md",
            self.docs_dir / "system_design.md",
            self.docs_dir.parent / "README.md" if self.docs_dir.name == "docs" else None
        ]
        
        # Filter out None values
        possible_paths = [p for p in possible_paths if p is not None]
        
        arch_doc_path = None
        for path in possible_paths:
            if path.exists():
                content = path.read_text()
                # Check if this file contains architecture information
                if any(keyword in content.lower() for keyword in ["architecture", "component", "system design", "diagram"]):
                    arch_doc_path = path
                    break
        
        if not arch_doc_path:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message="Architecture documentation not found",
                details={"checked_paths": [str(p) for p in possible_paths]},
                remediation_steps=[
                    f"Create architecture documentation at {self.docs_dir / 'architecture.md'}",
                    "Document system components (API, Orchestrator, Agents, Database)",
                    "Include component interaction diagrams",
                    "Document data flow through the system",
                    "Describe agent workflow and coordination",
                    "Include deployment architecture (AWS services, networking)"
                ]
            )
        
        # Read the architecture documentation
        content = arch_doc_path.read_text()
        
        # Check for required architecture topics
        missing_topics = []
        required_topics = {
            "components": ["component", "module", "service"],
            "architecture": ["architecture", "design", "structure"]
        }
        
        for topic_name, keywords in required_topics.items():
            if not any(keyword.lower() in content.lower() for keyword in keywords):
                missing_topics.append(topic_name)
        
        if missing_topics:
            return self.create_result(
                status=ValidationStatus.WARNING,
                message="Architecture documentation is incomplete",
                details={
                    "path": str(arch_doc_path),
                    "missing_topics": missing_topics
                },
                remediation_steps=[
                    f"Add missing architecture topics: {', '.join(missing_topics)}",
                    "Document all major system components",
                    "Include diagrams showing component relationships",
                    "Describe the overall system architecture"
                ]
            )
        
        return self.create_result(
            status=ValidationStatus.PASS,
            message="Architecture documentation is complete",
            details={"path": str(arch_doc_path)}
        )
    
    async def validate_troubleshooting_documentation(self) -> ValidationResult:
        """Validate troubleshooting guide exists and includes common error scenarios.
        
        Checks for:
        - Troubleshooting guide file exists
        - Includes common error scenarios
        - Includes solutions or debugging steps
        
        Returns:
            ValidationResult for troubleshooting documentation
        """
        # Check for troubleshooting documentation in various possible locations
        possible_paths = [
            self.docs_dir / "troubleshooting.md",
            self.docs_dir / "troubleshooting_guide.md",
            self.docs_dir / "faq.md",
            self.docs_dir / "deployment_guide.md"
        ]
        
        troubleshooting_doc_path = None
        for path in possible_paths:
            if path.exists():
                content = path.read_text()
                # Check if this file contains troubleshooting information
                if any(keyword in content.lower() for keyword in ["troubleshoot", "error", "problem", "issue", "debug", "faq"]):
                    troubleshooting_doc_path = path
                    break
        
        if not troubleshooting_doc_path:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message="Troubleshooting guide not found",
                details={"checked_paths": [str(p) for p in possible_paths]},
                remediation_steps=[
                    f"Create troubleshooting guide at {self.docs_dir / 'troubleshooting.md'}",
                    "Document common error scenarios and their solutions",
                    "Include debugging steps for deployment issues",
                    "Document how to check logs and diagnose problems",
                    "Include solutions for authentication/authorization errors",
                    "Document performance troubleshooting steps",
                    "Include AWS-specific troubleshooting (IAM, networking, etc.)"
                ]
            )
        
        # Read the troubleshooting documentation
        content = troubleshooting_doc_path.read_text()
        
        # Check for required troubleshooting topics
        missing_topics = []
        required_topics = {
            "errors": ["error", "exception", "failure"],
            "solutions": ["solution", "fix", "resolve", "debug"]
        }
        
        for topic_name, keywords in required_topics.items():
            if not any(keyword.lower() in content.lower() for keyword in keywords):
                missing_topics.append(topic_name)
        
        if missing_topics:
            return self.create_result(
                status=ValidationStatus.WARNING,
                message="Troubleshooting documentation is incomplete",
                details={
                    "path": str(troubleshooting_doc_path),
                    "missing_topics": missing_topics
                },
                remediation_steps=[
                    f"Add missing troubleshooting topics: {', '.join(missing_topics)}",
                    "Document common error scenarios",
                    "Provide clear solutions and debugging steps",
                    "Include examples of error messages and their meanings"
                ]
            )
        
        return self.create_result(
            status=ValidationStatus.PASS,
            message="Troubleshooting documentation is complete",
            details={"path": str(troubleshooting_doc_path)}
        )
    
    async def validate_api_examples(self) -> ValidationResult:
        """Validate API documentation includes request/response examples.
        
        Checks for:
        - API documentation includes code examples
        - Examples show request format
        - Examples show response format
        
        Returns:
            ValidationResult for API examples
        """
        api_doc_path = self.docs_dir / "api_reference.md"
        
        if not api_doc_path.exists():
            return self.create_result(
                status=ValidationStatus.FAIL,
                message="API reference documentation not found (cannot validate examples)",
                details={"expected_path": str(api_doc_path)},
                remediation_steps=[
                    f"Create API reference documentation at {api_doc_path}",
                    "Include request/response examples for all endpoints"
                ]
            )
        
        # Read the API documentation
        content = api_doc_path.read_text()
        
        # Check for code blocks (examples are typically in code blocks)
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        
        if not code_blocks:
            return self.create_result(
                status=ValidationStatus.WARNING,
                message="API documentation lacks code examples",
                details={"path": str(api_doc_path)},
                remediation_steps=[
                    "Add code examples for API requests",
                    "Include example request bodies (JSON format)",
                    "Include example response bodies",
                    "Show examples for both successful and error responses",
                    "Include curl or HTTP client examples"
                ]
            )
        
        # Split content into sections by endpoint headers (### METHOD /path)
        # This regex matches headers like "### GET /endpoint" or "### POST /resource/{id}"
        endpoint_pattern = r'###\s+(GET|POST|PUT|DELETE|PATCH)\s+(/[^\n]*)'
        endpoint_sections = re.split(endpoint_pattern, content)
        
        # If we found endpoint sections, validate each one
        if len(endpoint_sections) > 1:
            # endpoint_sections will be: [before_first, method1, path1, content1, method2, path2, content2, ...]
            # We need to check each content section for request and response examples
            endpoints_checked = 0
            endpoints_missing_examples = []
            
            # Process in groups of 3: method, path, content
            for i in range(1, len(endpoint_sections), 3):
                if i + 2 < len(endpoint_sections):
                    method = endpoint_sections[i]
                    path = endpoint_sections[i + 1]
                    section_content = endpoint_sections[i + 2]
                    endpoints_checked += 1
                    
                    # Check if this section has request and response examples
                    has_request = any(
                        keyword in section_content.lower() 
                        for keyword in ["request", "curl", "http"]
                    ) and "```" in section_content
                    
                    has_response = any(
                        keyword in section_content.lower() 
                        for keyword in ["response", "200", "201", "400", "401", "404"]
                    ) and "```" in section_content
                    
                    if not (has_request and has_response):
                        endpoints_missing_examples.append(f"{method} {path}")
            
            # If we checked endpoints and some are missing examples
            if endpoints_checked > 0 and endpoints_missing_examples:
                return self.create_result(
                    status=ValidationStatus.WARNING,
                    message="API documentation examples are incomplete",
                    details={
                        "path": str(api_doc_path),
                        "endpoints_checked": endpoints_checked,
                        "endpoints_missing_examples": endpoints_missing_examples,
                        "code_blocks_found": len(code_blocks)
                    },
                    remediation_steps=[
                        f"Add request and response examples for: {', '.join(endpoints_missing_examples)}",
                        "Ensure each endpoint has both request and response examples",
                        "Include HTTP status codes in response examples"
                    ]
                )
            
            # If we checked endpoints and all have examples
            if endpoints_checked > 0:
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="API documentation includes comprehensive examples",
                    details={
                        "path": str(api_doc_path),
                        "endpoints_checked": endpoints_checked,
                        "code_blocks_found": len(code_blocks)
                    }
                )
        
        # Fallback: If we couldn't parse endpoints, use the old logic
        # Check for request/response keywords in examples
        has_request_examples = any(
            keyword in content.lower() 
            for keyword in ["request", "post", "get", "curl", "http"]
        )
        
        has_response_examples = any(
            keyword in content.lower() 
            for keyword in ["response", "200", "201", "400", "401", "404"]
        )
        
        missing_example_types = []
        if not has_request_examples:
            missing_example_types.append("request examples")
        if not has_response_examples:
            missing_example_types.append("response examples")
        
        if missing_example_types:
            return self.create_result(
                status=ValidationStatus.WARNING,
                message="API documentation examples are incomplete",
                details={
                    "path": str(api_doc_path),
                    "missing_example_types": missing_example_types,
                    "code_blocks_found": len(code_blocks)
                },
                remediation_steps=[
                    f"Add missing example types: {', '.join(missing_example_types)}",
                    "Ensure each endpoint has both request and response examples",
                    "Include HTTP status codes in response examples"
                ]
            )
        
        return self.create_result(
            status=ValidationStatus.PASS,
            message="API documentation includes comprehensive examples",
            details={
                "path": str(api_doc_path),
                "code_blocks_found": len(code_blocks)
            }
        )
