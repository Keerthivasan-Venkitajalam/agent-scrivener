"""Unit tests for DocumentationValidator."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
from hypothesis import given, strategies as st, settings

from agent_scrivener.deployment.validation.documentation_validator import DocumentationValidator
from agent_scrivener.deployment.validation.models import ValidationStatus


class TestDocumentationValidator:
    """Test suite for DocumentationValidator."""
    
    @pytest.fixture
    def temp_docs_dir(self):
        """Create a temporary docs directory for testing."""
        temp_dir = tempfile.mkdtemp()
        docs_path = Path(temp_dir) / "docs"
        docs_path.mkdir()
        yield docs_path
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def validator(self, temp_docs_dir):
        """Create a DocumentationValidator instance with temp directory."""
        return DocumentationValidator(docs_dir=temp_docs_dir)
    
    @pytest.mark.asyncio
    async def test_validate_api_documentation_missing(self, validator):
        """Test validation fails when API documentation is missing."""
        result = await validator.validate_api_documentation()
        
        assert result.status == ValidationStatus.FAIL
        assert "not found" in result.message.lower()
        assert result.remediation_steps is not None
        assert len(result.remediation_steps) > 0
    
    @pytest.mark.asyncio
    async def test_validate_api_documentation_complete(self, validator, temp_docs_dir):
        """Test validation passes when API documentation is complete."""
        api_doc = temp_docs_dir / "api_reference.md"
        api_doc.write_text("""
# API Reference

## Authentication
All endpoints require JWT authentication.

## Authorization
Users can only access their own resources.

## Endpoints

### Health Check
GET /health

### Research Endpoints
POST /research - Create a new research session
GET /research/{id} - Get research session status
        """)
        
        result = await validator.validate_api_documentation()
        
        assert result.status == ValidationStatus.PASS
        assert "complete" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_api_documentation_incomplete(self, validator, temp_docs_dir):
        """Test validation warns when API documentation is incomplete."""
        api_doc = temp_docs_dir / "api_reference.md"
        api_doc.write_text("""
# API Reference

Some basic documentation without required sections.
        """)
        
        result = await validator.validate_api_documentation()
        
        assert result.status == ValidationStatus.WARNING
        assert "incomplete" in result.message.lower()
        assert "missing_sections" in result.details or "missing_endpoints" in result.details
    
    @pytest.mark.asyncio
    async def test_validate_deployment_documentation_missing(self, temp_docs_dir):
        """Test validation fails when deployment documentation is missing."""
        # Use a completely isolated directory that doesn't have README.md
        isolated_dir = temp_docs_dir / "isolated"
        isolated_dir.mkdir()
        validator = DocumentationValidator(docs_dir=isolated_dir)
        
        result = await validator.validate_deployment_documentation()
        
        assert result.status == ValidationStatus.FAIL
        assert "not found" in result.message.lower()
        assert result.remediation_steps is not None
    
    @pytest.mark.asyncio
    async def test_validate_deployment_documentation_complete(self, validator, temp_docs_dir):
        """Test validation passes when deployment documentation is complete."""
        deploy_doc = temp_docs_dir / "deployment_guide.md"
        deploy_doc.write_text("""
# Deployment Guide

## Docker Deployment
Build the Docker image:
```bash
docker build -t agent-scrivener .
```

## AWS Deployment
Deploy using CDK:
```bash
cdk deploy
```

## Environment Variables
Configure the following environment variables:
- AWS_REGION
- DATABASE_URL
        """)
        
        result = await validator.validate_deployment_documentation()
        
        assert result.status == ValidationStatus.PASS
        assert "complete" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_deployment_documentation_incomplete(self, validator, temp_docs_dir):
        """Test validation warns when deployment documentation is incomplete."""
        deploy_doc = temp_docs_dir / "deployment_guide.md"
        deploy_doc.write_text("""
# Deployment Guide

Some basic deployment info without Docker or AWS details.
        """)
        
        result = await validator.validate_deployment_documentation()
        
        assert result.status == ValidationStatus.WARNING
        assert "incomplete" in result.message.lower()
        assert "missing_topics" in result.details
    
    @pytest.mark.asyncio
    async def test_validate_user_documentation_missing(self, validator):
        """Test validation fails when user documentation is missing."""
        result = await validator.validate_user_documentation()
        
        assert result.status == ValidationStatus.FAIL
        assert "not found" in result.message.lower()
        assert result.remediation_steps is not None
    
    @pytest.mark.asyncio
    async def test_validate_user_documentation_complete(self, validator, temp_docs_dir):
        """Test validation passes when user documentation is complete."""
        user_doc = temp_docs_dir / "user_guide.md"
        user_doc.write_text("""
# User Guide

## Getting Started
Learn how to use Agent Scrivener.

## Example Queries
Here are some example research queries:
- "What are the latest developments in quantum computing?"
- "Analyze the impact of climate change on agriculture"

## Expected Output
The system will generate a comprehensive research document with:
- Introduction
- Analysis
- Synthesis
- Conclusion

## Usage
Submit queries via the API and check results.
        """)
        
        result = await validator.validate_user_documentation()
        
        assert result.status == ValidationStatus.PASS
        assert "complete" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_user_documentation_incomplete(self, validator, temp_docs_dir):
        """Test validation warns when user documentation is incomplete."""
        user_doc = temp_docs_dir / "user_guide.md"
        user_doc.write_text("""
# User Guide

Basic user information without examples or usage details.
        """)
        
        result = await validator.validate_user_documentation()
        
        assert result.status == ValidationStatus.WARNING
        assert "incomplete" in result.message.lower()
        assert "missing_sections" in result.details
    
    @pytest.mark.asyncio
    async def test_validate_architecture_documentation_missing(self, temp_docs_dir):
        """Test validation fails when architecture documentation is missing."""
        # Use a completely isolated directory that doesn't have README.md
        isolated_dir = temp_docs_dir / "isolated"
        isolated_dir.mkdir()
        validator = DocumentationValidator(docs_dir=isolated_dir)
        
        result = await validator.validate_architecture_documentation()
        
        assert result.status == ValidationStatus.FAIL
        assert "not found" in result.message.lower()
        assert result.remediation_steps is not None
    
    @pytest.mark.asyncio
    async def test_validate_architecture_documentation_complete(self, validator, temp_docs_dir):
        """Test validation passes when architecture documentation is complete."""
        arch_doc = temp_docs_dir / "architecture.md"
        arch_doc.write_text("""
# System Architecture

## Components
The system consists of the following components:
- API Gateway
- Orchestrator
- Research Agent
- Analysis Agent
- Database

## Architecture Overview
The system follows a microservices architecture with agent-based design.
        """)
        
        result = await validator.validate_architecture_documentation()
        
        assert result.status == ValidationStatus.PASS
        assert "complete" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_architecture_documentation_incomplete(self, validator, temp_docs_dir):
        """Test validation warns when architecture documentation is incomplete."""
        arch_doc = temp_docs_dir / "architecture.md"
        arch_doc.write_text("""
# Architecture

Some basic info without required keywords.
        """)
        
        result = await validator.validate_architecture_documentation()
        
        assert result.status == ValidationStatus.WARNING
        assert "incomplete" in result.message.lower()
        assert "missing_topics" in result.details
    
    @pytest.mark.asyncio
    async def test_validate_troubleshooting_documentation_missing(self, validator):
        """Test validation fails when troubleshooting documentation is missing."""
        result = await validator.validate_troubleshooting_documentation()
        
        assert result.status == ValidationStatus.FAIL
        assert "not found" in result.message.lower()
        assert result.remediation_steps is not None
    
    @pytest.mark.asyncio
    async def test_validate_troubleshooting_documentation_complete(self, validator, temp_docs_dir):
        """Test validation passes when troubleshooting documentation is complete."""
        troubleshoot_doc = temp_docs_dir / "troubleshooting.md"
        troubleshoot_doc.write_text("""
# Troubleshooting Guide

## Common Errors

### Authentication Error
**Problem:** 401 Unauthorized error when calling API
**Solution:** Ensure you have a valid JWT token

### Connection Error
**Problem:** Cannot connect to database
**Solution:** Check DATABASE_URL environment variable and debug connection settings
        """)
        
        result = await validator.validate_troubleshooting_documentation()
        
        assert result.status == ValidationStatus.PASS
        assert "complete" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_troubleshooting_documentation_incomplete(self, validator, temp_docs_dir):
        """Test validation warns when troubleshooting documentation is incomplete."""
        troubleshoot_doc = temp_docs_dir / "troubleshooting.md"
        troubleshoot_doc.write_text("""
# Troubleshooting

Some basic info without required keywords.
        """)
        
        result = await validator.validate_troubleshooting_documentation()
        
        assert result.status == ValidationStatus.WARNING
        assert "incomplete" in result.message.lower()
        assert "missing_topics" in result.details
    
    @pytest.mark.asyncio
    async def test_validate_api_examples_missing_doc(self, validator):
        """Test validation fails when API documentation is missing (cannot check examples)."""
        result = await validator.validate_api_examples()
        
        assert result.status == ValidationStatus.FAIL
        assert "not found" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_api_examples_no_code_blocks(self, validator, temp_docs_dir):
        """Test validation warns when API documentation has no code examples."""
        api_doc = temp_docs_dir / "api_reference.md"
        api_doc.write_text("""
# API Reference

Documentation without any code examples.
        """)
        
        result = await validator.validate_api_examples()
        
        assert result.status == ValidationStatus.WARNING
        assert "lacks code examples" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_api_examples_complete(self, validator, temp_docs_dir):
        """Test validation passes when API documentation has comprehensive examples."""
        api_doc = temp_docs_dir / "api_reference.md"
        api_doc.write_text("""
# API Reference

## Example Request
```bash
curl -X POST http://localhost:8000/research \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is quantum computing?"}'
```

## Example Response
```json
{
  "session_id": "123",
  "status": "pending",
  "code": 201
}
```

## Error Response
```json
{
  "error": "Unauthorized",
  "code": 401
}
```
        """)
        
        result = await validator.validate_api_examples()
        
        assert result.status == ValidationStatus.PASS
        assert "comprehensive examples" in result.message.lower()
        assert result.details["code_blocks_found"] == 3
    
    @pytest.mark.asyncio
    async def test_validate_api_examples_incomplete(self, validator, temp_docs_dir):
        """Test validation warns when API examples are incomplete."""
        api_doc = temp_docs_dir / "api_reference.md"
        api_doc.write_text("""
# API Reference

## Example
```json
{
  "query": "test"
}
```
        """)
        
        result = await validator.validate_api_examples()
        
        assert result.status == ValidationStatus.WARNING
        assert "incomplete" in result.message.lower()
        assert "missing_example_types" in result.details
    
    @pytest.mark.asyncio
    async def test_validate_all_checks(self, validator, temp_docs_dir):
        """Test running all validation checks together."""
        # Create minimal documentation
        (temp_docs_dir / "api_reference.md").write_text("# API\nAuthentication\nEndpoints\n/health\nPOST\nGET")
        (temp_docs_dir / "deployment_guide.md").write_text("# Deploy\nDocker\nAWS\nEnvironment")
        (temp_docs_dir / "user_guide.md").write_text("# User Guide\nExample\nUsage\nOutput")
        (temp_docs_dir / "architecture.md").write_text("# Architecture\nComponents\nArchitecture")
        (temp_docs_dir / "troubleshooting.md").write_text("# Troubleshooting\nError\nSolution")
        
        results = await validator.validate()
        
        assert len(results) == 6  # All 6 validation methods
        assert all(isinstance(r.status, ValidationStatus) for r in results)
    
    @pytest.mark.asyncio
    async def test_validate_checks_readme_fallback(self, validator, temp_docs_dir):
        """Test that validator checks README.md as fallback for deployment/architecture docs."""
        # Create README with deployment and architecture info
        readme_path = temp_docs_dir.parent / "README.md"
        readme_path.write_text("""
# Agent Scrivener

## Architecture
System components and design.

## Deployment
Deploy with Docker and AWS CDK.
Configure environment variables.
        """)
        
        # Should find deployment info in README
        deploy_result = await validator.validate_deployment_documentation()
        assert deploy_result.status in (ValidationStatus.PASS, ValidationStatus.WARNING)
        
        # Should find architecture info in README
        arch_result = await validator.validate_architecture_documentation()
        assert arch_result.status in (ValidationStatus.PASS, ValidationStatus.WARNING)
    
    def test_validator_initialization(self):
        """Test validator initializes with correct defaults."""
        validator = DocumentationValidator()
        assert validator.name == "DocumentationValidator"
        assert validator.docs_dir == Path("docs")
    
    def test_validator_custom_docs_dir(self, temp_docs_dir):
        """Test validator accepts custom docs directory."""
        validator = DocumentationValidator(docs_dir=temp_docs_dir)
        assert validator.docs_dir == temp_docs_dir


class TestDocumentationValidatorPropertyBased:
    """Property-based tests for DocumentationValidator."""
    
    @given(
        # Generate API endpoints with various HTTP methods
        endpoints=st.lists(
            st.fixed_dictionaries({
                'method': st.sampled_from(['GET', 'POST', 'PUT', 'DELETE', 'PATCH']),
                'path': st.from_regex(r'/[a-z_]+(/\{[a-z_]+\})?', fullmatch=True),
                'has_request_example': st.booleans(),
                'has_response_example': st.booleans()
            }),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_property_api_documentation_completeness(self, endpoints):
        """Property 23: API documentation completeness
        
        **Validates: Requirements 6.6**
        
        For any documented API endpoint, the documentation should include 
        request/response examples.
        
        This property verifies that:
        1. If an endpoint is documented, it must have both request and response examples
        2. Documentation without examples should be flagged as incomplete
        3. Complete documentation (with examples) should pass validation
        """
        # Create temporary directory for this test run
        temp_dir = tempfile.mkdtemp()
        try:
            temp_docs_dir = Path(temp_dir) / "docs"
            temp_docs_dir.mkdir()
            
            validator = DocumentationValidator(docs_dir=temp_docs_dir)
            api_doc_path = temp_docs_dir / "api_reference.md"
            
            # Build API documentation content
            doc_content = "# API Reference\n\n"
            doc_content += "## Authentication\nAll endpoints require JWT authentication.\n\n"
            doc_content += "## Endpoints\n\n"
            
            # Track whether all endpoints have complete examples
            all_have_request_examples = True
            all_have_response_examples = True
            
            for endpoint in endpoints:
                method = endpoint['method']
                path = endpoint['path']
                has_request = endpoint['has_request_example']
                has_response = endpoint['has_response_example']
            
                # Document the endpoint
                doc_content += f"### {method} {path}\n\n"
                doc_content += f"Description of {method} {path} endpoint.\n\n"
                
                # Add request example if specified
                if has_request:
                    doc_content += "**Request Example:**\n\n"
                    doc_content += "```bash\n"
                    doc_content += f"curl -X {method} http://localhost:8000{path}\n"
                    doc_content += "```\n\n"
                else:
                    all_have_request_examples = False
                
                # Add response example if specified
                if has_response:
                    doc_content += "**Response Example:**\n\n"
                    doc_content += "```json\n"
                    doc_content += '{"status": "success", "code": 200}\n'
                    doc_content += "```\n\n"
                else:
                    all_have_response_examples = False
            
            # Write the documentation
            api_doc_path.write_text(doc_content)
            
            # Validate the documentation
            result = await validator.validate_api_examples()
            
            # Property: For any documented API endpoint, the documentation should 
            # include request/response examples
            if all_have_request_examples and all_have_response_examples:
                # All endpoints have complete examples - should PASS
                assert result.status == ValidationStatus.PASS, \
                    f"Expected PASS when all endpoints have complete examples, got {result.status}"
                assert "comprehensive" in result.message.lower() or "complete" in result.message.lower(), \
                    "Success message should indicate comprehensive/complete examples"
            else:
                # Some endpoints lack examples - should WARN
                assert result.status == ValidationStatus.WARNING, \
                    f"Expected WARNING when endpoints lack examples, got {result.status}"
                assert "incomplete" in result.message.lower() or "lacks" in result.message.lower(), \
                    "Warning message should indicate incomplete or lacking examples"
                
                # Check that missing example types are identified
                if not all_have_request_examples:
                    assert ("endpoints_missing_examples" in result.details or 
                            "missing_example_types" in result.details or 
                            "lacks" in result.message.lower()), \
                        "Should identify missing request examples"
                
                if not all_have_response_examples:
                    assert ("endpoints_missing_examples" in result.details or 
                            "missing_example_types" in result.details or 
                            "lacks" in result.message.lower()), \
                        "Should identify missing response examples"
                
                # Should provide remediation steps
                assert result.remediation_steps is not None and len(result.remediation_steps) > 0, \
                    "Should provide remediation steps for incomplete documentation"
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
