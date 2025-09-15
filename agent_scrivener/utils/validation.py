"""
Validation utilities for Agent Scrivener.
"""

from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, ValidationError
import re
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        bool: True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_email(email: str) -> bool:
    """
    Validate if a string is a valid email address.
    
    Args:
        email: Email string to validate
        
    Returns:
        bool: True if valid email, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_doi(doi: str) -> bool:
    """
    Validate if a string is a valid DOI.
    
    Args:
        doi: DOI string to validate
        
    Returns:
        bool: True if valid DOI, False otherwise
    """
    pattern = r'^10\.\d{4,}/.+'
    return re.match(pattern, doi) is not None


def validate_model_data(data: Dict[str, Any], model_class: Type[BaseModel]) -> tuple[bool, Optional[str]]:
    """
    Validate data against a Pydantic model.
    
    Args:
        data: Data dictionary to validate
        model_class: Pydantic model class to validate against
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        model_class(**data)
        return True, None
    except ValidationError as e:
        return False, str(e)


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input by removing potentially harmful content.
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if necessary
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip() + "..."
    
    return text


def validate_research_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate a research query according to system requirements.
    
    Args:
        query: Research query to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    query = query.strip()
    
    if len(query) < 10:
        return False, "Query must be at least 10 characters long"
    
    if len(query) > 1000:
        return False, "Query cannot exceed 1000 characters"
    
    # Check for potentially harmful content
    harmful_patterns = [
        r'<script',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Query contains potentially harmful content"
    
    return True, None


class ValidationResult(BaseModel):
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


def validate_agent_input(data: Dict[str, Any], required_fields: List[str]) -> ValidationResult:
    """
    Validate input data for agent operations.
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
        
    Returns:
        ValidationResult: Validation result with errors and warnings
    """
    result = ValidationResult(is_valid=True)
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            result.add_error(f"Missing required field: {field}")
        elif data[field] is None:
            result.add_error(f"Field '{field}' cannot be None")
        elif isinstance(data[field], str) and not data[field].strip():
            result.add_error(f"Field '{field}' cannot be empty")
    
    # Validate specific field types
    if 'url' in data and data['url']:
        if not validate_url(data['url']):
            result.add_error("Invalid URL format")
    
    if 'email' in data and data['email']:
        if not validate_email(data['email']):
            result.add_error("Invalid email format")
    
    if 'doi' in data and data['doi']:
        if not validate_doi(data['doi']):
            result.add_error("Invalid DOI format")
    
    return result