"""
Logging configuration and utilities for Agent Scrivener.
"""

import logging
import structlog
from typing import Any, Dict
import sys
from datetime import datetime


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """
    Configure structured logging for Agent Scrivener.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON formatting for logs
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        structlog.BoundLogger: Configured logger instance
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    """
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            module_name = self.__class__.__module__
            self._logger = get_logger(f"{module_name}.{class_name}")
        return self._logger
    
    def log_operation_start(self, operation: str, **context: Any) -> None:
        """Log the start of an operation."""
        self.logger.info(
            "Operation started",
            operation=operation,
            timestamp=datetime.now().isoformat(),
            **context
        )
    
    def log_operation_success(self, operation: str, duration_ms: int, **context: Any) -> None:
        """Log successful completion of an operation."""
        self.logger.info(
            "Operation completed successfully",
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            **context
        )
    
    def log_operation_error(self, operation: str, error: Exception, **context: Any) -> None:
        """Log an operation error."""
        self.logger.error(
            "Operation failed",
            operation=operation,
            error=str(error),
            error_type=type(error).__name__,
            timestamp=datetime.now().isoformat(),
            **context
        )