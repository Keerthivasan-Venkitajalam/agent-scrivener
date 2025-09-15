"""
Agent Scrivener API module.

This module provides REST API endpoints for the Agent Scrivener research platform.
"""

from .main import app
from .models import *
from .routes import *

__all__ = ["app"]