"""
Authentication and authorization utilities for Agent Scrivener API.
"""

import jwt
import time
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    username: str
    scopes: list[str]
    exp: int
    iat: int


class AuthConfig:
    """Authentication configuration."""
    def __init__(self):
        # In production, these should come from environment variables
        self.secret_key = "your-secret-key-change-in-production"
        self.algorithm = "HS256"
        self.token_expire_minutes = 60 * 24  # 24 hours
        self.rate_limit_requests_per_minute = 60


auth_config = AuthConfig()
security = HTTPBearer()


def create_access_token(user_id: str, username: str, scopes: list[str] = None) -> str:
    """Create a JWT access token."""
    if scopes is None:
        scopes = ["read", "write"]
    
    now = int(time.time())
    payload = {
        "user_id": user_id,
        "username": username,
        "scopes": scopes,
        "iat": now,
        "exp": now + (auth_config.token_expire_minutes * 60)
    }
    
    return jwt.encode(payload, auth_config.secret_key, algorithm=auth_config.algorithm)


def verify_token(token: str) -> TokenData:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, auth_config.secret_key, algorithms=[auth_config.algorithm])
        return TokenData(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """Get current authenticated user from token."""
    return verify_token(credentials.credentials)


def require_scope(required_scope: str):
    """Dependency to require specific scope."""
    def scope_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if required_scope not in current_user.scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required scope: {required_scope}"
            )
        return current_user
    return scope_checker


# Rate limiting storage (in production, use Redis)
_rate_limit_storage: Dict[str, Dict[str, Any]] = {}


def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit."""
    now = time.time()
    minute_window = int(now // 60)
    
    if user_id not in _rate_limit_storage:
        _rate_limit_storage[user_id] = {}
    
    user_data = _rate_limit_storage[user_id]
    
    # Clean old windows
    old_windows = [w for w in user_data.keys() if int(w) < minute_window - 1]
    for window in old_windows:
        del user_data[window]
    
    # Check current window
    current_window = str(minute_window)
    if current_window not in user_data:
        user_data[current_window] = 0
    
    if user_data[current_window] >= auth_config.rate_limit_requests_per_minute:
        return False
    
    user_data[current_window] += 1
    return True


async def rate_limit_dependency(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Rate limiting dependency."""
    if not check_rate_limit(current_user.user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    return current_user