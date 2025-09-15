"""
Unit tests for API authentication and authorization.
"""

import pytest
import jwt
import time
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from agent_scrivener.api.auth import (
    create_access_token, verify_token, get_current_user,
    check_rate_limit, rate_limit_dependency, auth_config, TokenData
)


class TestTokenOperations:
    """Test JWT token creation and verification."""
    
    def test_create_access_token(self):
        """Test creating a valid access token."""
        token = create_access_token("user123", "testuser", ["read", "write"])
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify payload
        payload = jwt.decode(token, auth_config.secret_key, algorithms=[auth_config.algorithm])
        assert payload["user_id"] == "user123"
        assert payload["username"] == "testuser"
        assert payload["scopes"] == ["read", "write"]
        assert "iat" in payload
        assert "exp" in payload
    
    def test_create_token_default_scopes(self):
        """Test creating token with default scopes."""
        token = create_access_token("user123", "testuser")
        
        payload = jwt.decode(token, auth_config.secret_key, algorithms=[auth_config.algorithm])
        assert payload["scopes"] == ["read", "write"]
    
    def test_verify_valid_token(self):
        """Test verifying a valid token."""
        token = create_access_token("user123", "testuser", ["read"])
        
        token_data = verify_token(token)
        
        assert isinstance(token_data, TokenData)
        assert token_data.user_id == "user123"
        assert token_data.username == "testuser"
        assert token_data.scopes == ["read"]
    
    def test_verify_expired_token(self):
        """Test verifying an expired token."""
        # Create token that expires immediately
        now = int(time.time())
        payload = {
            "user_id": "user123",
            "username": "testuser",
            "scopes": ["read"],
            "iat": now - 3600,
            "exp": now - 1800  # Expired 30 minutes ago
        }
        
        expired_token = jwt.encode(payload, auth_config.secret_key, algorithm=auth_config.algorithm)
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(expired_token)
        
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()
    
    def test_verify_invalid_token(self):
        """Test verifying an invalid token."""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(invalid_token)
        
        assert exc_info.value.status_code == 401
        assert "invalid" in exc_info.value.detail.lower()
    
    def test_verify_token_wrong_secret(self):
        """Test verifying token with wrong secret."""
        # Create token with different secret
        wrong_payload = {
            "user_id": "user123",
            "username": "testuser",
            "scopes": ["read"],
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600
        }
        
        wrong_token = jwt.encode(wrong_payload, "wrong-secret", algorithm=auth_config.algorithm)
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(wrong_token)
        
        assert exc_info.value.status_code == 401


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Clear rate limit storage before each test."""
        from agent_scrivener.api.auth import _rate_limit_storage
        _rate_limit_storage.clear()
    
    def test_rate_limit_within_limit(self):
        """Test rate limiting within allowed limit."""
        user_id = "test_user"
        
        # Should allow requests within limit
        for i in range(auth_config.rate_limit_requests_per_minute):
            assert check_rate_limit(user_id) is True
    
    def test_rate_limit_exceeded(self):
        """Test rate limiting when limit is exceeded."""
        user_id = "test_user"
        
        # Fill up the rate limit
        for i in range(auth_config.rate_limit_requests_per_minute):
            check_rate_limit(user_id)
        
        # Next request should be denied
        assert check_rate_limit(user_id) is False
    
    def test_rate_limit_different_users(self):
        """Test rate limiting for different users."""
        user1 = "user1"
        user2 = "user2"
        
        # Fill up rate limit for user1
        for i in range(auth_config.rate_limit_requests_per_minute):
            check_rate_limit(user1)
        
        # user1 should be rate limited
        assert check_rate_limit(user1) is False
        
        # user2 should still be allowed
        assert check_rate_limit(user2) is True
    
    @patch('time.time')
    def test_rate_limit_window_reset(self, mock_time):
        """Test rate limit window reset."""
        user_id = "test_user"
        
        # Set initial time
        mock_time.return_value = 1000.0
        
        # Fill up rate limit
        for i in range(auth_config.rate_limit_requests_per_minute):
            check_rate_limit(user_id)
        
        # Should be rate limited
        assert check_rate_limit(user_id) is False
        
        # Move to next minute window
        mock_time.return_value = 1060.0  # 60 seconds later
        
        # Should be allowed again
        assert check_rate_limit(user_id) is True


class TestAuthenticationDependencies:
    """Test FastAPI authentication dependencies."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self):
        """Test getting current user with valid token."""
        token = create_access_token("user123", "testuser", ["read", "write"])
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        user_data = await get_current_user(credentials)
        
        assert user_data.user_id == "user123"
        assert user_data.username == "testuser"
        assert user_data.scopes == ["read", "write"]
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test getting current user with invalid token."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid.token")
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials)
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_rate_limit_dependency_allowed(self):
        """Test rate limit dependency when within limit."""
        token = create_access_token("user123", "testuser")
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        # Mock get_current_user to return user data
        with patch('agent_scrivener.api.auth.get_current_user') as mock_get_user:
            mock_get_user.return_value = TokenData(
                user_id="user123",
                username="testuser",
                scopes=["read", "write"],
                iat=int(time.time()),
                exp=int(time.time()) + 3600
            )
            
            user_data = await rate_limit_dependency(mock_get_user.return_value)
            assert user_data.user_id == "user123"
    
    @pytest.mark.asyncio
    async def test_rate_limit_dependency_exceeded(self):
        """Test rate limit dependency when limit exceeded."""
        user_data = TokenData(
            user_id="user123",
            username="testuser",
            scopes=["read", "write"],
            iat=int(time.time()),
            exp=int(time.time()) + 3600
        )
        
        # Fill up rate limit
        for i in range(auth_config.rate_limit_requests_per_minute):
            check_rate_limit("user123")
        
        with patch('agent_scrivener.api.auth.get_current_user') as mock_get_user:
            mock_get_user.return_value = user_data
            
            with pytest.raises(HTTPException) as exc_info:
                await rate_limit_dependency(user_data)
            
            assert exc_info.value.status_code == 429
            assert "rate limit" in exc_info.value.detail.lower()