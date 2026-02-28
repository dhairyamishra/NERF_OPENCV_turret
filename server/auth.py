"""
Authentication module for the remote admin server.
Uses JWT tokens with password-based login.
"""

import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("python-jose not installed. Auth will use simple tokens.")

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 480  # 8 hours

security = HTTPBearer(auto_error=False)


class AuthManager:
    """Handles password verification and JWT token creation/validation."""

    def __init__(self, secret_key: str, admin_username: str, admin_password: str):
        self.secret_key = secret_key
        self.admin_username = admin_username
        self._admin_hash = self._hash_password(admin_password)

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def verify_credentials(self, username: str, password: str) -> bool:
        if username != self.admin_username:
            return False
        return hmac.compare_digest(self._hash_password(password), self._admin_hash)

    def create_token(self, username: str) -> str:
        if not JWT_AVAILABLE:
            return "auth-disabled-token"
        expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
        payload = {"sub": username, "exp": expire}
        return jwt.encode(payload, self.secret_key, algorithm=ALGORITHM)

    def verify_token(self, token: str) -> Optional[str]:
        if not JWT_AVAILABLE:
            return self.admin_username
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            return username
        except JWTError:
            return None


def check_default_credentials(admin_username: str, admin_password: str):
    """Warn loudly if the server is using default credentials."""
    defaults = [("admin", "changeme"), ("admin", "admin"), ("admin", "password")]
    if (admin_username, admin_password) in defaults:
        logger.warning("=" * 60)
        logger.warning("  SECURITY WARNING: Default credentials detected!")
        logger.warning(f"  Username: {admin_username}, Password: {'*' * len(admin_password)}")
        logger.warning("  Change 'admin_username' and 'admin_password' in config.yaml")
        logger.warning("=" * 60)
        return True
    return False


def get_auth_dependency(auth_manager: AuthManager):
    """Create a FastAPI dependency for route protection."""

    async def verify(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        username = auth_manager.verify_token(credentials.credentials)
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username

    return verify
