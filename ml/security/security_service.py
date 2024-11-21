from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer
from ..monitoring.custom_metrics import MetricsCollector
import logging

class SecurityService:
    def __init__(self, config: Dict[str, Any]):
        self.secret_key = config["SECRET_KEY"]
        self.algorithm = config.get("ALGORITHM", "HS256")
        self.access_token_expire_minutes = config.get("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.metrics_collector = MetricsCollector()
        
        # Rate limiting configuration
        self.rate_limit_window = config.get("RATE_LIMIT_WINDOW", 60)  # seconds
        self.max_requests = config.get("MAX_REQUESTS", 100)
        self.rate_limit_store = {}  # In-memory store for rate limiting
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(
                to_encode,
                self.secret_key,
                algorithm=self.algorithm
            )
            self.metrics_collector.record_security_metric(
                "token_creation",
                "success"
            )
            return encoded_jwt
        except Exception as e:
            self.metrics_collector.record_security_metric(
                "token_creation",
                "failure"
            )
            logging.error(f"Token creation failed: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            self.metrics_collector.record_security_metric(
                "token_verification",
                "success"
            )
            return payload
        except JWTError as e:
            self.metrics_collector.record_security_metric(
                "token_verification",
                "failure"
            )
            logging.error(f"Token verification failed: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limits"""
        current_time = datetime.utcnow()
        
        # Clean up old entries
        self._cleanup_rate_limit_store()
        
        if client_id not in self.rate_limit_store:
            self.rate_limit_store[client_id] = {
                'count': 1,
                'window_start': current_time
            }
            return True
        
        client_data = self.rate_limit_store[client_id]
        window_start = client_data['window_start']
        
        if (current_time - window_start).seconds > self.rate_limit_window:
            # Reset window
            self.rate_limit_store[client_id] = {
                'count': 1,
                'window_start': current_time
            }
            return True
        
        if client_data['count'] >= self.max_requests:
            self.metrics_collector.record_security_metric(
                "rate_limit_exceeded",
                client_id
            )
            return False
        
        client_data['count'] += 1
        return True
    
    def _cleanup_rate_limit_store(self):
        """Clean up expired rate limit entries"""
        current_time = datetime.utcnow()
        expired_clients = []
        
        for client_id, data in self.rate_limit_store.items():
            if (current_time - data['window_start']).seconds > self.rate_limit_window:
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            del self.rate_limit_store[client_id]
    
    async def get_current_user(
        self,
        token: str = Depends(oauth2_scheme)
    ) -> Dict[str, Any]:
        """Get current user from token"""
        return self.verify_token(token)

class RequestValidator:
    """Validate and sanitize incoming requests"""
    @staticmethod
    def validate_model_input(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model input data"""
        # Add input validation logic
        return data
    
    @staticmethod
    def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data"""
        # Add input sanitization logic
        return data

class AccessControl:
    """Handle role-based access control"""
    def __init__(self):
        self.role_permissions = {
            "admin": ["read", "write", "delete", "manage"],
            "user": ["read", "write"],
            "viewer": ["read"]
        }
    
    def check_permission(
        self,
        user_role: str,
        required_permission: str
    ) -> bool:
        """Check if user has required permission"""
        if user_role not in self.role_permissions:
            return False
        return required_permission in self.role_permissions[user_role] 