from typing import Dict, Any, List, Optional
from pydantic import BaseSettings
import ssl
import jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
import logging

class SecurityConfig(BaseSettings):
    # JWT Settings
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Password Settings
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_NUMBERS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    # Rate Limiting
    RATE_LIMIT_WINDOW: int = 60  # seconds
    MAX_REQUESTS_PER_WINDOW: int = 100
    
    # SSL/TLS Settings
    SSL_CERT_PATH: Optional[str] = None
    SSL_KEY_PATH: Optional[str] = None
    SSL_VERIFY: bool = True
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # Security Headers
    SECURITY_HEADERS: Dict[str, str] = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }
    
    class Config:
        env_file = ".env"

class SecurityService:
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
        # Initialize SSL context if certificates are provided
        self.ssl_context = self._create_ssl_context() if config.SSL_CERT_PATH else None
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure communications"""
        context = ssl.create_default_context()
        context.load_cert_chain(
            self.config.SSL_CERT_PATH,
            self.config.SSL_KEY_PATH
        )
        if not self.config.SSL_VERIFY:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return self.pwd_context.hash(password)
    
    def validate_password_strength(self, password: str) -> Dict[str, bool]:
        """Validate password strength"""
        validations = {
            "length": len(password) >= self.config.PASSWORD_MIN_LENGTH,
            "uppercase": any(c.isupper() for c in password) if self.config.PASSWORD_REQUIRE_UPPERCASE else True,
            "numbers": any(c.isdigit() for c in password) if self.config.PASSWORD_REQUIRE_NUMBERS else True,
            "special": any(not c.isalnum() for c in password) if self.config.PASSWORD_REQUIRE_SPECIAL else True
        }
        return validations
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (
            expires_delta or timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            self.config.JWT_SECRET_KEY,
            algorithm=self.config.JWT_ALGORITHM
        )
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        return self.create_access_token(
            data,
            timedelta(days=self.config.REFRESH_TOKEN_EXPIRE_DAYS)
        )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.JWT_SECRET_KEY,
                algorithms=[self.config.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logging.error("Token has expired")
            raise ValueError("Token expired")
        except jwt.JWTError as e:
            logging.error(f"Token verification failed: {str(e)}")
            raise ValueError("Invalid token")
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for responses"""
        return self.config.SECURITY_HEADERS.copy() 