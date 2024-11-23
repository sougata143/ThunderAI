from datetime import datetime, timedelta
from typing import Any, Union, Optional
from jose import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from .config import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    scopes: list[str] = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    if scopes:
        to_encode["scopes"] = scopes
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def decode_token(token: str) -> dict:
    """
    Decode a JWT token and return its payload
    """
    try:
        decoded_token = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return decoded_token
    except jwt.JWTError:
        return None

def verify_token_scopes(token: str, required_scopes: list[str]) -> bool:
    """
    Verify if a token has the required scopes
    """
    payload = decode_token(token)
    if not payload:
        return False
    
    token_scopes = payload.get("scopes", [])
    for scope in required_scopes:
        if scope not in token_scopes:
            return False
    return True
