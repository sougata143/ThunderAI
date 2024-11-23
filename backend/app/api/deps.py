from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.utils import get_authorization_scheme_param
from jose import jwt, JWTError
from datetime import datetime
import logging

from ..core.config import settings
from ..core.security import verify_password
from ..services.user_service import UserService
from ..schemas.token import TokenPayload
from ..schemas.user import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomOAuth2PasswordBearer(OAuth2PasswordBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        authorization = request.headers.get("Authorization")
        logger.info(f"Authorization header: {authorization[:30] if authorization else None}")
        
        if not authorization:
            logger.error("No authorization header")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        scheme, param = get_authorization_scheme_param(authorization)
        if scheme.lower() != "bearer":
            logger.error(f"Invalid authorization scheme: {scheme}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization scheme",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        if not param:
            logger.error("Empty token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Empty token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return param

# Use custom OAuth2 bearer
reusable_oauth2 = CustomOAuth2PasswordBearer(tokenUrl=settings.TOKEN_URL)
user_service = UserService()

async def get_current_user(
    token: str = Depends(reusable_oauth2)
) -> User:
    try:
        # Log raw token
        logger.info(f"Raw token received: {token[:20] if token else 'None'}...")
        
        if not token:
            logger.error("No token provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No credentials provided",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        try:
            # Decode token
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            logger.info(f"Token decoded successfully: {payload}")
            
            # Validate required fields
            required_fields = ["sub", "exp", "type"]
            missing_fields = [field for field in required_fields if field not in payload]
            if missing_fields:
                logger.error(f"Token missing required fields: {missing_fields}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token: missing fields {missing_fields}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            # Parse token data
            token_data = TokenPayload(**payload)
            logger.info("Token payload validated")
            
            # Check expiration
            if datetime.fromtimestamp(token_data.exp) < datetime.now():
                logger.warning("Token has expired")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            # Check token type
            if token_data.type != "access":
                logger.warning(f"Invalid token type: {token_data.type}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            # Get user
            logger.info(f"Looking up user with ID: {token_data.sub}")
            user = await user_service.get(token_data.sub)
            if not user:
                logger.warning(f"User not found for ID: {token_data.sub}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            logger.info("User authentication successful")
            return user
            
        except jwt.JWTError as e:
            logger.error(f"JWT decode error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Could not validate credentials: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user
