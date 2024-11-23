from datetime import timedelta
from typing import Any
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ....core.config import settings
from ....core.security import create_access_token
from ....schemas.token import Token
from ....schemas.user import User, UserCreate
from ....services.user_service import UserService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
user_service = UserService()

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    try:
        logger.info(f"Login attempt for user: {form_data.username}")
        user = await user_service.authenticate(
            email=form_data.username,  # OAuth2 uses username field for email
            password=form_data.password
        )
        if not user:
            logger.warning(f"Authentication failed for user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        elif not user.is_active:
            logger.warning(f"Inactive user attempt to login: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            subject=str(user.id),
            expires_delta=access_token_expires
        )
        
        logger.info(f"Login successful for user: {form_data.username}, token: {access_token[:20]}...")
        return {"access_token": access_token, "token_type": "Bearer"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post("/signup", response_model=User)
async def create_user(user_in: UserCreate) -> Any:
    """
    Create new user
    """
    try:
        user = await user_service.create(user_in)
        return user
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User creation failed: {str(e)}"
        )
