from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any
from pydantic import EmailStr, constr
from datetime import datetime
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from ....core.security import decode_token
from ....services.user_service import UserService
from ....schemas.user import User, UserCreate, UserBase, UserUpdate
from ....core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(
    tags=["Users"],
    responses={404: {"description": "Not found"}}
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=settings.TOKEN_URL)
user_service = UserService()

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

@router.get("/", response_model=List[User])
async def list_users() -> List[User]:
    """List all users."""
    raise HTTPException(status_code=501, detail="Not implemented")

@router.get("/me", response_model=User)
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current user
    """
    try:
        payload = decode_token(token)
        if not payload:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = await user_service.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me/me", response_model=User)
async def get_current_user_me(current_user: User = Depends(get_current_user)) -> User:
    """Get current user information."""
    return current_user

@router.get("/{user_id}", response_model=User)
async def get_user(user_id: str) -> User:
    """
    Get user by ID
    """
    try:
        user = await user_service.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user: {str(e)}"
        )

@router.post("/", response_model=User)
async def create_user(user_in: UserCreate) -> User:
    """Create a new user."""
    try:
        user = await user_service.create(user_in)
        return user
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create user: {str(e)}"
        )

@router.put("/{user_id}", response_model=User)
async def update_user(user_id: str, user_update: UserUpdate) -> User:
    """Update user details."""
    try:
        user = await user_service.update(user_id, user_update)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update user: {str(e)}"
        )

@router.delete("/{user_id}")
async def delete_user(user_id: str) -> Dict[str, Any]:
    """Delete a user."""
    raise HTTPException(status_code=501, detail="Not implemented")
