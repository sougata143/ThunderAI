from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from ..db.session import get_async_db
from ..models.user import User
from .config import get_settings

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

async def get_db() -> AsyncIOMotorDatabase:
    """
    Dependency function that returns MongoDB database
    """
    return await get_async_db()

async def get_current_user(
    db: AsyncIOMotorDatabase = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> Optional[User]:
    """
    Get the current authenticated user based on the JWT token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    try:
        user_dict = await db["users"].find_one({"_id": ObjectId(user_id)})
        if user_dict is None:
            raise credentials_exception
            
        # Convert ObjectId to string for the id field
        user_dict["id"] = str(user_dict["_id"])
        del user_dict["_id"]
            
        return User(**user_dict)
    except Exception as e:
        raise credentials_exception
