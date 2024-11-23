from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime
from ..core.config import get_settings
from ..models.user import User
from .mongodb import db

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=settings.TOKEN_URL)

async def get_db():
    """Get database instance with active connection"""
    if db.db is None:
        await db.connect_to_database()
    return db.db

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    database = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user_doc = await database.users.find_one({"email": email})
    if user_doc is None:
        raise credentials_exception
        
    return User(**user_doc)

async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=400, detail="The user doesn't have enough privileges"
        )
    return current_user
