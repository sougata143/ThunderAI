from datetime import timedelta
from typing import Optional, Tuple
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from ..core.security import verify_password, create_access_token, get_password_hash
from ..core.config import get_settings
from ..models.user import UserCreate, UserInDB
from ..schemas.user import Token, User

settings = get_settings()

async def authenticate_user(db: AsyncIOMotorDatabase, email: str, password: str) -> Optional[UserInDB]:
    user_dict = await db["users"].find_one({"email": email})
    if not user_dict:
        return None
    user = UserInDB(**user_dict)
    if not verify_password(password, user.hashed_password):
        return None
    return user

async def get_user_by_email(db: AsyncIOMotorDatabase, email: str) -> Optional[UserInDB]:
    user_dict = await db["users"].find_one({"email": email})
    if user_dict:
        return UserInDB(**user_dict)
    return None

async def create_user(db: AsyncIOMotorDatabase, user_in: UserCreate) -> UserInDB:
    # Check if user exists
    existing_user = await get_user_by_email(db, user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_dict = user_in.dict(exclude={"password"})
    user_dict["hashed_password"] = get_password_hash(user_in.password)
    
    result = await db["users"].insert_one(user_dict)
    user_dict["_id"] = result.inserted_id
    
    return UserInDB(**user_dict)

def create_token_for_user(user: UserInDB) -> Token:
    scopes = ["admin"] if user.is_superuser else ["user"]
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.user_id,
        expires_delta=access_token_expires,
        scopes=scopes
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=User(
            id=user.user_id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    )

async def init_first_admin(db: AsyncIOMotorDatabase):
    # Check if admin exists
    admin = await get_user_by_email(db, settings.FIRST_ADMIN_EMAIL)
    if not admin:
        admin_user = UserCreate(
            email=settings.FIRST_ADMIN_EMAIL,
            password=settings.FIRST_ADMIN_PASSWORD,
            full_name="Admin User",
            username="admin",
            is_superuser=True
        )
        await create_user(db, admin_user)
