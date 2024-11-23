from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from fastapi import HTTPException, status
from bson import ObjectId

from ..core.security import verify_password, get_password_hash
from ..core.config import settings
from ..db.mongodb import db
from ..schemas.user import User, UserCreate

async def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Authenticate a user by email and password
    """
    database = db.get_db()
    user = await database["users"].find_one({"email": email})
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return User(**user)

async def create_access_token(user_id: str) -> str:
    """
    Create access token for user
    """
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "exp": expire,
        "sub": str(user_id),
        "type": "access"
    }
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def create_user(user_data: UserCreate) -> User:
    """
    Create a new user
    """
    database = db.get_db()
    
    # Check if user exists
    if await database["users"].find_one({"email": user_data.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user_dict = user_data.dict()
    user_dict["hashed_password"] = get_password_hash(user_data.password)
    user_dict.pop("password")
    user_dict["_id"] = str(ObjectId())
    user_dict["created_at"] = datetime.utcnow()
    user_dict["is_active"] = True
    
    await database["users"].insert_one(user_dict)
    return User(**user_dict)

async def init_first_admin():
    # Check if admin exists
    database = db.get_db()
    admin = await database["users"].find_one({"email": settings.FIRST_ADMIN_EMAIL})
    if not admin:
        admin_user = UserCreate(
            email=settings.FIRST_ADMIN_EMAIL,
            password=settings.FIRST_ADMIN_PASSWORD,
            full_name="Admin User",
            username="admin",
            is_superuser=True
        )
        await create_user(admin_user)
