from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
from pydantic import BaseModel, EmailStr, constr
from datetime import datetime
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: str = "user"  # "admin" or "user"

class UserCreate(UserBase):
    password: constr(min_length=8)

class User(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True

router = APIRouter(
    tags=["Users"],
    responses={404: {"description": "Not found"}}
)

# Mock data store
USERS = {
    1: {
        "id": 1,
        "email": "admin@thunderai.com",
        "full_name": "Admin User",
        "role": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "is_active": True
    }
}

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

@router.get("/", response_model=List[User])
async def list_users() -> List[User]:
    """List all users."""
    return list(USERS.values())

@router.get("/{user_id}", response_model=User)
async def get_user(user_id: int) -> User:
    """Get user details by ID."""
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail="User not found")
    return USERS[user_id]

@router.post("/", response_model=User)
async def create_user(user: UserCreate) -> User:
    """Create a new user."""
    # Check if email already exists
    if any(u["email"] == user.email for u in USERS.values()):
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    new_id = max(USERS.keys()) + 1 if USERS else 1
    current_time = datetime.now()
    
    new_user = {
        "id": new_id,
        **user.model_dump(exclude={"password"}),
        "hashed_password": get_password_hash(user.password),
        "created_at": current_time,
        "updated_at": current_time,
        "is_active": True
    }
    
    USERS[new_id] = new_user
    return new_user

@router.put("/{user_id}", response_model=User)
async def update_user(user_id: int, user_update: UserBase) -> User:
    """Update user details."""
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if email already exists for other users
    if any(u["email"] == user_update.email and u["id"] != user_id for u in USERS.values()):
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    user_data = USERS[user_id]
    update_data = user_update.model_dump()
    
    user_data.update({
        **update_data,
        "updated_at": datetime.now()
    })
    
    USERS[user_id] = user_data
    return user_data

@router.delete("/{user_id}")
async def delete_user(user_id: int):
    """Delete a user."""
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail="User not found")
    
    del USERS[user_id]
    return {"message": "User deleted successfully"}
