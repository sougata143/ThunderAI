from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool = True
    created_at: Optional[datetime] = None
    organization: Optional[str] = None
    job_title: Optional[str] = None
    settings: Optional[dict] = None
    
    class Config:
        from_attributes = True