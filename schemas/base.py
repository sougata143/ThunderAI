from pydantic import BaseModel, EmailStr, validator
from typing import Dict, Any, Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str
    
    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v

class UserInDB(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class ModelBase(BaseModel):
    name: str
    version: Optional[str]

class ModelCreate(ModelBase):
    config: Dict[str, Any]
    metrics: Optional[Dict[str, float]]

class ModelInDB(ModelBase):
    id: int
    path: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class PredictionCreate(BaseModel):
    input_data: Dict[str, Any]
    model_name: str
    model_version: Optional[str]

class PredictionInDB(BaseModel):
    id: int
    user_id: int
    model_id: int
    input_data: Dict[str, Any]
    output: Dict[str, Any]
    confidence: float
    created_at: datetime
    
    class Config:
        orm_mode = True 