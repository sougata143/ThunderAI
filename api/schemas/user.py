from pydantic import BaseModel, EmailStr, Field, ConfigDict

class UserBase(BaseModel):
    email: EmailStr = Field(..., description="User's email address")

class UserCreate(UserBase):
    password: str = Field(
        ..., 
        min_length=8,
        description="User's password (min 8 characters)"
    )

class User(UserBase):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "email": "user@example.com",
                "is_active": True
            }
        }
    )
    
    id: int
    is_active: bool = True 