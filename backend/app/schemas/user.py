from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}

class UserCreate(UserBase):
    password: str

class UserUpdate(UserBase):
    password: Optional[str] = None

class UserPreferences(BaseModel):
    emailNotifications: bool = Field(default=False)
    twoFactorEnabled: bool = Field(default=False)
    theme: str = Field(default="light")
    language: str = Field(default="en")

class UserStats(BaseModel):
    modelsCreated: int = Field(default=0)
    experimentsRun: int = Field(default=0)
    totalTrainingHours: float = Field(default=0.0)
    lastActive: datetime = Field(default_factory=datetime.utcnow)

class ApiKeyBase(BaseModel):
    name: str

class ApiKeyCreate(ApiKeyBase):
    pass

class ApiKeyResponse(ApiKeyBase):
    id: str
    lastUsed: datetime
    createdAt: datetime
    key: Optional[str] = None  # Only included when key is first created

class UserInDBBase(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    preferences: Optional[UserPreferences] = None
    stats: Optional[UserStats] = None
    apiKeys: Optional[List[ApiKeyResponse]] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}
        schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "email": "user@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "is_active": True,
                "is_superuser": False,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00",
                "preferences": {
                    "emailNotifications": False,
                    "twoFactorEnabled": False,
                    "theme": "light",
                    "language": "en"
                },
                "stats": {
                    "modelsCreated": 0,
                    "experimentsRun": 0,
                    "totalTrainingHours": 0.0,
                    "lastActive": "2023-01-01T00:00:00"
                },
                "apiKeys": []
            }
        }

class UserInDB(UserInDBBase):
    hashed_password: str

class User(UserBase):
    id: str = Field(alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    preferences: Optional[UserPreferences] = None
    stats: Optional[UserStats] = None
    apiKeys: Optional[List[ApiKeyResponse]] = None

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}
        schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "email": "user@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "is_active": True,
                "is_superuser": False,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00",
                "preferences": {
                    "emailNotifications": False,
                    "twoFactorEnabled": False,
                    "theme": "light",
                    "language": "en"
                },
                "stats": {
                    "modelsCreated": 0,
                    "experimentsRun": 0,
                    "totalTrainingHours": 0.0,
                    "lastActive": "2023-01-01T00:00:00"
                },
                "apiKeys": []
            }
        }

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User
