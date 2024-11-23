from typing import List, Optional
from pydantic import BaseSettings
import os
from motor.motor_asyncio import AsyncIOMotorClient

class Settings(BaseSettings):
    PROJECT_NAME: str = "ThunderAI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    
    # Database
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "thunderai")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3030", "http://localhost:8001", "http://localhost:3000"]
    
    # First Admin User
    FIRST_ADMIN_EMAIL: str = "admin@thunderai.com"
    FIRST_ADMIN_PASSWORD: str = "admin123"  # Change in production
    
    # JWT Token Settings
    TOKEN_URL: str = f"{API_V1_STR}/auth/login"
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")  # Required for text generation
    
    # Database client
    mongodb_client: Optional[AsyncIOMotorClient] = None
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()

async def get_database():
    """
    Get database instance
    """
    if not settings.mongodb_client:
        settings.mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
    return settings.mongodb_client[settings.MONGODB_DB_NAME]

async def close_database():
    """
    Close database connection
    """
    if settings.mongodb_client:
        settings.mongodb_client.close()
        settings.mongodb_client = None
