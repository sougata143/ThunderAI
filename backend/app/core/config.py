from pydantic import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "ThunderAI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    
    # Database
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "thunderai"
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3030"]
    
    # First Admin User
    FIRST_ADMIN_EMAIL: str = "admin@thunderai.com"
    FIRST_ADMIN_PASSWORD: str = "admin123"  # Change in production
    
    # JWT Token Settings
    TOKEN_URL: str = f"{API_V1_STR}/auth/login"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
