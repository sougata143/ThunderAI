from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "ThunderAI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENV: str = "development"
    
    # Database settings
    DATABASE_URL: str = "postgresql://thunderai_user:mypass@localhost:5432/thunderai"
    TEST_DATABASE_URL: str = "postgresql://thunderai_user:mypass@localhost:5432/thunderai_test"
    
    # JWT settings
    SECRET_KEY: str = "your-secret-key"  # Change in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]  # Change in production
    
    # Model settings
    MODEL_PATH: str = "models"
    
    # Cache settings
    REDIS_URL: str = "redis://localhost"
    CACHE_TTL: int = 60  # Cache time to live in seconds
    
    # Model Registry settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    
    # Guest settings
    GUEST_TOKEN_EXPIRE_MINUTES: int = 15  # Shorter expiry for guest tokens
    GUEST_ALLOWED_ENDPOINTS: List[str] = [
        "/api/v1/models",
        "/api/v1/predictions"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from environment variables

settings = Settings() 