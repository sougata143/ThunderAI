from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "ThunderAI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./thunderai.db"  # Default SQLite database
    
    # JWT settings
    SECRET_KEY: str = "your-secret-key-here"  # In production, use proper secret management
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Redis settings
    REDIS_URL: str = "redis://localhost"
    
    # Model settings
    MODEL_PATH: str = "./models"
    CACHE_TTL: int = 60  # Cache time to live in seconds
    
    # Monitoring settings
    PROMETHEUS_MULTIPROC_DIR: Optional[str] = None
    METRICS_PORT: int = 8001

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 