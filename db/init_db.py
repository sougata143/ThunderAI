from sqlalchemy.orm import Session
from alembic import command
from alembic.config import Config
import logging
from .crud import create_user
from api.schemas.user import UserCreate

logger = logging.getLogger(__name__)

def init_db(db: Session) -> None:
    """Initialize the database"""
    try:
        # Create an Alembic configuration object
        alembic_cfg = Config("alembic.ini")
        
        # Run all migrations
        command.upgrade(alembic_cfg, "head")
        
        # Create a test user
        test_user = UserCreate(
            email="test@example.com",
            password="password123"
        )
        create_user(db, test_user)
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise 