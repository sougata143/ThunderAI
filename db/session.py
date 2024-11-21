from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create engine with echo=True for debugging
engine = create_engine(
    settings.DATABASE_URL,
    echo=True  # Enable SQL logging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()  # Rollback on error
        raise
    finally:
        db.close() 