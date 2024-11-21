import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from core.config import settings
import os

os.environ["TESTING"] = "1"

def test_database_connection():
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        db = SessionLocal()
        # Try to execute a simple query
        result = db.execute(text("SELECT 1"))
        assert result.scalar() == 1
        db.close()
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}") 