import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base
from core.config import settings
from fastapi.testclient import TestClient
from api.main import app
from api.auth.jwt import create_access_token
from datetime import timedelta

@pytest.fixture(scope="session")
def engine():
    return create_engine(settings.DATABASE_URL)

@pytest.fixture(scope="session")
def tables(engine):
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(engine, tables):
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_user():
    # Create a valid access token for testing
    access_token = create_access_token(
        data={"sub": "test@example.com"},
        expires_delta=timedelta(minutes=30)
    )
    return {
        "id": 1,
        "email": "test@example.com",
        "access_token": access_token
    } 