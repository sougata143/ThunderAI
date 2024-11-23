from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from ..core.config import get_settings

settings = get_settings()

# Async MongoDB client for FastAPI
async_client = AsyncIOMotorClient(settings.MONGODB_URL)
async_db = async_client[settings.DATABASE_NAME]

# Sync MongoDB client for dependencies
client = MongoClient(settings.MONGODB_URL)
db = client[settings.DATABASE_NAME]

def get_db_session():
    """Get a MongoDB database session"""
    return db

async def get_async_db():
    """Get an async MongoDB database session"""
    return async_db

def init_db():
    """Initialize database with required collections"""
    # Create collections if they don't exist
    if "users" not in db.list_collection_names():
        db.create_collection("users")
        # Create indexes
        db.users.create_index("email", unique=True)
        db.users.create_index("username", unique=True)

    if "models" not in db.list_collection_names():
        db.create_collection("models")
        # Create indexes for model collection
        db.models.create_index("name", unique=True)
        db.models.create_index("created_at")

    if "experiments" not in db.list_collection_names():
        db.create_collection("experiments")
        # Create indexes for experiments collection
        db.experiments.create_index([("user_id", 1), ("name", 1)], unique=True)
        db.experiments.create_index("created_at")
