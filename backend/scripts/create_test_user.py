import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from passlib.context import CryptContext

# MongoDB connection
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "thunderai"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_test_user():
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    
    # Test user data
    test_user = {
        "email": "test@example.com",
        "full_name": "Test User",
        "username": "testuser",
        "is_active": True,
        "is_superuser": False,
        "hashed_password": pwd_context.hash("password123"),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    # Check if user exists
    existing_user = await db.users.find_one({"email": test_user["email"]})
    if existing_user:
        print(f"User {test_user['email']} already exists")
        return
    
    # Create user
    result = await db.users.insert_one(test_user)
    print(f"Created test user with id: {result.inserted_id}")
    
    # Close connection
    client.close()

if __name__ == "__main__":
    asyncio.run(create_test_user())
