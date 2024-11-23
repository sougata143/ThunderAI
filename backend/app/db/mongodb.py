from motor.motor_asyncio import AsyncIOMotorClient
from ..core.config import get_settings

settings = get_settings()

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    async def connect_to_database(self):
        """Connect to MongoDB if not already connected"""
        if self.client is None:
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=10,
                minPoolSize=1
            )
            self.db = self.client[settings.DATABASE_NAME]
            print("Connected to MongoDB.")

    async def close_database_connection(self):
        """Close MongoDB connection - should only be called on application shutdown"""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
            print("Closed MongoDB connection.")

    def get_db(self):
        """Get database instance - creates connection if needed"""
        if self.db is None:
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=10,
                minPoolSize=1
            )
            self.db = self.client[settings.DATABASE_NAME]
            print("Connected to MongoDB.")
        return self.db

db = MongoDB()
