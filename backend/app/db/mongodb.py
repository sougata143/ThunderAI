from motor.motor_asyncio import AsyncIOMotorClient
from ..core.config import settings
from fastapi import HTTPException, status

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    async def connect_to_database(self):
        """Connect to MongoDB database."""
        if self.client is None:
            try:
                self.client = AsyncIOMotorClient(settings.MONGODB_URL)
                self.db = self.client[settings.MONGODB_DB_NAME]
                # Test the connection
                await self.client.admin.command('ping')
                print(f"Connected to MongoDB at {settings.MONGODB_URL}")
            except Exception as e:
                print(f"Failed to connect to MongoDB: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Could not connect to database"
                )

    def get_db(self):
        """Get database instance."""
        if self.db is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database connection not initialized"
            )
        return self.db

    async def close_database_connection(self):
        """Close database connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
            print("Closed MongoDB connection")

db = MongoDB()
