import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import db
from app.services.auth_service import init_first_admin
from app.core.config import get_settings

settings = get_settings()

async def init_database():
    print("Connecting to database...")
    await db.connect_to_database()
    
    print("Initializing first admin user...")
    await init_first_admin(db.db)
    
    print("Database initialization complete!")
    await db.close_database_connection()

if __name__ == "__main__":
    asyncio.run(init_database())
