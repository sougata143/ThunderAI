import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pprint import pprint

async def check_users():
    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.thunderai
    
    try:
        # Check if users collection exists
        collections = await db.list_collection_names()
        print(f"Collections in database: {collections}")
        
        # List all users
        users = []
        async for user in db.users.find({}):
            users.append(user)
        
        if users:
            print("\nFound users:")
            for user in users:
                # Remove password for security
                if 'hashed_password' in user:
                    user['hashed_password'] = '[REDACTED]'
                pprint(user)
        else:
            print("\nNo users found in database")
            
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(check_users())
