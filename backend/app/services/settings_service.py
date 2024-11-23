from fastapi import HTTPException, status
from app.schemas.settings import Settings, SettingsUpdate
from app.core.config import settings
from app.db.mongodb import db
import logging

logger = logging.getLogger(__name__)

class SettingsService:
    def __init__(self):
        self.db_name = settings.MONGODB_DB_NAME
        self.collection_name = "settings"

    async def _get_collection(self):
        database = db.get_db()
        return database[self.collection_name]

    async def get_settings(self) -> Settings:
        """
        Retrieve user settings.
        """
        try:
            collection = await self._get_collection()
            settings_doc = await collection.find_one({})
            
            if not settings_doc:
                # Return default settings if none exist
                return Settings()
                
            # Remove MongoDB _id field
            if '_id' in settings_doc:
                del settings_doc['_id']
                
            return Settings(**settings_doc)
        except Exception as e:
            logger.error(f"Failed to retrieve settings: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve settings: {str(e)}"
            )

    async def update_settings(self, settings_update: SettingsUpdate) -> Settings:
        """
        Update user settings.
        """
        try:
            collection = await self._get_collection()
            
            # Convert model to dict and remove None values
            update_dict = {k: v for k, v in settings_update.dict().items() if v is not None}
            
            if not update_dict:
                return await self.get_settings()

            # Upsert settings
            await collection.update_one(
                {}, 
                {"$set": update_dict},
                upsert=True
            )

            return await self.get_settings()
        except Exception as e:
            logger.error(f"Failed to update settings: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update settings: {str(e)}"
            )
