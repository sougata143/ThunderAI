from typing import Optional, List, Dict, Any
from fastapi import HTTPException, status
from bson import ObjectId
from datetime import datetime

from ..core.security import get_password_hash, verify_password
from ..schemas.user import UserCreate, UserUpdate, UserInDB, User
from ..db.mongodb import db

class UserService:
    def __init__(self):
        self.collection_name = "users"

    async def _get_collection(self):
        database = db.get_db()
        return database[self.collection_name]

    def prepare_user_dict(self, user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare user dictionary for model conversion."""
        # Create a copy to avoid modifying the original
        user_dict = dict(user_dict)
        
        # Handle ObjectId conversion
        if isinstance(user_dict.get("_id"), ObjectId):
            user_dict["_id"] = str(user_dict["_id"])
        elif isinstance(user_dict.get("id"), ObjectId):
            user_dict["_id"] = str(user_dict.pop("id"))
        
        # Ensure timestamps are present
        now = datetime.utcnow()
        if "created_at" not in user_dict:
            user_dict["created_at"] = now
        if "updated_at" not in user_dict:
            user_dict["updated_at"] = now
            
        return user_dict

    def convert_to_user(self, user_data: Dict[str, Any]) -> User:
        """Convert user data to User model."""
        user_dict = self.prepare_user_dict(user_data)
        return User(**user_dict)

    async def get(self, id: str) -> Optional[User]:
        """Get a user by ID."""
        try:
            collection = await self._get_collection()
            user_doc = await collection.find_one({"_id": ObjectId(id)})
            if user_doc:
                return self.convert_to_user(user_doc)
            return None
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve user: {str(e)}"
            )

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        try:
            collection = await self._get_collection()
            user_doc = await collection.find_one({"email": email})
            if user_doc:
                return self.convert_to_user(user_doc)
            return None
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve user: {str(e)}"
            )

    async def create(self, user_in: UserCreate) -> User:
        """Create a new user."""
        try:
            collection = await self._get_collection()
            
            # Check if user with email exists
            existing_user = await self.get_by_email(user_in.email)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Create user document
            now = datetime.utcnow()
            user_doc = UserInDB(
                _id=ObjectId(),
                email=user_in.email,
                username=user_in.username,
                full_name=user_in.full_name,
                hashed_password=get_password_hash(user_in.password),
                is_active=True,
                is_superuser=False,
                created_at=now,
                updated_at=now
            )
            
            # Insert into database
            await collection.insert_one(user_doc.dict(by_alias=True))
            
            # Return user without hashed password
            user_dict = user_doc.dict(exclude={"hashed_password"})
            return self.convert_to_user(user_dict)
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user: {str(e)}"
            )

    async def update(self, id: str, user_in: UserUpdate) -> Optional[User]:
        """Update a user."""
        try:
            collection = await self._get_collection()
            update_data = user_in.dict(exclude_unset=True)
            
            if "password" in update_data:
                update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
            
            update_data["updated_at"] = datetime.utcnow()
            
            if update_data:
                result = await collection.update_one(
                    {"_id": ObjectId(id)},
                    {"$set": update_data}
                )
                if result.modified_count:
                    return await self.get(id)
            return None
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update user: {str(e)}"
            )

    async def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        try:
            collection = await self._get_collection()
            user_doc = await collection.find_one({"email": email})
            if not user_doc:
                return None
            
            user_in_db = UserInDB(**self.prepare_user_dict(user_doc))
            
            if not verify_password(password, user_in_db.hashed_password):
                return None
            
            # Return user without hashed password
            user_dict = user_in_db.dict(exclude={"hashed_password"})
            return self.convert_to_user(user_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to authenticate user: {str(e)}"
            )
