from typing import Optional, Any, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from pydantic import EmailStr, BaseModel, Field
from ....core.security import get_current_user, get_password_hash, verify_password
from ....schemas.user import UserUpdate, UserResponse, ApiKeyResponse, ApiKeyCreate
from ....schemas.project import ProjectResponse
from ....models.user import User
from ....models.project import Project
from ....db.mongodb import db
from ....core.config import settings
import os
from PIL import Image
import secrets
from datetime import datetime
import uuid

router = APIRouter()

class UserPreferences(BaseModel):
    emailNotifications: bool = Field(default=False)
    twoFactorEnabled: bool = Field(default=False)
    theme: str = Field(default="light")
    language: str = Field(default="en")

class UserStats(BaseModel):
    modelsCreated: int = Field(default=0)
    experimentsRun: int = Field(default=0)
    totalTrainingHours: float = Field(default=0.0)
    lastActive: datetime = Field(default_factory=datetime.utcnow)

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    organization: Optional[str] = None
    jobTitle: Optional[str] = None
    bio: Optional[str] = None
    preferences: Optional[UserPreferences] = None

class PasswordChange(BaseModel):
    currentPassword: str
    newPassword: str

@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get current user's profile with extended information"""
    user_data = current_user.dict()
    
    # Get user statistics
    stats = await db.find_one("user_stats", {"user_id": current_user.id}) or {}
    user_data["stats"] = UserStats(
        modelsCreated=stats.get("models_created", 0),
        experimentsRun=stats.get("experiments_run", 0),
        totalTrainingHours=stats.get("total_training_hours", 0.0),
        lastActive=stats.get("last_active", datetime.utcnow())
    )
    
    # Get user preferences
    preferences = await db.find_one("user_preferences", {"user_id": current_user.id}) or {}
    user_data["preferences"] = UserPreferences(
        emailNotifications=preferences.get("email_notifications", False),
        twoFactorEnabled=preferences.get("two_factor_enabled", False),
        theme=preferences.get("theme", "light"),
        language=preferences.get("language", "en")
    )
    
    # Get API keys
    api_keys = await db.find_many("api_keys", {"user_id": current_user.id})
    user_data["apiKeys"] = [
        ApiKeyResponse(
            id=str(key["_id"]),
            name=key["name"],
            lastUsed=key.get("last_used", key["created_at"]),
            createdAt=key["created_at"]
        ) for key in api_keys
    ]
    
    return user_data

@router.put("/me", response_model=UserResponse)
async def update_profile(
    profile_update: UserProfileUpdate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """Update user profile"""
    try:
        update_data = profile_update.dict(exclude_unset=True)
        
        if "email" in update_data:
            existing_user = await db.find_one("users", {"email": update_data["email"], "_id": {"$ne": current_user.id}})
            if existing_user:
                raise HTTPException(status_code=400, detail="Email already registered")
        
        # Update user preferences if provided
        if profile_update.preferences:
            await db.update_one(
                "user_preferences",
                {"user_id": current_user.id},
                {"$set": profile_update.preferences.dict()},
                upsert=True
            )
        
        # Update user document
        result = await db.update_one(
            "users",
            {"_id": current_user.id},
            {"$set": {**update_data, "updated_at": datetime.utcnow()}}
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to update profile")
        
        return await get_profile(current_user)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: User = Depends(get_current_user)
):
    """Change user password"""
    if not verify_password(password_change.currentPassword, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    hashed_password = get_password_hash(password_change.newPassword)
    result = await db.update_one(
        "users",
        {"_id": current_user.id},
        {"$set": {"hashed_password": hashed_password}}
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to update password")
    
    return {"message": "Password updated successfully"}

@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    key_data: ApiKeyCreate,
    current_user: User = Depends(get_current_user)
):
    """Generate new API key"""
    api_key = {
        "_id": str(uuid.uuid4()),
        "user_id": current_user.id,
        "name": key_data.name,
        "key": f"sk-{secrets.token_urlsafe(32)}",
        "created_at": datetime.utcnow(),
        "last_used": datetime.utcnow()
    }
    
    result = await db.insert_one("api_keys", api_key)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create API key")
    
    return ApiKeyResponse(
        id=api_key["_id"],
        name=api_key["name"],
        key=api_key["key"],  # Only shown once during creation
        lastUsed=api_key["last_used"],
        createdAt=api_key["created_at"]
    )

@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete an API key"""
    result = await db.delete_one(
        "api_keys",
        {"_id": key_id, "user_id": current_user.id}
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return {"message": "API key deleted successfully"}

@router.get("/me/projects", response_model=list[ProjectResponse])
async def get_user_projects(current_user: User = Depends(get_current_user)):
    """Get current user's projects"""
    projects = await db.find_many(
        "projects",
        {"user_id": current_user.id}
    )
    return projects

@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a user's project"""
    try:
        # Check if project exists
        project = await db.find_one("projects", {"_id": project_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Check if user owns the project
        if project["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this project")

        # Delete project
        result = await db.delete_one("projects", {"_id": project_id})
        if result:
            return {"message": "Project deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete project")

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
