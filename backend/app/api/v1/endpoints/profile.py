from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from pydantic import EmailStr
from ....core.security import get_current_user
from ....schemas.user import UserUpdate, UserResponse
from ....schemas.project import ProjectResponse
from ....models.user import User
from ....models.project import Project
from ....db.mongodb import db
from ....core.config import settings
import os
from PIL import Image
import secrets
from datetime import datetime

router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get current user's profile"""
    return current_user

@router.get("/me/projects", response_model=list[ProjectResponse])
async def get_user_projects(current_user: User = Depends(get_current_user)):
    """Get current user's projects"""
    projects = await db.find_many(
        "projects",
        {"user_id": current_user.id}
    )
    return projects

@router.put("/me", response_model=UserResponse)
async def update_profile(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Update current user.
    """
    try:
        # Verify current password
        if user_in.current_password and not current_user.verify_password(user_in.current_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # Check username availability
        if user_in.username and user_in.username != current_user.username:
            existing_user = await db.find_one("users", {"username": user_in.username})
            if existing_user:
                raise HTTPException(status_code=400, detail="Username is already taken")

        # Check email availability
        if user_in.email and user_in.email != current_user.email:
            existing_user = await db.find_one("users", {"email": user_in.email})
            if existing_user:
                raise HTTPException(status_code=400, detail="Email is already taken")

        # Update user data
        update_data = {
            "username": user_in.username or current_user.username,
            "email": user_in.email or current_user.email,
            "updated_at": datetime.utcnow()
        }

        # Handle password update
        if user_in.new_password:
            update_data["hashed_password"] = User.get_password_hash(user_in.new_password)

        # Handle profile picture upload
        if user_in.profile_picture:
            # Validate file type
            if user_in.profile_picture.content_type not in settings.ALLOWED_IMAGE_TYPES:
                raise HTTPException(status_code=400, detail="File must be a valid image type (JPEG, PNG, or GIF)")

            # Check file size
            contents = await user_in.profile_picture.read()
            if len(contents) > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=400, detail=f"File size must be less than {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB")
            
            # Reset file pointer for later use
            await user_in.profile_picture.seek(0)

            # Generate unique filename
            ext = user_in.profile_picture.filename.split('.')[-1].lower()
            filename = f"{secrets.token_hex(8)}_{current_user.id}.{ext}"
            file_path = os.path.join(settings.PROFILE_PICTURES_DIR, filename)

            # Save and process image
            try:
                # Save uploaded file
                with open(file_path, "wb") as buffer:
                    await user_in.profile_picture.seek(0)
                    buffer.write(contents)

                # Process image (resize if needed)
                with Image.open(file_path) as img:
                    # Resize to maximum dimensions while maintaining aspect ratio
                    max_size = (500, 500)
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    img.save(file_path, quality=85, optimize=True)

                # Delete old profile picture if it exists
                if current_user.profile_picture:
                    old_file_path = os.path.join(settings.STATIC_DIR, current_user.profile_picture.lstrip('/'))
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)

                # Update profile picture path
                update_data["profile_picture"] = f"/static/uploads/profile_pictures/{filename}"

            except Exception as e:
                # Clean up file if there was an error
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(status_code=500, detail=str(e))

        # Update user in database
        result = await db.update_one(
            "users",
            {"_id": current_user.id},
            {"$set": update_data}
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to update profile")

        # Get updated user data
        updated_user = await db.find_one("users", {"_id": current_user.id})
        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return JSONResponse(content={
            "message": "Profile updated successfully",
            "user": UserResponse(**updated_user).dict()
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
