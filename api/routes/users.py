from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
from db.session import get_db
from db.models import User as UserModel
from api.auth.jwt import verify_token
from api.schemas.user import User as UserSchema, UserCreate

router = APIRouter()

@router.get("/me", response_model=UserSchema)
async def get_current_user(
    token: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get current user"""
    try:
        username = token.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

        user = db.query(UserModel).filter(UserModel.username == username).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/profile", response_model=Dict[str, UserSchema])
async def get_profile(
    token: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get user profile"""
    try:
        username = token.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

        user = db.query(UserModel).filter(UserModel.username == username).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Convert to dict and filter out None values
        user_dict = {
            "username": user.username,
            "email": user.email,
            "id": user.id,
            "is_active": user.is_active,
            "full_name": user.full_name,
            "organization": user.organization,
            "job_title": user.job_title,
            "settings": user.settings,
            "created_at": user.created_at
        }
        
        # Remove None values
        user_dict = {k: v for k, v in user_dict.items() if v is not None}
        
        return {"data": UserSchema(**user_dict)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/profile", response_model=Dict[str, str])
async def update_profile(
    profile_data: UserCreate,
    token: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    try:
        username = token.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

        user = db.query(UserModel).filter(UserModel.username == username).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Update user fields
        for field, value in profile_data.model_dump(exclude_unset=True).items():
            if hasattr(user, field):
                setattr(user, field, value)

        db.commit()
        db.refresh(user)

        return {"message": "Profile updated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/", response_model=UserSchema)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Create new user"""
    try:
        # Check if user exists
        if db.query(UserModel).filter(UserModel.username == user_data.username).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if db.query(UserModel).filter(UserModel.email == user_data.email).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create new user
        user = UserModel(
            username=user_data.username,
            email=user_data.email,
            hashed_password=user_data.password  # Note: You should hash this password
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{username}", response_model=UserSchema)
async def get_user(username: str, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.username == username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user 