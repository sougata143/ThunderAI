from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import uuid

from ..auth.jwt import create_access_token
from db.session import get_db
from db.crud import get_user_by_email, verify_password
from core.config import settings

router = APIRouter()

@router.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Get access token for login"""
    user = get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email
        }
    } 

@router.post("/guest-token")
async def get_guest_token():
    """Get access token for guest login"""
    # Create a unique guest identifier
    guest_id = str(uuid.uuid4())
    
    # Create guest user data
    guest_data = {
        "id": guest_id,
        "email": f"guest_{guest_id}@example.com",
        "is_guest": True
    }
    
    # Create access token with shorter expiry for guests
    access_token_expires = timedelta(minutes=settings.GUEST_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": guest_data["email"], "is_guest": True},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": guest_data
    } 