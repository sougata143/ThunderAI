from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.session import get_db
from db.crud import CRUDUser
from db.models import User
from typing import Dict, Any
from api.auth.jwt import verify_token

router = APIRouter(tags=["users"])
user_crud = CRUDUser(model=User)

@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Create a new user"""
    if user_crud.get_by_email(db, email=user_data["email"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    return user_crud.create_user(db=db, **user_data)

@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: Dict = Depends(verify_token)
):
    """Get user by ID"""
    user = user_crud.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.get("/users/me")
async def get_current_user(
    db: Session = Depends(get_db),
    token_data: Dict = Depends(verify_token)
):
    """Get current authenticated user"""
    user = user_crud.get(db, id=token_data["sub"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user 