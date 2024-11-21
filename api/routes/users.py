from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from db.session import get_db
from db.crud import create_user, get_user_by_email
from api.schemas.user import UserCreate, User
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Check if user already exists
        db_user = get_user_by_email(db, email=user.email)
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
            
        # Create new user
        new_user = create_user(db=db, user=user)
        return new_user
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 