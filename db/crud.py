from sqlalchemy.orm import Session
from . import models
from typing import Dict, Any, List, Optional
from datetime import datetime
from db.models import User
from api.schemas.user import UserCreate
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash using bcrypt"""
    return bcrypt.checkpw(
        plain_password.encode(), 
        hashed_password.encode()
    )

class CRUDBase:
    def __init__(self, model):
        self.model = model

    def get(self, db: Session, id: int):
        return db.query(self.model).filter(self.model.id == id).first()
    
    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[Any]:
        return db.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, db: Session, *, obj_in: Dict[str, Any]) -> Any:
        obj = self.model(**obj_in)
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj

class CRUDUser(CRUDBase):
    def get_by_email(self, db: Session, *, email: str):
        return db.query(models.User).filter(models.User.email == email).first()
    
    def create_user(self, db: Session, *, email: str, password: str):
        hashed_password = get_password_hash(password)
        user = models.User(
            email=email,
            hashed_password=hashed_password
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

class CRUDPrediction(CRUDBase):
    def create_prediction(
        self,
        db: Session,
        *,
        user_id: int,
        model_id: int,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        confidence: float
    ):
        prediction = models.Prediction(
            user_id=user_id,
            model_id=model_id,
            input_data=input_data,
            output=output,
            confidence=confidence,
            created_at=datetime.utcnow()
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction

def create_model_record(db: Session, name: str, version: str, metrics: dict):
    db_model = Model(name=name, version=version, metrics=metrics)
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        is_active=True
    )
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        db.rollback()
        raise e

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_user(db: Session, user_data: UserCreate) -> User:
    """Create new user"""
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

user = CRUDUser(models.User)
prediction = CRUDPrediction(models.Prediction) 