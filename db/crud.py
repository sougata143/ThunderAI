from sqlalchemy.orm import Session
from . import models
from typing import Dict, Any, List, Optional
from datetime import datetime
from db.models import User
from api.schemas.user import UserCreate
import bcrypt

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

user = CRUDUser(models.User)
prediction = CRUDPrediction(models.Prediction) 