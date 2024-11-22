from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

Base = declarative_base()

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    full_name = Column(String)
    organization = Column(String)
    job_title = Column(String)
    settings = Column(String)

# Pydantic Models
class UserBase(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    organization: Optional[str] = None
    job_title: Optional[str] = None
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)

class UserInDB(UserBase):
    id: int
    hashed_password: str
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class UserSettingsDB(Base):
    __tablename__ = "user_settings"

    user_id = Column(Integer, primary_key=True)
    model_update_alerts = Column(Boolean, default=True)
    performance_alerts = Column(Boolean, default=True)
    collaboration_notifications = Column(Boolean, default=True)

# Pydantic Models for API
class UserSettings(BaseModel):
    model_update_alerts: bool = True
    performance_alerts: bool = True
    collaboration_notifications: bool = True

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=()
    )

class UserSettingsResponse(BaseModel):
    user_id: int
    model_update_alerts: bool
    performance_alerts: bool
    collaboration_notifications: bool

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=()
    ) 