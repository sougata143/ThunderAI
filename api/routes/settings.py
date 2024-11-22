from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
from core.model import UserSettings, UserSettingsResponse, UserSettingsDB
from core.auth import get_current_user
from core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/settings/{user_id}", response_model=UserSettingsResponse)
async def get_user_settings(
    user_id: str, 
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> UserSettingsResponse:
    try:
        settings = db.query(UserSettingsDB).filter(UserSettingsDB.user_id == user_id).first()
        if not settings:
            settings = UserSettingsDB(user_id=user_id)
            db.add(settings)
            db.commit()
            db.refresh(settings)
        return settings
    except Exception as e:
        raise HTTPException(status_code=404, detail="Settings not found")

@router.post("/settings/{user_id}", response_model=UserSettingsResponse)
async def update_user_settings(
    user_id: str, 
    settings: UserSettings,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> UserSettingsResponse:
    try:
        db_settings = db.query(UserSettingsDB).filter(UserSettingsDB.user_id == user_id).first()
        if not db_settings:
            db_settings = UserSettingsDB(user_id=user_id)
            db.add(db_settings)
        
        for key, value in settings.dict().items():
            setattr(db_settings, key, value)
        
        db.commit()
        db.refresh(db_settings)
        return db_settings
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to update settings") 