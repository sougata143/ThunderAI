from fastapi import APIRouter, Depends, HTTPException
from app.schemas.settings import Settings, SettingsUpdate
from app.services.settings_service import SettingsService
from app.api.deps import get_current_active_user
from app.schemas.user import User

router = APIRouter()

@router.get("/", response_model=Settings)
async def get_settings(
    current_user: User = Depends(get_current_active_user),
    settings_service: SettingsService = Depends()
):
    """
    Retrieve user settings.
    """
    return await settings_service.get_settings()

@router.put("/", response_model=Settings)
async def update_settings(
    settings_update: SettingsUpdate,
    current_user: User = Depends(get_current_active_user),
    settings_service: SettingsService = Depends()
):
    """
    Update user settings.
    """
    return await settings_service.update_settings(settings_update)
