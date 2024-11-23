from fastapi import APIRouter
from .endpoints import auth, users, metrics, llm, settings

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(settings.router, prefix="/settings", tags=["settings"])
