from fastapi import APIRouter
from .endpoints import experiments, users, llm, auth

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
