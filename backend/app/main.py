from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings, close_database
from app.api.v1.api import api_router
from app.db.mongodb import db

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Initialize database connection on startup
    """
    await db.connect_to_database()

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown
    """
    await db.close_database_connection()
    await close_database()

app.include_router(api_router, prefix=settings.API_V1_STR)
