from fastapi import FastAPI
from api.routes import models, predictions, users
from core.config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with the API version prefix
app.include_router(
    models.router,
    prefix=settings.API_V1_STR,
    tags=["models"]
)
app.include_router(
    predictions.router,
    prefix=settings.API_V1_STR,
    tags=["predictions"]
)
app.include_router(
    users.router,
    prefix=settings.API_V1_STR,
    tags=["users"]
)

@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to ThunderAI API",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION
    }