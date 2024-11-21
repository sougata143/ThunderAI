from fastapi import FastAPI, Depends, HTTPException, status, WebSocket
from api.routes import models, predictions, users, auth, experiments
from core.config import settings
from fastapi.middleware.cors import CORSMiddleware
from .auth.jwt import verify_token

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

# Include routers
app.include_router(
    auth.router,
    prefix=settings.API_V1_STR,
    tags=["auth"]
)

app.include_router(
    users.router,
    prefix=settings.API_V1_STR,
    tags=["users"]
)

app.include_router(
    models.router,
    prefix=settings.API_V1_STR,
    tags=["models"],
    dependencies=[Depends(verify_token)]
)

app.include_router(
    predictions.router,
    prefix=settings.API_V1_STR,
    tags=["predictions"],
    dependencies=[Depends(verify_token)]
)

app.include_router(
    experiments.router,
    prefix=settings.API_V1_STR,
    tags=["experiments"],
    dependencies=[Depends(verify_token)]
)

# Include WebSocket routes
app.include_router(models.router)

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