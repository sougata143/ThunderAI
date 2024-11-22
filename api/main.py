from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import models, predictions, users, auth, experiments
from api.routes import settings as settings_routes
from api.middleware import AuthMiddleware
from core.config import settings
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

# Configure CORS - Move this AFTER the auth middleware
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add other origins as needed
]

# Add auth middleware first
app.add_middleware(AuthMiddleware)

# Then add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(experiments.router, prefix="/api/v1", tags=["experiments"])
app.include_router(settings_routes.router, prefix="/api/v1", tags=["settings"])

@app.on_event("startup")
async def startup_event():
    try:
        if settings.USE_CACHE:
            try:
                from core.cache import CacheManager
                await CacheManager.init(app)
                logger.info("Cache initialized successfully")
            except ImportError:
                logger.warning("Cache dependencies not installed. Running without cache.")
            except Exception as e:
                logger.error(f"Failed to initialize cache: {str(e)}")
        else:
            logger.info("Cache disabled by configuration")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        if settings.USE_CACHE:
            try:
                from core.cache import CacheManager
                await CacheManager.close()
                logger.info("Cache connection closed")
            except Exception as e:
                logger.error(f"Error closing cache connection: {str(e)}")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}