from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
from api.auth.jwt import verify_token
from ml.versioning.model_registry import ModelRegistry
from ml.monitoring.metrics import MODEL_TRAINING_TIME
from db.crud import create_model_record
import time

router = APIRouter()
model_registry = ModelRegistry(tracking_uri="sqlite:///mlflow.db")

@router.post("/models/train/{model_name}")
async def train_model(
    model_name: str,
    training_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    hyperparameters: Optional[Dict[str, Any]] = None,
    token: dict = Depends(verify_token)
):
    async def train_model_task(data: Dict[str, Any], params: Dict[str, Any]):
        start_time = time.time()
        try:
            # Initialize and train model
            model = get_model_class(model_name)(params)
            metrics = model.train(data)
            
            # Register model with MLflow
            model_info = model_registry.register_model(
                model=model,
                name=model_name,
                metrics=metrics
            )
            
            # Record training time
            training_time = time.time() - start_time
            MODEL_TRAINING_TIME.labels(
                model_name=model_name,
                version=model_info.version
            ).observe(training_time)
            
            # Create database record
            create_model_record(
                name=model_name,
                version=model_info.version,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {str(e)}")
            raise
    
    background_tasks.add_task(
        train_model_task,
        training_data,
        hyperparameters or {}
    )
    
    return {"message": "Model training started", "model_name": model_name}

@router.get("/models/{model_name}/versions")
async def list_model_versions(
    model_name: str,
    token: dict = Depends(verify_token)
):
    return model_registry.list_versions(model_name)

@router.post("/models/{model_name}/deploy/{version}")
async def deploy_model(
    model_name: str,
    version: str,
    token: dict = Depends(verify_token)
):
    return model_registry.deploy_model(model_name, version) 