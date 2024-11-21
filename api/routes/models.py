from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, WebSocket
from typing import Dict, Any, Optional, List
from api.auth.jwt import verify_token
from ml.versioning.model_registry import ModelRegistry
from ml.monitoring.metrics import MODEL_TRAINING_TIME
from db.crud import create_model_record
from ml.models.bert import BERTModel
from api.schemas.training import TrainingData, TrainingConfig, TrainingResponse, TrainingStatus
import time
import logging
import uuid
import asyncio
from ml.training.pipeline_manager import ModelTrainingPipeline
from core.config import settings

router = APIRouter()
model_registry = ModelRegistry(tracking_uri="sqlite:///mlflow.db")
logger = logging.getLogger(__name__)

# Store active training sessions
active_trainings: Dict[str, ModelTrainingPipeline] = {}

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

@router.post("/train/bert", status_code=status.HTTP_200_OK)
async def train_bert_model(
    training_data: TrainingData,
    token_data: Dict = Depends(verify_token)
):
    """Train a BERT model with the provided data"""
    try:
        # Initialize model
        config = {
            "max_length": 512,
            "num_labels": 2,
            "batch_size": 32
        }
        model = BERTModel(config)
        
        # Convert Pydantic model to dict for training
        train_data_dict = training_data.model_dump()
        
        # Train model
        metrics = model.train(train_data_dict)
        
        # Save model (you might want to add a unique identifier)
        model.save("models/bert_latest.pt")
        
        return {
            "message": "Model trained successfully",
            "model_name": "bert_latest",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 

@router.post("/models/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start model training"""
    try:
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Initialize training pipeline
        pipeline = ModelTrainingPipeline(
            model_type=config.modelType,
            params=config.params
        )
        
        # Store in active trainings
        active_trainings[model_id] = pipeline
        
        # Start training in background
        background_tasks.add_task(pipeline.train)
        
        return TrainingResponse(
            modelId=model_id,
            status="started",
            message="Training started successfully"
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/stop")
async def stop_training(model_id: str):
    """Stop model training"""
    if model_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    try:
        pipeline = active_trainings[model_id]
        await pipeline.stop()
        return {"status": "stopped", "message": "Training stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/status", response_model=TrainingStatus)
async def get_training_status(model_id: str):
    """Get training status"""
    if model_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    pipeline = active_trainings[model_id]
    return TrainingStatus(
        modelId=model_id,
        status=pipeline.status,
        progress=pipeline.progress,
        currentEpoch=pipeline.current_epoch,
        metrics=pipeline.metrics
    )

@router.websocket("/ws/training/{model_id}")
async def websocket_endpoint(websocket: WebSocket, model_id: str):
    """WebSocket endpoint for real-time training updates"""
    await websocket.accept()
    
    try:
        if model_id not in active_trainings:
            await websocket.close(code=4004, reason="Training session not found")
            return
            
        pipeline = active_trainings[model_id]
        
        # Subscribe to training updates
        async for update in pipeline.get_updates():
            await websocket.send_json(update)
            
    except Exception as e:
        await websocket.close(code=4000, reason=str(e)) 