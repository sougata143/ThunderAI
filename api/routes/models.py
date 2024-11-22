from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, WebSocket, Request
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError
import logging
import uuid
import asyncio
from api.auth.jwt import verify_token
import random
import time
from core.auth import get_current_user
from core.model import UserInDB

# Create router instance
router = APIRouter(
    tags=["models"]
)

logger = logging.getLogger(__name__)

class TrainingParams(BaseModel):
    model_type: str = Field(default="bert")
    optimizer: str = Field(default="adam")
    loss: str = Field(default="categorical_crossentropy")
    metrics: List[str] = Field(default=["accuracy"])
    validation_split: float = Field(default=0.2)
    shuffle: bool = Field(default=True)
    verbose: int = Field(default=1)
    batch_size: int = Field(default=32)
    epochs: int = Field(default=10)
    learning_rate: float = Field(default=0.001)

    class Config:
        protected_namespaces = ()

class TrainingRequest(BaseModel):
    training_params: TrainingParams

# Store active training sessions
active_trainings: Dict[str, Dict[str, Any]] = {}

def get_pipeline(model_type: str, params: Dict[str, Any]):
    from ml.training.pipeline_manager import ModelTrainingPipeline
    return ModelTrainingPipeline(model_type=model_type, params=params)

@router.post("/models/train")
async def start_training(
    training_params: TrainingParams,
    background_tasks: BackgroundTasks,
    current_user: Optional[UserInDB] = None
):
    """Start model training"""
    try:
        logger.info(f"Received training request: {training_params}")
        
        # Convert training params to dict and add user_id
        config_dict = training_params.dict()
        if current_user:
            config_dict["user_id"] = current_user.id
        else:
            config_dict["user_id"] = None
        
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Create training pipeline with config
        pipeline = get_pipeline(
            model_type=config_dict['model_type'],
            params=config_dict
        )
        
        # Store active training
        active_trainings[model_id] = {
            'pipeline': pipeline,
            'status': 'starting',
            'user_id': config_dict["user_id"]
        }
        
        # Start training in background
        background_tasks.add_task(pipeline.train)
        
        return {
            'model_id': model_id,
            'status': 'starting'
        }
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/models/{model_id}/status")
async def get_training_status(model_id: str):
    if model_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = active_trainings[model_id]
    return {
        "status": job.pipeline.status,
        "progress": job.pipeline.progress,
        "metrics": job.pipeline.get_metrics()
    }

@router.post("/models/{model_id}/stop")
async def stop_training(
    model_id: str,
    token: dict = Depends(verify_token)
):
    """Stop a running training job"""
    if model_id not in active_trainings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Training job not found"
        )
    
    try:
        job = active_trainings[model_id]
        pipeline = job['pipeline']
        pipeline._should_stop = True  # Set stop flag
        pipeline.status = "completed"  # Update status
        
        # Wait briefly for training to stop
        await asyncio.sleep(1)
        
        # Update job status instead of removing it
        job['status'] = 'completed'
        
        return {
            "status": "stopped",
            "message": "Training stopped successfully"
        }
    except Exception as e:
        logger.error(f"Failed to stop training job {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop training: {str(e)}"
        )

@router.websocket("/models/ws/training/{model_id}")
async def training_websocket(websocket: WebSocket, model_id: str):
    """WebSocket endpoint for real-time training updates"""
    await websocket.accept()
    
    if model_id not in active_trainings:
        await websocket.send_json({
            "error": "Training job not found"
        })
        await websocket.close()
        return
    
    try:
        # Attach WebSocket to training pipeline
        training_job = active_trainings[model_id]
        training_job['pipeline'].websocket = websocket
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            if data == "stop":
                training_job['pipeline']._should_stop = True
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if model_id in active_trainings:
            active_trainings[model_id]['pipeline'].websocket = None 

@router.post("/models/{model_id}/evaluate")
async def evaluate_model(
    model_id: str,
    evaluation_data: Dict[str, Any],
    token: dict = Depends(verify_token)
) -> Dict[str, Any]:
    """Evaluate a model with provided data"""
    if model_id not in active_trainings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    try:
        job = active_trainings[model_id]
        pipeline = job['pipeline']
        
        # Check if training is completed
        if pipeline.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model training is not completed. Current status: {pipeline.status}"
            )
        
        evaluation_results = await pipeline.evaluate(evaluation_data)
        
        return {
            "model_id": model_id,
            "results": evaluation_results
        }
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )

@router.get("/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: str,
    token: dict = Depends(verify_token)
) -> Dict[str, Any]:
    """Get model training and monitoring metrics"""
    if model_id not in active_trainings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    try:
        pipeline = active_trainings[model_id]['pipeline']
        
        # Get training metrics
        training_metrics = pipeline.get_metrics()
        
        # Add resource usage metrics (simulated)
        resource_metrics = {
            'memory_usage': [random.uniform(20, 80) for _ in range(len(training_metrics))],
            'cpu_usage': [random.uniform(10, 90) for _ in range(len(training_metrics))],
            'timestamp': [time.time() - i * 60 for i in range(len(training_metrics))]
        }
        
        # Add prediction metrics (simulated)
        prediction_metrics = {
            'predictions': [random.uniform(0, 1) for _ in range(len(training_metrics))]
        }
        
        # Combine all metrics
        metrics = []
        for i in range(len(training_metrics)):
            metric_point = {
                **training_metrics[i],
                'memory_usage': resource_metrics['memory_usage'][i],
                'cpu_usage': resource_metrics['cpu_usage'][i],
                'timestamp': resource_metrics['timestamp'][i],
                'predictions': prediction_metrics['predictions'][i]
            }
            metrics.append(metric_point)
        
        return {
            "model_id": model_id,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Failed to get model metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )