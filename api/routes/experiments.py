from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from db.session import get_db
from api.auth.jwt import verify_token
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active experiments with some initial sample data
active_experiments: Dict[str, Dict[str, Any]] = {
    str(uuid.uuid4()): {
        "name": "BERT Classification",
        "model_type": "bert",
        "status": "completed",
        "metrics": {
            "loss": [0.8, 0.6, 0.4, 0.3, 0.2],
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
            "epochs": [1, 2, 3, 4, 5]
        },
        "training_time": 15
    },
    str(uuid.uuid4()): {
        "name": "GPT Text Generation",
        "model_type": "gpt",
        "status": "running",
        "metrics": {
            "loss": [0.9, 0.7, 0.5],
            "accuracy": [0.5, 0.65, 0.75],
            "epochs": [1, 2, 3]
        },
        "training_time": 10
    },
    str(uuid.uuid4()): {
        "name": "LSTM Sequence Model",
        "model_type": "lstm",
        "status": "failed",
        "metrics": {
            "loss": [1.0, 0.8],
            "accuracy": [0.4, 0.5],
            "epochs": [1, 2]
        },
        "training_time": 5
    }
}

@router.get("/experiments")
async def get_experiments(
    token: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get all experiments"""
    try:
        experiments = [
            {
                "id": exp_id,
                "name": exp["name"],
                "modelType": exp["model_type"],
                "status": exp["status"],
                "metrics": exp["metrics"],
                "trainingTime": exp["training_time"]
            }
            for exp_id, exp in active_experiments.items()
        ]
        return {"data": experiments}
    except Exception as e:
        logger.error(f"Error fetching experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    token: dict = Depends(verify_token)
):
    """Start an experiment"""
    if experiment_id not in active_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    try:
        experiment = active_experiments[experiment_id]
        if experiment["status"] == "completed":
            raise HTTPException(
                status_code=400, 
                detail="Cannot start a completed experiment"
            )
            
        experiment["status"] = "running"
        return {"message": "Experiment started successfully"}
    except Exception as e:
        logger.error(f"Error starting experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    token: dict = Depends(verify_token)
):
    """Stop an experiment"""
    if experiment_id not in active_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    try:
        experiment = active_experiments[experiment_id]
        if experiment["status"] != "running":
            raise HTTPException(
                status_code=400, 
                detail="Experiment is not running"
            )
            
        experiment["status"] = "stopped"
        return {"message": "Experiment stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    token: dict = Depends(verify_token)
):
    """Delete an experiment"""
    if experiment_id not in active_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    try:
        experiment = active_experiments[experiment_id]
        if experiment["status"] == "running":
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete a running experiment"
            )
            
        del active_experiments[experiment_id]
        return {"message": "Experiment deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/export")
async def export_experiment_results(
    experiment_id: str,
    token: dict = Depends(verify_token)
):
    """Export experiment results"""
    if experiment_id not in active_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    try:
        experiment = active_experiments[experiment_id]
        # Here you would format the results for export
        # For now, we'll just return the metrics
        return experiment["metrics"]
    except Exception as e:
        logger.error(f"Error exporting experiment results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 