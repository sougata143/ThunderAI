from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from core.auth import get_current_user
from core.model import UserInDB

router = APIRouter()

class Experiment(BaseModel):
    id: str
    name: str
    modelType: str
    status: str
    created_at: datetime
    user_id: Optional[int] = None

@router.get("/experiments")
async def get_experiments(
    current_user: Optional[UserInDB] = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get list of experiments"""
    try:
        # For now, return mock data
        experiments = [
            {
                "id": "exp1",
                "name": "BERT Fine-tuning",
                "modelType": "bert",
                "status": "completed",
                "created_at": datetime.now(),
                "user_id": current_user.id if current_user else None
            },
            {
                "id": "exp2",
                "name": "GPT Training",
                "modelType": "gpt",
                "status": "running",
                "created_at": datetime.now(),
                "user_id": current_user.id if current_user else None
            }
        ]
        
        # If user is authenticated, filter experiments by user_id
        if current_user:
            experiments = [exp for exp in experiments if exp["user_id"] == current_user.id]
            
        return experiments
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch experiments: {str(e)}"
        )

@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    current_user: Optional[UserInDB] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get experiment details"""
    try:
        # Mock data for now
        experiment = {
            "id": experiment_id,
            "name": "BERT Fine-tuning",
            "modelType": "bert",
            "status": "completed",
            "created_at": datetime.now(),
            "user_id": current_user.id if current_user else None,
            "metrics": {
                "accuracy": 0.95,
                "loss": 0.05
            }
        }
        
        # Check if user has access to this experiment
        if current_user and experiment["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this experiment"
            )
            
        return experiment
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch experiment: {str(e)}"
        ) 