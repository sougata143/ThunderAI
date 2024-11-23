from fastapi import APIRouter, HTTPException
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime

class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None
    model_type: str
    status: str = "pending"

class Experiment(ExperimentCreate):
    id: int
    created_at: datetime
    updated_at: datetime

router = APIRouter(
    tags=["Experiments"],
    responses={404: {"description": "Not found"}}
)

# Mock data for experiments
EXPERIMENTS = {
    1: {
        "id": 1,
        "name": "BERT Classification",
        "description": "Text classification using BERT",
        "model_type": "nlp",
        "status": "completed",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    },
    2: {
        "id": 2,
        "name": "GPT Training",
        "description": "Fine-tuning GPT model",
        "model_type": "nlp",
        "status": "in_progress",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
}

@router.get("/", response_model=List[Dict])
async def list_experiments() -> List[Dict]:
    """List all experiments."""
    return list(EXPERIMENTS.values())

@router.get("/{experiment_id}", response_model=Dict)
async def get_experiment(experiment_id: int) -> Dict:
    """Get experiment details by ID."""
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return EXPERIMENTS[experiment_id]

@router.post("/", response_model=Dict)
async def create_experiment(experiment: ExperimentCreate) -> Dict:
    """Create a new experiment."""
    new_id = max(EXPERIMENTS.keys()) + 1 if EXPERIMENTS else 1
    current_time = datetime.now()
    
    new_experiment = {
        "id": new_id,
        **experiment.dict(),
        "created_at": current_time,
        "updated_at": current_time
    }
    
    EXPERIMENTS[new_id] = new_experiment
    return new_experiment