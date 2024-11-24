from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime
from ....api.deps import get_current_user
from ....models.user import User

class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None
    model: str
    status: str = "pending"

class Experiment(ExperimentCreate):
    id: int
    created_at: datetime
    updated_at: datetime
    user_id: str

router = APIRouter(
    tags=["Experiments"],
    responses={404: {"description": "Not found"}}
)

# Mock data for experiments (will be replaced with database)
EXPERIMENTS = {}

@router.get("/", response_model=List[Dict])
async def list_experiments(current_user: User = Depends(get_current_user)) -> List[Dict]:
    """List all experiments for the current user."""
    user_experiments = [exp for exp in EXPERIMENTS.values() if exp.get("user_id") == str(current_user.id)]
    return user_experiments

@router.get("/{experiment_id}", response_model=Dict)
async def get_experiment(experiment_id: int, current_user: User = Depends(get_current_user)) -> Dict:
    """Get experiment details by ID."""
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = EXPERIMENTS[experiment_id]
    if experiment.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized to access this experiment")
    
    return experiment

@router.post("/", response_model=Dict)
async def create_experiment(experiment: ExperimentCreate, current_user: User = Depends(get_current_user)) -> Dict:
    """Create a new experiment."""
    new_id = max(EXPERIMENTS.keys()) + 1 if EXPERIMENTS else 1
    current_time = datetime.now()
    
    new_experiment = {
        "id": new_id,
        **experiment.dict(),
        "user_id": str(current_user.id),
        "created_at": current_time,
        "updated_at": current_time
    }
    
    EXPERIMENTS[new_id] = new_experiment
    return new_experiment