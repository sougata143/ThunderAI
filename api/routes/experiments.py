from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ..auth.jwt import verify_token
from ...ml.experiments.ab_testing import ExperimentManager

router = APIRouter()
experiment_manager = ExperimentManager()

@router.post("/experiments/")
async def create_experiment(
    experiment_id: str,
    model_a: str,
    model_b: str,
    traffic_split: float = 0.5,
    token: dict = Depends(verify_token)
):
    try:
        experiment = experiment_manager.create_experiment(
            experiment_id=experiment_id,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split
        )
        return {"message": f"Experiment {experiment_id} created."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    token: dict = Depends(verify_token)
):
    experiment = experiment_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.get_results()

@router.post("/experiments/{experiment_id}/end")
async def end_experiment(
    experiment_id: str,
    token: dict = Depends(verify_token)
):
    try:
        results = experiment_manager.end_experiment(experiment_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) 