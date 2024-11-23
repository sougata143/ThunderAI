"""
Temporary mock service for experiments.
Database functionality is currently disabled for simplification.
"""

from typing import Dict, List, Optional

# Mock storage
experiments: Dict[int, Dict] = {
    1: {"id": 1, "name": "BERT Classification", "status": "completed", "user_id": 1},
    2: {"id": 2, "name": "GPT Training", "status": "in_progress", "user_id": 1}
}

async def get_experiment(experiment_id: int) -> Optional[Dict]:
    return experiments.get(experiment_id)

async def get_experiments() -> List[Dict]:
    return list(experiments.values())

async def create_experiment(experiment_data: Dict) -> Dict:
    new_id = max(experiments.keys()) + 1
    experiment_data["id"] = new_id
    experiments[new_id] = experiment_data
    return experiment_data

async def update_experiment(experiment_id: int, experiment_data: Dict) -> Optional[Dict]:
    if experiment_id not in experiments:
        return None
    experiment_data["id"] = experiment_id
    experiments[experiment_id] = experiment_data
    return experiment_data