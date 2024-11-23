"""
Temporary mock service for models.
Database functionality is currently disabled for simplification.
"""

from typing import Dict, List, Optional

# Mock storage
models: Dict[int, Dict] = {
    1: {
        "id": 1,
        "name": "BERT-base",
        "version": "1.0.0",
        "status": "deployed",
        "experiment_id": 1
    },
    2: {
        "id": 2,
        "name": "GPT-small",
        "version": "0.1.0",
        "status": "training",
        "experiment_id": 2
    }
}

async def get_model(model_id: int) -> Optional[Dict]:
    return models.get(model_id)

async def get_models() -> List[Dict]:
    return list(models.values())

async def create_model(model_data: Dict) -> Dict:
    new_id = max(models.keys()) + 1
    model_data["id"] = new_id
    models[new_id] = model_data
    return model_data

async def update_model(model_id: int, model_data: Dict) -> Optional[Dict]:
    if model_id not in models:
        return None
    model_data["id"] = model_id
    models[model_id] = model_data
    return model_data
