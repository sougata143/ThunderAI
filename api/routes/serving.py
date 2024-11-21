from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ..auth.jwt import verify_token
from ...ml.versioning.model_registry import ModelRegistry
from ...monitoring.custom_metrics import MetricsCollector

router = APIRouter()
model_registry = ModelRegistry()
metrics_collector = MetricsCollector()

@router.post("/serve/{model_name}/{version}")
async def serve_model(
    model_name: str,
    version: str,
    input_data: Dict[str, Any],
    token: dict = Depends(verify_token)
):
    try:
        # Load model
        model = model_registry.load_model(model_name, version)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Record metrics
        metrics_collector.record_prediction(
            model_name=model_name,
            version=version,
            prediction_class=str(prediction["prediction"]),
            latency=prediction.get("latency", 0)
        )
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 