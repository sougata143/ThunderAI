import mlflow
from typing import Dict, Any
import os

class ModelRegistry:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        
    def register_model(self, model: Any, name: str, metrics: Dict[str, float]):
        """Register trained model with metrics"""
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            
            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(model, name)
            elif isinstance(model, tf.keras.Model):
                mlflow.tensorflow.log_model(model, name)
                
            return mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{name}", name)
    
    def load_model(self, name: str, version: str = "latest"):
        model_uri = f"models:/{name}/{version}"
        return mlflow.pyfunc.load_model(model_uri)
    
    def get_model_metrics(self, name: str, version: str = "latest"):
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(name, version)
        run = client.get_run(model_version.run_id)
        return run.data.metrics 