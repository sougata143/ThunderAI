from prefect import flow, task
from typing import Dict, Any
from ..versioning.model_registry import ModelRegistry
from ...db.crud import get_training_data
from ...monitoring.custom_metrics import MetricsCollector
from ...schemas.base import ModelCreate
import logging

model_registry = ModelRegistry()
metrics_collector = MetricsCollector()

@task
def fetch_training_data() -> Dict[str, Any]:
    # Fetch training data from the database
    logging.info("Fetching training data...")
    return get_training_data()

@task
def validate_data(data: Dict[str, Any]) -> bool:
    # Validate training data
    logging.info("Validating training data...")
    if not data or "texts" not in data or "labels" not in data:
        logging.error("Invalid training data format.")
        return False
    return True

@task
def train_model(data: Dict[str, Any], model_name: str) -> Dict[str, float]:
    # Initialize and train model
    logging.info(f"Training model {model_name}...")
    model = model_registry.get_model_class(model_name)()
    metrics = model.train(data)
    
    # Register model with MLflow
    model_info = model_registry.register_model(
        model=model,
        name=model_name,
        metrics=metrics
    )
    
    # Update metrics
    metrics_collector.update_model_accuracy(
        model_name=model_name,
        version=model_info.version,
        accuracy=metrics["accuracy"]
    )
    
    return metrics

@task
def notify_completion(model_name: str, metrics: Dict[str, float]):
    # Notify stakeholders about the retraining completion
    logging.info(f"Model {model_name} retraining completed with metrics: {metrics}")

@flow
def retrain_model_pipeline(model_name: str):
    data = fetch_training_data()
    if validate_data(data):
        metrics = train_model(data, model_name)
        notify_completion(model_name, metrics)
    else:
        logging.error("Retraining pipeline aborted due to data validation failure.") 