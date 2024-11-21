from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import json
from ..versioning.model_registry import ModelRegistry
from ...monitoring.custom_metrics import MetricsCollector
import logging

class ABTest:
    def __init__(
        self,
        experiment_id: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        metrics: List[str] = ["accuracy", "latency"]
    ):
        self.experiment_id = experiment_id
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.metrics = metrics
        self.results = {
            "model_a": {"predictions": 0, "metrics": {}},
            "model_b": {"predictions": 0, "metrics": {}}
        }
        self.start_time = datetime.utcnow()
        
    def select_model(self) -> str:
        """Randomly select model based on traffic split"""
        return self.model_a if np.random.random() < self.traffic_split else self.model_b
    
    def record_prediction(
        self,
        model: str,
        prediction: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """Record prediction and metrics for the selected model"""
        model_key = "model_a" if model == self.model_a else "model_b"
        self.results[model_key]["predictions"] += 1
        
        for metric, value in metrics.items():
            if metric not in self.results[model_key]["metrics"]:
                self.results[model_key]["metrics"][metric] = []
            self.results[model_key]["metrics"][metric].append(value)
    
    def get_results(self) -> Dict[str, Any]:
        """Calculate and return experiment results"""
        results = {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "end_time": datetime.utcnow(),
            "models": {}
        }
        
        for model_key in ["model_a", "model_b"]:
            metrics = self.results[model_key]["metrics"]
            results["models"][model_key] = {
                "total_predictions": self.results[model_key]["predictions"],
                "metrics": {
                    metric: {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }
                    for metric, values in metrics.items()
                }
            }
        
        return results

class ExperimentManager:
    def __init__(self):
        self.active_experiments: Dict[str, ABTest] = {}
        self.model_registry = ModelRegistry()
        self.metrics_collector = MetricsCollector()
    
    def create_experiment(
        self,
        experiment_id: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        metrics: List[str] = ["accuracy", "latency"]
    ) -> ABTest:
        """Create and start a new A/B test experiment"""
        if experiment_id in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")
            
        experiment = ABTest(
            experiment_id=experiment_id,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            metrics=metrics
        )
        
        self.active_experiments[experiment_id] = experiment
        logging.info(f"Experiment {experiment_id} created.")
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[ABTest]:
        """Get active experiment by ID"""
        return self.active_experiments.get(experiment_id)
    
    def end_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """End experiment and return results"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.active_experiments.pop(experiment_id)
        results = experiment.get_results()
        
        # Store results
        self._store_experiment_results(experiment_id, results)
        
        logging.info(f"Experiment {experiment_id} ended. Results: {results}")
        return results
    
    def _store_experiment_results(self, experiment_id: str, results: Dict[str, Any]):
        """Store experiment results in a persistent storage"""
        # Implement storage logic (e.g., save to a database or file)
        logging.info(f"Storing results for experiment {experiment_id}.")