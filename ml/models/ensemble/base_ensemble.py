from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from ..base import BaseModel
import numpy as np
from abc import ABC, abstractmethod

class BaseEnsemble(BaseModel, ABC):
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
    @abstractmethod
    def aggregate_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate predictions from multiple models"""
        pass
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        predictions = []
        for model in self.models:
            pred = model.predict(input_data)
            predictions.append(pred)
        
        return self.aggregate_predictions(predictions)
    
    def save(self, path: str):
        ensemble_state = {
            'weights': self.weights,
            'model_states': []
        }
        
        for i, model in enumerate(self.models):
            model_path = f"{path}_model_{i}"
            model.save(model_path)
            ensemble_state['model_states'].append(model_path)
            
        torch.save(ensemble_state, f"{path}_ensemble")
    
    def load(self, path: str):
        ensemble_state = torch.load(f"{path}_ensemble")
        self.weights = ensemble_state['weights']
        
        for i, model_path in enumerate(ensemble_state['model_states']):
            self.models[i].load(model_path) 