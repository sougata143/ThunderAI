from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Train the model and return metrics"""
        pass
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk"""
        pass 