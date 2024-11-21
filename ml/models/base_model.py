import torch
import torch.nn as nn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented by subclass")
        
    def train_step(self, batch):
        raise NotImplementedError("Train step must be implemented by subclass")
        
    def validate_step(self, batch):
        raise NotImplementedError("Validate step must be implemented by subclass")
        
    def save(self, path: str):
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': self.config
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load(self, path: str):
        try:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.config = checkpoint['config']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 