from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the PyTorch model architecture."""
        pass
    
    @abstractmethod
    def preprocess_data(self, data: Any) -> torch.Tensor:
        """Preprocess input data into model-ready format."""
        pass
    
    @abstractmethod
    def postprocess_output(self, output: torch.Tensor) -> Any:
        """Convert model output into desired format."""
        pass
    
    def save_model(self, path: str) -> None:
        """Save model weights and configuration."""
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        
        save_dict = {
            'model_state': self.model.state_dict(),
            'config': self.config
        }
        torch.save(save_dict, path)
    
    def load_model(self, path: str) -> None:
        """Load model weights and configuration."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)
    
    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_size(self) -> float:
        """Return the model size in MB."""
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024  # Convert to MB
