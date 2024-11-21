from typing import Dict, Any, List, Optional, Type
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ...utils.fine_tuning import FineTuningConfig
from ..base import BaseModel

class CustomArchitecture(nn.Module, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.build_layers()
    
    @abstractmethod
    def build_layers(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        pass
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True

class CustomModelWrapper(BaseModel):
    def __init__(
        self,
        architecture_class: Type[CustomArchitecture],
        config: Dict[str, Any],
        fine_tuning_config: Optional[FineTuningConfig] = None
    ):
        self.config = config
        self.model = architecture_class(config)
        self.fine_tuning_config = fine_tuning_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up optimizer and scheduler
        self.setup_training()
    
    def setup_training(self):
        """Set up training components"""
        if self.fine_tuning_config:
            trainable_params = self.model.get_trainable_params()
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.fine_tuning_config.learning_rate,
                weight_decay=self.fine_tuning_config.weight_decay
            )
            self.scheduler = None  # Add scheduler if needed
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4)
            )
    
    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        
        inputs = torch.tensor(data['inputs']).to(self.device)
        labels = torch.tensor(data['labels']).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        if self.fine_tuning_config:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.fine_tuning_config.max_grad_norm
            )
        
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
            
        return {
            'loss': loss.item()
        }
    
    def predict(self, inputs: Any) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs).to(self.device)
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        return {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy()
        }
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'fine_tuning_config': self.fine_tuning_config
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.fine_tuning_config = checkpoint['fine_tuning_config'] 