from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging
from ..base import BaseModel

class TransferLearningConfig:
    def __init__(
        self,
        base_model_name: str,
        num_frozen_layers: int = -1,
        fine_tune_embeddings: bool = False,
        adapter_size: Optional[int] = None,
        learning_rate: float = 2e-5,
        layer_lr_decay: float = 0.95
    ):
        self.base_model_name = base_model_name
        self.num_frozen_layers = num_frozen_layers
        self.fine_tune_embeddings = fine_tune_embeddings
        self.adapter_size = adapter_size
        self.learning_rate = learning_rate
        self.layer_lr_decay = layer_lr_decay

class AdapterLayer(nn.Module):
    def __init__(self, input_size: int, adapter_size: int):
        super().__init__()
        self.down_project = nn.Linear(input_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        x = x + residual
        return self.layer_norm(x)

class TransferLearningHelper:
    @staticmethod
    def freeze_layers(model: PreTrainedModel, num_layers: int = -1):
        """Freeze layers of the model starting from bottom"""
        if num_layers == -1:
            # Freeze all layers except the classifier
            for param in model.base_model.parameters():
                param.requires_grad = False
        else:
            # Freeze specific number of layers
            for i, layer in enumerate(model.base_model.encoder.layer):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    @staticmethod
    def add_adapters(model: PreTrainedModel, adapter_size: int):
        """Add adapter layers to the model"""
        for i, layer in enumerate(model.base_model.encoder.layer):
            hidden_size = layer.output.dense.out_features
            adapter = AdapterLayer(hidden_size, adapter_size)
            setattr(layer, 'adapter', adapter)
            
            # Modify the forward pass to include adapter
            original_forward = layer.forward
            def new_forward(self, *args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                if hasattr(self, 'adapter'):
                    outputs = self.adapter(outputs)
                return outputs
            layer.forward = new_forward.__get__(layer)

    @staticmethod
    def get_layer_wise_learning_rates(
        model: PreTrainedModel,
        learning_rate: float,
        decay_factor: float
    ) -> List[Dict[str, float]]:
        """Generate layer-wise learning rates with decay"""
        parameter_groups = []
        num_layers = len(model.base_model.encoder.layer)
        
        # Handle embeddings
        parameter_groups.append({
            'params': model.base_model.embeddings.parameters(),
            'lr': learning_rate * (decay_factor ** num_layers)
        })
        
        # Handle encoder layers
        for i, layer in enumerate(reversed(model.base_model.encoder.layer)):
            parameter_groups.append({
                'params': layer.parameters(),
                'lr': learning_rate * (decay_factor ** i)
            })
            
        # Handle classifier
        parameter_groups.append({
            'params': model.classifier.parameters(),
            'lr': learning_rate
        })
        
        return parameter_groups

    @staticmethod
    def apply_transfer_learning(
        model: PreTrainedModel,
        config: TransferLearningConfig
    ) -> PreTrainedModel:
        """Apply transfer learning configurations to the model"""
        # Freeze layers if specified
        if config.num_frozen_layers >= 0:
            TransferLearningHelper.freeze_layers(model, config.num_frozen_layers)
        
        # Add adapters if specified
        if config.adapter_size is not None:
            TransferLearningHelper.add_adapters(model, config.adapter_size)
        
        # Handle embeddings
        if not config.fine_tune_embeddings:
            for param in model.base_model.embeddings.parameters():
                param.requires_grad = False
        
        # Set up layer-wise learning rates
        parameter_groups = TransferLearningHelper.get_layer_wise_learning_rates(
            model,
            config.learning_rate,
            config.layer_lr_decay
        )
        
        return model, parameter_groups

class TransferLearningModel(BaseModel):
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: TransferLearningConfig
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Apply transfer learning configurations
        self.model, self.parameter_groups = TransferLearningHelper.apply_transfer_learning(
            base_model,
            config
        )
        
        # Set up optimizer with layer-wise learning rates
        self.optimizer = torch.optim.AdamW(self.parameter_groups)
        
    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        
        for batch in data:
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
        
        return {"loss": total_loss}
    
    def predict(self, inputs: Union[str, List[str]]) -> Dict[str, Any]:
        self.model.eval()
        
        # Tokenize inputs
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        return {
            "logits": logits.tolist(),
            "probabilities": probabilities.tolist()
        } 