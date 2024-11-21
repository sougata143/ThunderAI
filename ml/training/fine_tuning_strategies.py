from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from ..utils.fine_tuning import FineTuningConfig

class LayerwiseFineTuning:
    """Implements layerwise fine-tuning with gradual unfreezing"""
    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig,
        layer_groups: List[List[str]]
    ):
        self.model = model
        self.config = config
        self.layer_groups = layer_groups
        self.current_group = 0
        
    def step(self):
        """Unfreeze next layer group"""
        if self.current_group < len(self.layer_groups):
            # Freeze all layers first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze layers up to current group
            for group_idx in range(self.current_group + 1):
                for layer_name in self.layer_groups[group_idx]:
                    for name, param in self.model.named_parameters():
                        if layer_name in name:
                            param.requires_grad = True
            
            self.current_group += 1
            return True
        return False

class DiscriminativeFinetuning:
    """Implements discriminative fine-tuning with different learning rates"""
    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        layer_groups: List[List[str]],
        multiplier: float = 0.9
    ):
        self.model = model
        self.base_lr = base_lr
        self.layer_groups = layer_groups
        self.multiplier = multiplier
        
    def get_layer_lrs(self) -> List[Dict[str, float]]:
        """Get learning rates for each layer group"""
        param_groups = []
        curr_lr = self.base_lr
        
        for group in reversed(self.layer_groups):
            params = []
            for layer_name in group:
                for name, param in self.model.named_parameters():
                    if layer_name in name:
                        params.append(param)
            
            if params:
                param_groups.append({
                    'params': params,
                    'lr': curr_lr
                })
            
            curr_lr *= self.multiplier
        
        return param_groups

class GradualUnfreezing:
    """Implements gradual unfreezing with custom schedules"""
    def __init__(
        self,
        model: nn.Module,
        total_steps: int,
        layer_groups: List[List[str]],
        schedule: Optional[Callable[[int, int], float]] = None
    ):
        self.model = model
        self.total_steps = total_steps
        self.layer_groups = layer_groups
        self.schedule = schedule or self._default_schedule
        self.current_step = 0
        
    def _default_schedule(self, step: int, total_steps: int) -> float:
        """Default linear unfreezing schedule"""
        return step / total_steps
    
    def step(self):
        """Update unfreezing based on current step"""
        if self.current_step < self.total_steps:
            progress = self.schedule(self.current_step, self.total_steps)
            groups_to_unfreeze = int(progress * len(self.layer_groups))
            
            # Freeze all layers first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze layers according to schedule
            for group_idx in range(groups_to_unfreeze):
                for layer_name in self.layer_groups[group_idx]:
                    for name, param in self.model.named_parameters():
                        if layer_name in name:
                            param.requires_grad = True
            
            self.current_step += 1
            return True
        return False

class FineTuningManager:
    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig,
        strategy: str = 'layerwise'
    ):
        self.model = model
        self.config = config
        
        # Define layer groups (can be customized based on model architecture)
        self.layer_groups = self._get_layer_groups()
        
        # Initialize strategy
        if strategy == 'layerwise':
            self.strategy = LayerwiseFineTuning(model, config, self.layer_groups)
        elif strategy == 'discriminative':
            self.strategy = DiscriminativeFinetuning(
                model,
                config.learning_rate,
                self.layer_groups
            )
        elif strategy == 'gradual':
            self.strategy = GradualUnfreezing(
                model,
                config.num_training_steps,
                self.layer_groups
            )
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {strategy}")
    
    def _get_layer_groups(self) -> List[List[str]]:
        """Get layer groups for the model"""
        # This can be customized based on model architecture
        return [
            ['embeddings'],
            ['encoder.layer.0', 'encoder.layer.1'],
            ['encoder.layer.2', 'encoder.layer.3'],
            ['encoder.layer.4', 'encoder.layer.5'],
            ['pooler', 'classifier']
        ]
    
    def get_optimizer(self) -> Optimizer:
        """Get optimizer with appropriate parameter groups"""
        if isinstance(self.strategy, DiscriminativeFinetuning):
            param_groups = self.strategy.get_layer_lrs()
        else:
            param_groups = [{'params': self.model.parameters()}]
        
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def step(self):
        """Perform one step of the fine-tuning strategy"""
        if hasattr(self.strategy, 'step'):
            return self.strategy.step()
        return False 