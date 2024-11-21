from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from .utils import MetricTracker
from ..monitoring.custom_metrics import MetricsCollector

class MixedPrecisionTraining:
    """Implements mixed precision training using torch.cuda.amp"""
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        
    def forward_backward(self, batch: Dict[str, torch.Tensor], criterion: nn.Module) -> torch.Tensor:
        """Perform forward and backward passes with mixed precision"""
        with torch.cuda.amp.autocast():
            outputs = self.model(**batch)
            loss = criterion(outputs, batch['labels'])
        
        self.scaler.scale(loss).backward()
        return loss, outputs
    
    def optimizer_step(self):
        """Perform optimizer step with gradient scaling"""
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

class DistributedTraining:
    """Implements distributed training across multiple GPUs"""
    def __init__(
        self,
        model: nn.Module,
        device_ids: List[int],
        output_device: Optional[int] = None
    ):
        self.model = nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device
        )
        self.device_ids = device_ids
        
    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for distributed training"""
        return {k: v.to(self.device_ids[0]) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}

class CurriculumLearning:
    """Implements curriculum learning strategy"""
    def __init__(
        self,
        difficulty_fn: Callable[[Dict[str, torch.Tensor]], float],
        num_stages: int,
        epochs_per_stage: int
    ):
        self.difficulty_fn = difficulty_fn
        self.num_stages = num_stages
        self.epochs_per_stage = epochs_per_stage
        self.current_stage = 0
        self.metrics_collector = MetricsCollector()
        
    def should_advance_stage(self, epoch: int) -> bool:
        """Check if we should advance to the next stage"""
        if self.current_stage >= self.num_stages - 1:
            return False
        return epoch > 0 and epoch % self.epochs_per_stage == 0
    
    def filter_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Filter batch based on current difficulty stage"""
        difficulties = self.difficulty_fn(batch)
        threshold = (self.current_stage + 1) / self.num_stages
        mask = difficulties <= threshold
        
        return {k: v[mask] if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}

class AdversarialTraining:
    """Implements adversarial training strategy"""
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        num_steps: int = 3
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        
    def generate_adversarial_examples(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Generate adversarial examples using PGD attack"""
        x = batch['input_ids'].clone().detach()
        x.requires_grad = True
        
        for _ in range(self.num_steps):
            outputs = self.model(input_ids=x, attention_mask=batch['attention_mask'])
            loss = criterion(outputs, batch['labels'])
            
            grad = torch.autograd.grad(loss, x)[0]
            perturbed = x + self.alpha * grad.sign()
            x = torch.min(
                torch.max(perturbed, x - self.epsilon),
                x + self.epsilon
            )
        
        return {**batch, 'input_ids': x.detach()}

class StrategyManager:
    """Manages different training strategies"""
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any]
    ):
        self.model = model
        self.config = config
        self.strategies = {}
        self.metric_tracker = MetricTracker()
        
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize requested training strategies"""
        if self.config.get('mixed_precision', False):
            self.strategies['mixed_precision'] = MixedPrecisionTraining(
                self.model,
                self.config['optimizer']
            )
        
        if self.config.get('distributed', False):
            self.strategies['distributed'] = DistributedTraining(
                self.model,
                device_ids=self.config.get('device_ids', [0])
            )
        
        if self.config.get('curriculum', False):
            self.strategies['curriculum'] = CurriculumLearning(
                difficulty_fn=self.config['difficulty_fn'],
                num_stages=self.config.get('num_stages', 3),
                epochs_per_stage=self.config.get('epochs_per_stage', 2)
            )
        
        if self.config.get('adversarial', False):
            self.strategies['adversarial'] = AdversarialTraining(
                self.model,
                epsilon=self.config.get('epsilon', 0.01),
                alpha=self.config.get('alpha', 0.001),
                num_steps=self.config.get('num_steps', 3)
            )
    
    def apply_strategies(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, Any]:
        """Apply all active training strategies"""
        if 'distributed' in self.strategies:
            batch = self.strategies['distributed'].prepare_batch(batch)
        
        if 'curriculum' in self.strategies:
            if self.strategies['curriculum'].should_advance_stage(epoch):
                self.strategies['curriculum'].current_stage += 1
            batch = self.strategies['curriculum'].filter_batch(batch)
        
        if 'adversarial' in self.strategies:
            adv_batch = self.strategies['adversarial'].generate_adversarial_examples(
                batch,
                criterion
            )
            # Mix clean and adversarial examples
            batch = {k: torch.cat([v, adv_batch[k]]) for k, v in batch.items()}
        
        if 'mixed_precision' in self.strategies:
            loss, outputs = self.strategies['mixed_precision'].forward_backward(
                batch,
                criterion
            )
            self.strategies['mixed_precision'].optimizer_step()
        else:
            outputs = self.model(**batch)
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            self.config['optimizer'].step()
            self.config['optimizer'].zero_grad()
        
        return {
            'loss': loss,
            'outputs': outputs,
            'batch': batch
        } 