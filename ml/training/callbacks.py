from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
from pathlib import Path
import logging
import time
from ..monitoring.custom_metrics import MetricsCollector

class TrainingCallback:
    """Base class for training callbacks"""
    def __call__(
        self,
        batch_idx: int,
        epoch: int,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor,
        loss: torch.Tensor
    ):
        raise NotImplementedError

class ModelCheckpoint(TrainingCallback):
    """Save model checkpoints during training"""
    def __init__(
        self,
        checkpoint_dir: str,
        save_freq: int = 1,
        save_best_only: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(
        self,
        batch_idx: int,
        epoch: int,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor,
        loss: torch.Tensor
    ):
        if (epoch + 1) % self.save_freq == 0:
            self._save_checkpoint(epoch, loss.item())
    
    def _save_checkpoint(self, epoch: int, monitor_value: float):
        """Save model checkpoint"""
        if self.save_best_only:
            if self._is_better(monitor_value):
                self.best_value = monitor_value
                self._save(epoch, monitor_value, is_best=True)
        else:
            self._save(epoch, monitor_value)
    
    def _is_better(self, value: float) -> bool:
        """Check if current value is better than best"""
        if self.mode == 'min':
            return value < self.best_value
        return value > self.best_value
    
    def _save(self, epoch: int, monitor_value: float, is_best: bool = False):
        """Save model state"""
        filename = f"checkpoint_epoch_{epoch}"
        if is_best:
            filename += "_best"
        filename += f"_{self.monitor}_{monitor_value:.4f}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            f'{self.monitor}': monitor_value
        }, self.checkpoint_dir / filename)

class ProgressCallback(TrainingCallback):
    """Log training progress"""
    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq
        self.metrics_collector = MetricsCollector()
        self.start_time = time.time()
    
    def __call__(
        self,
        batch_idx: int,
        epoch: int,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor,
        loss: torch.Tensor
    ):
        if batch_idx % self.log_freq == 0:
            elapsed = time.time() - self.start_time
            logging.info(
                f"Epoch {epoch} [{batch_idx}] - "
                f"Loss: {loss.item():.4f} - "
                f"Time: {elapsed:.2f}s"
            )
            
            # Log metrics
            self.metrics_collector.record_training_metric(
                metric_name='batch_loss',
                value=loss.item()
            )

class LearningRateSchedulerCallback(TrainingCallback):
    """Adjust learning rate during training"""
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode
    
    def __call__(
        self,
        batch_idx: int,
        epoch: int,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor,
        loss: torch.Tensor
    ):
        # Step the scheduler
        if isinstance(
            self.scheduler,
            (torch.optim.lr_scheduler.ReduceLROnPlateau)
        ):
            self.scheduler.step(loss.item())
        else:
            self.scheduler.step()

class GradientMonitorCallback(TrainingCallback):
    """Monitor gradient statistics during training"""
    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.metrics_collector = MetricsCollector()
    
    def __call__(
        self,
        batch_idx: int,
        epoch: int,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor,
        loss: torch.Tensor
    ):
        if batch_idx % self.log_freq == 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    self.metrics_collector.record_training_metric(
                        metric_name=f'grad_norm_{name}',
                        value=grad_norm
                    ) 