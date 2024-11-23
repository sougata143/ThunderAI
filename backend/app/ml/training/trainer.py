import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import logging
from tqdm import tqdm
import time
import numpy as np
from pathlib import Path

from ..models.base import BaseModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model: BaseModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Dict[str, Any] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer or torch.optim.AdamW(
            model.model.parameters(),
            lr=config.get('learning_rate', 2e-5)
        )
        self.scheduler = scheduler
        self.config = config or {}
        
        # Training configuration
        self.num_epochs = config.get('num_epochs', 10)
        self.early_stopping_patience = config.get('early_stopping_patience', 3)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def train(self, callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """Train the model with early stopping and checkpointing."""
        start_time = time.time()
        callbacks = callbacks or []
        
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(epoch)
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validation phase
            if self.val_dataloader:
                val_metrics = self._validate_epoch(epoch)
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_accuracy'].append(val_metrics['accuracy'])
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
                self.training_history['learning_rates'].append(
                    self.optimizer.param_groups[0]['lr']
                )
            
            # Execute callbacks
            for callback in callbacks:
                callback(epoch, train_metrics, val_metrics if self.val_dataloader else None)
        
        training_time = time.time() - start_time
        
        # Compile training summary
        summary = {
            'total_epochs': epoch + 1,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'history': self.training_history,
            'model_size_mb': self.model.get_model_size(),
            'trainable_params': self.model.get_trainable_params()
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return summary
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]"
        )
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            loss = self.model.train_step(data, target, self.optimizer)
            total_loss += loss
            
            # Calculate accuracy
            with torch.no_grad():
                output = self.model.model(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss:.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        metrics = {
            'loss': total_loss / len(self.train_dataloader),
            'accuracy': correct / total
        }
        
        return metrics
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        metrics = self.model.evaluate(self.val_dataloader)
        
        logger.info(
            f"Epoch {epoch + 1} validation: "
            f"loss = {metrics['loss']:.4f}, "
            f"accuracy = {metrics['accuracy']:.4f}"
        )
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['metrics']
