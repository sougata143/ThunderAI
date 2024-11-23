import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, Callable, List, Union
import logging
from tqdm import tqdm
import time
import numpy as np
from pathlib import Path
import wandb
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..models.base import BaseModel

logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """Advanced trainer with distributed training, logging, and optimization strategies."""
    
    def __init__(
        self,
        model: BaseModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        config: Dict[str, Any] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer or torch.optim.AdamW(
            model.model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        self.scheduler = scheduler
        self.config = config or {}
        
        # Training configuration
        self.num_epochs = config.get('num_epochs', 10)
        self.early_stopping_patience = config.get('early_stopping_patience', 3)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.use_distributed = config.get('use_distributed', False)
        self.use_wandb = config.get('use_wandb', True)
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Setup distributed training if enabled
        if self.use_distributed:
            self._setup_distributed_training()
        
        # Initialize W&B logging
        if self.use_wandb and (not self.use_distributed or self.is_main_process()):
            self._setup_wandb()
        
        # Setup model checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def _setup_distributed_training(self) -> None:
        """Setup distributed training environment."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        self.local_rank = dist.get_rank()
        torch.cuda.set_device(self.local_rank)
        
        # Wrap model in DDP
        self.model.model = DDP(
            self.model.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank
        )
    
    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.get('wandb_project', 'thunderai'),
            name=self.config.get('wandb_run_name'),
            config=self.config
        )
    
    @staticmethod
    def is_main_process() -> bool:
        """Check if current process is the main process."""
        return not dist.is_initialized() or dist.get_rank() == 0
    
    def train(self, callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """Train the model with advanced features and logging."""
        start_time = time.time()
        callbacks = callbacks or []
        
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            if self.use_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Training phase
            train_metrics = self._train_epoch(epoch)
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)
            
            # Validation phase
            if self.val_dataloader:
                val_metrics = self._validate_epoch(epoch)
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_metrics'].append(val_metrics)
                
                # Log metrics
                if self.use_wandb and self.is_main_process():
                    wandb.log({
                        f'val/{k}': v for k, v in val_metrics.items()
                    }, step=epoch)
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    if self.is_main_process():
                        self._save_checkpoint(epoch, val_metrics)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'] if self.val_dataloader else train_metrics['loss'])
                else:
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
        
        if self.use_wandb and self.is_main_process():
            wandb.log({'training_summary': summary})
            wandb.finish()
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return summary
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with advanced features."""
        self.model.model.train()
        total_loss = 0
        metrics = {}
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]",
            disable=not self.is_main_process()
        )
        
        for step, (inputs, labels) in enumerate(progress_bar):
            # Mixed precision training
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = self.model.train_step(inputs, labels, self.optimizer)
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss = self.model.train_step(inputs, labels, self.optimizer)
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to W&B
            if self.use_wandb and self.is_main_process() and step % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        metrics['loss'] = total_loss / len(self.train_dataloader)
        return metrics
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch with metric computation."""
        metrics = self.model.evaluate(self.val_dataloader)
        
        logger.info(
            f"Epoch {epoch + 1} validation: "
            f"loss = {metrics['loss']:.4f}, "
            f"accuracy = {metrics.get('accuracy', 0):.4f}"
        )
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint with distributed training support."""
        if not self.is_main_process():
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        model_state_dict = (
            self.model.model.module.state_dict()
            if isinstance(self.model.model, DDP)
            else self.model.model.state_dict()
        )
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint with distributed training support."""
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        if isinstance(self.model.model, DDP):
            self.model.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['metrics']
