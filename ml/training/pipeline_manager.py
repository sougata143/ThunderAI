from typing import Dict, Any, List, Optional, Callable, AsyncIterator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from .utils import EarlyStopping, MetricTracker, TrainingLogger, GradientClipping
from ..monitoring.custom_metrics import MetricsCollector
from ..evaluation.evaluation_service import ModelEvaluator
import asyncio
from ml.models.bert import BERTModel
from ml.models.gpt import GPTModel
from ml.models.transformer import TransformerModel
from ml.models.lstm import LSTMModel

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: torch.device = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training utilities
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 3),
            mode=config.get('monitor_mode', 'min')
        )
        self.metric_tracker = MetricTracker()
        self.logger = TrainingLogger(config.get('model_name', 'unknown'))
        self.gradient_clipper = GradientClipping(
            max_norm=config.get('max_grad_norm', 1.0)
        )
        
        # Evaluation
        self.evaluator = ModelEvaluator(config.get('model_name', 'unknown'))
        
        # Initialize metrics
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup training metrics"""
        self.metric_tracker.add_metric('loss', lambda x: x)
        for metric_name, metric_fn in self.config.get('metrics', {}).items():
            self.metric_tracker.add_metric(metric_name, metric_fn)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = self.criterion(outputs, batch['labels'])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('clip_gradients', False):
                self.gradient_clipper(self.model.parameters())
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.metric_tracker.update('loss', loss.item())
            
            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    callback(batch_idx, epoch, batch, outputs, loss)
        
        # Compute epoch metrics
        metrics = self.metric_tracker.compute_epoch()
        self.logger.log_epoch(epoch, metrics)
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch['labels'])
                
                total_loss += loss.item()
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
        
        # Compute validation metrics
        val_metrics = self.evaluator.evaluate_model(
            predictions=predictions,
            targets=targets
        )
        
        val_metrics['loss'] = total_loss / len(val_loader)
        self.logger.log_validation(val_metrics)
        
        return val_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """Full training loop"""
        best_model_state = None
        best_metric = float('inf') if self.config.get('monitor_mode', 'min') == 'min' else float('-inf')
        history = {'train': [], 'val': []}
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch, callbacks)
            history['train'].append(train_metrics)
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.validate(val_loader, epoch)
                history['val'].append(val_metrics)
                
                # Early stopping check
                monitor_metric = val_metrics[self.config.get('monitor_metric', 'loss')]
                if self.early_stopping(monitor_metric):
                    logging.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Save best model
                if self._is_best_metric(monitor_metric, best_metric):
                    best_metric = monitor_metric
                    best_model_state = self.model.state_dict()
        
        # Restore best model if available
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def _is_best_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best"""
        if self.config.get('monitor_mode', 'min') == 'min':
            return current < best
        return current > best 

class ModelTrainingPipeline:
    def __init__(self, model_type: str, params: Dict):
        self.model_type = model_type
        self.params = params
        self.status = "initialized"
        self.progress = 0
        self.current_epoch = 0
        self.metrics = {
            "loss": [],
            "accuracy": []
        }
        self._should_stop = False
        self._subscribers = set()
        
        # Initialize model based on type
        self.model = self._create_model()
        
    def _create_model(self):
        if self.model_type == "bert":
            return BERTModel()
        elif self.model_type == "gpt":
            return GPTModel()
        elif self.model_type == "transformer":
            return TransformerModel()
        elif self.model_type == "lstm":
            return LSTMModel()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    async def train(self):
        """Start the training process"""
        try:
            self.status = "training"
            # Convert epochs to integer and ensure it's positive
            total_epochs = max(1, int(self.params.get("epochs", 10)))
            
            for epoch in range(total_epochs):
                if self._should_stop:
                    self.status = "stopped"
                    break
                    
                self.current_epoch = epoch
                
                # Simulate training step
                await asyncio.sleep(1)  # Replace with actual training
                
                # Update metrics
                self.metrics["loss"].append(1.0 - (epoch / total_epochs))
                self.metrics["accuracy"].append(epoch / total_epochs)
                
                # Update progress
                self.progress = ((epoch + 1) / total_epochs) * 100
                
                # Notify subscribers
                await self._notify_subscribers()
            
            if not self._should_stop:
                self.status = "completed"
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.status = "failed"
            raise
            
    async def stop(self):
        """Stop the training process"""
        self._should_stop = True
        
    async def get_updates(self) -> AsyncIterator[Dict]:
        """Subscribe to training updates"""
        queue = asyncio.Queue()
        self._subscribers.add(queue)
        
        try:
            while True:
                update = await queue.get()
                yield update
        finally:
            self._subscribers.remove(queue)
            
    async def _notify_subscribers(self):
        """Send updates to all subscribers"""
        update = {
            "status": self.status,
            "progress": self.progress,
            "currentEpoch": self.current_epoch,
            "metrics": self.metrics
        }
        
        for queue in self._subscribers:
            await queue.put(update)