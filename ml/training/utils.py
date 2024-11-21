from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from ..monitoring.custom_metrics import MetricsCollector
import time

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping handler"""
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        
        self.should_stop = self.counter >= self.patience
        return self.should_stop

class MetricTracker:
    """Tracks and computes metrics during training"""
    def __init__(self):
        self.metrics = {}
        self.current_values = {}
        self.history = {}
    
    def add_metric(self, name: str, compute_fn: Callable):
        self.metrics[name] = compute_fn
        self.current_values[name] = []
        self.history[name] = []
    
    def update(self, name: str, value: float):
        if name not in self.current_values:
            self.current_values[name] = []
        self.current_values[name].append(value)
    
    def compute_epoch(self):
        epoch_metrics = {}
        for name, values in self.current_values.items():
            if values:
                epoch_metrics[name] = np.mean(values)
                self.history[name].append(epoch_metrics[name])
                self.current_values[name] = []
        return epoch_metrics

class TrainingLogger:
    """Handles logging during training"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_collector = MetricsCollector(model_name)
        self.epoch_start_time = None
    
    def start_epoch(self):
        self.epoch_start_time = time.time()
    
    def end_epoch(self, metrics: Dict[str, float]):
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            self.metrics_collector.record_epoch_duration(duration)
        self.metrics_collector.record_batch_metrics(metrics)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics for the current training step"""
        try:
            self.metrics_collector.record_batch_metrics(metrics)
            
            # Log to console or file
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Training metrics - {metrics_str}")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")

class GradientClipping:
    """Handles gradient clipping during training"""
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, parameters):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=self.max_norm,
            norm_type=self.norm_type
        ) 