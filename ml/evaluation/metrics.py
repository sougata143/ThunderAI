from typing import Dict, Any, List, Union, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)
import torch
from ..monitoring.custom_metrics import MetricsCollector

class EvaluationMetrics:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
    
    def compute_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # For binary classification with probabilities
        if predictions.shape[-1] == 2:
            pred_probs = predictions[:, 1]
            pred_labels = (pred_probs >= threshold).astype(int)
        else:
            pred_labels = predictions
            pred_probs = None
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(targets, pred_labels))
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            pred_labels,
            average='weighted'
        )
        metrics.update({
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        })
        
        # Confusion matrix
        cm = confusion_matrix(targets, pred_labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC and PR curves if probabilities available
        if pred_probs is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(targets, pred_probs))
                metrics['pr_auc'] = float(average_precision_score(targets, pred_probs))
            except:
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        
        # Detailed classification report
        report = classification_report(
            targets,
            pred_labels,
            output_dict=True
        )
        metrics['classification_report'] = report
        
        # Log metrics
        self._log_metrics(metrics)
        
        return metrics
    
    def compute_regression_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute regression-specific metrics"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        # Log metrics
        self._log_metrics(metrics)
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to monitoring system"""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_collector.record_evaluation_metric(
                    metric_name=name,
                    value=value
                ) 