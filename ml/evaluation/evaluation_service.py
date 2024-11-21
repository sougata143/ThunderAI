from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, confusion_matrix
)
import logging
from ..monitoring.custom_metrics import MetricsCollector

class ModelEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(f"evaluation.{model_name}")
    
    def evaluate_model(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate model performance with multiple metrics"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # For binary classification
        if predictions.shape[-1] == 2:
            predictions = (predictions[:, 1] >= threshold).astype(int)
            
        # Calculate basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average='weighted'
        )
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)
        
        # Calculate ROC and AUC if applicable
        try:
            fpr, tpr, _ = roc_curve(targets, predictions)
            roc_auc = auc(fpr, tpr)
        except:
            fpr, tpr, roc_auc = None, None, None
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist() if fpr is not None else None,
                'tpr': tpr.tolist() if tpr is not None else None,
                'auc': float(roc_auc) if roc_auc is not None else None
            }
        }
        
        # Log metrics
        self.log_metrics(metrics)
        
        return metrics
    
    def analyze_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        inputs: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze prediction errors"""
        errors = {
            'false_positives': [],
            'false_negatives': []
        }
        
        for i, (pred, target, input_text) in enumerate(zip(predictions, targets, inputs)):
            if pred != target:
                error_type = 'false_positives' if pred == 1 else 'false_negatives'
                errors[error_type].append({
                    'input': input_text,
                    'predicted': int(pred),
                    'actual': int(target),
                    'index': i
                })
        
        return errors
    
    def evaluate_robustness(
        self,
        model: Any,
        test_data: List[str],
        perturbations: List[callable]
    ) -> Dict[str, float]:
        """Evaluate model robustness against perturbations"""
        base_predictions = model.predict(test_data)
        robustness_scores = {}
        
        for perturb_fn in perturbations:
            perturbed_data = [perturb_fn(text) for text in test_data]
            perturbed_predictions = model.predict(perturbed_data)
            
            # Calculate prediction stability
            stability = np.mean(
                base_predictions == perturbed_predictions
            )
            robustness_scores[perturb_fn.__name__] = float(stability)
        
        return robustness_scores
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log evaluation metrics"""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_collector.record_evaluation_metric(
                    model_name=self.model_name,
                    metric_name=metric_name,
                    value=value
                )
        
        self.logger.info(f"Evaluation metrics for {self.model_name}: {metrics}")

class ModelAnalyzer:
    """Analyzes model behavior and generates insights"""
    def __init__(self, model: Any):
        self.model = model
    
    def analyze_feature_importance(
        self,
        inputs: List[str],
        method: str = 'integrated_gradients'
    ) -> Dict[str, List[Dict[str, float]]]:
        """Analyze feature importance using various methods"""
        if method == 'integrated_gradients':
            return self._integrated_gradients(inputs)
        elif method == 'lime':
            return self._lime_explanation(inputs)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
    
    def analyze_prediction_confidence(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction confidence distribution"""
        confidence_stats = {
            'mean': float(np.mean(confidence_scores)),
            'std': float(np.std(confidence_scores)),
            'min': float(np.min(confidence_scores)),
            'max': float(np.max(confidence_scores)),
            'quartiles': [
                float(np.percentile(confidence_scores, q))
                for q in [25, 50, 75]
            ]
        }
        
        # Analyze correlation between confidence and correctness
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        
        confidence_stats['confidence_calibration'] = {
            'high_confidence_accuracy': float(
                np.mean(predictions[confidence_scores > 0.9])
            ),
            'low_confidence_accuracy': float(
                np.mean(predictions[confidence_scores < 0.5])
            )
        }
        
        return confidence_stats
    
    def _integrated_gradients(self, inputs: List[str]) -> Dict[str, List[Dict[str, float]]]:
        """Implement integrated gradients analysis"""
        # Implementation depends on the model architecture
        pass
    
    def _lime_explanation(self, inputs: List[str]) -> Dict[str, List[Dict[str, float]]]:
        """Implement LIME-based explanation"""
        # Implementation depends on the model architecture
        pass 