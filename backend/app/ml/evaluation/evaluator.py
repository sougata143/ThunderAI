import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from datetime import datetime
from .model_explainer import ModelExplainer

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation with advanced metrics and visualizations."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        task_type: str = 'classification',
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ):
        self.model = model
        self.device = device
        self.task_type = task_type
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)] if num_classes else None
        self.output_dir = Path(output_dir) if output_dir else Path('evaluation_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model explainer
        self.explainer = ModelExplainer(model, device, task_type)
        
        # Metrics history
        self.evaluation_history = []
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        metrics: Optional[List[str]] = None,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation with multiple metrics."""
        self.model.eval()
        metrics = metrics or ['accuracy', 'precision', 'recall', 'f1']
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                if self.task_type == 'classification':
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    predictions = outputs
                    probabilities = outputs
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        results = self._calculate_metrics(
            all_predictions,
            all_targets,
            all_probabilities,
            metrics
        )
        
        # Generate visualizations
        self._generate_visualizations(
            all_predictions,
            all_targets,
            all_probabilities
        )
        
        # Save evaluation results
        self._save_results(results)
        
        if return_predictions:
            results['predictions'] = all_predictions.tolist()
            results['probabilities'] = all_probabilities.tolist()
        
        return results
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Calculate multiple evaluation metrics."""
        results = {}
        
        if self.task_type == 'classification':
            if 'accuracy' in metrics:
                results['accuracy'] = accuracy_score(targets, predictions)
            
            if any(m in metrics for m in ['precision', 'recall', 'f1']):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets,
                    predictions,
                    average='weighted'
                )
                results.update({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            
            if 'confusion_matrix' in metrics:
                results['confusion_matrix'] = confusion_matrix(
                    targets,
                    predictions
                ).tolist()
            
            if 'roc_auc' in metrics and self.num_classes == 2:
                results['roc_auc'] = roc_auc_score(
                    targets,
                    probabilities[:, 1]
                )
        
        else:  # Regression metrics
            results.update({
                'mse': mean_squared_error(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions)),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions)
            })
        
        return results
    
    def _generate_visualizations(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray
    ) -> None:
        """Generate evaluation visualizations."""
        if self.task_type == 'classification':
            # Confusion Matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(targets, predictions)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(self.output_dir / 'confusion_matrix.png')
            plt.close()
            
            # ROC Curve for binary classification
            if self.num_classes == 2:
                plt.figure(figsize=(8, 8))
                fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.savefig(self.output_dir / 'roc_curve.png')
                plt.close()
        
        else:  # Regression plots
            plt.figure(figsize=(10, 6))
            plt.scatter(targets, predictions, alpha=0.5)
            plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title('Prediction vs True Values')
            plt.savefig(self.output_dir / 'regression_scatter.png')
            plt.close()
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f'evaluation_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.evaluation_history.append(results)
    
    def analyze_errors(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """Analyze model errors and generate explanations."""
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                if self.task_type == 'classification':
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    predictions = outputs
                
                # Find errors
                error_indices = (predictions.cpu() != targets).nonzero().squeeze()
                
                for idx in error_indices[:num_samples]:
                    error_input = inputs[idx].unsqueeze(0)
                    error_target = targets[idx].item()
                    error_pred = predictions[idx].item()
                    
                    # Get explanation for error
                    explanation = self.explainer.explain_prediction(
                        error_input,
                        target_class=error_target
                    )
                    
                    errors.append({
                        'input': error_input.cpu().numpy().tolist(),
                        'true_label': error_target,
                        'predicted_label': error_pred,
                        'explanation': explanation
                    })
                
                if len(errors) >= num_samples:
                    break
        
        return {'error_analysis': errors}
    
    def get_feature_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        method: str = 'integrated_gradients'
    ) -> Dict[str, Any]:
        """Get global feature importance across dataset."""
        feature_importances = []
        
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(self.device)
            
            batch_importance = self.explainer.get_feature_importance(
                inputs,
                method=method
            )
            feature_importances.append(batch_importance)
        
        # Aggregate feature importances
        aggregated_importance = {}
        for importance in feature_importances:
            for feature, score in importance.items():
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = []
                aggregated_importance[feature].append(score)
        
        # Calculate mean importance
        mean_importance = {
            feature: np.mean(scores)
            for feature, scores in aggregated_importance.items()
        }
        
        return {
            'feature_importance': mean_importance,
            'method': method
        }
    
    def compare_models(
        self,
        other_model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare performance with another model."""
        # Evaluate current model
        current_results = self.evaluate(dataloader, metrics)
        
        # Evaluate other model
        other_evaluator = ModelEvaluator(
            other_model,
            self.device,
            self.task_type,
            self.num_classes,
            self.class_names
        )
        other_results = other_evaluator.evaluate(dataloader, metrics)
        
        # Compare results
        comparison = {
            'current_model': current_results,
            'other_model': other_results,
            'differences': {
                metric: current_results[metric] - other_results[metric]
                for metric in current_results
                if isinstance(current_results[metric], (int, float))
            }
        }
        
        return comparison
