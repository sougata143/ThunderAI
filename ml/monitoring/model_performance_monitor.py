from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta
import torch
from ..versioning.model_registry import ModelRegistry
from .custom_metrics import MetricsCollector
import logging

class ModelPerformanceMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.model_registry = ModelRegistry()
        
        # Configure monitoring thresholds
        self.accuracy_threshold = config.get('accuracy_threshold', 0.9)
        self.latency_threshold = config.get('latency_threshold', 100)  # ms
        self.memory_threshold = config.get('memory_threshold', 0.9)  # 90% of available memory
        
        # Initialize monitoring state
        self.performance_history = {}
        self.alert_history = {}
    
    def monitor_prediction(
        self,
        model_id: str,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Monitor a single prediction"""
        try:
            monitoring_data = {
                'timestamp': datetime.utcnow(),
                'model_id': model_id,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'metadata': metadata or {},
                'metrics': {}
            }
            
            # Calculate metrics
            metrics = self._calculate_prediction_metrics(
                prediction,
                ground_truth,
                metadata
            )
            monitoring_data['metrics'] = metrics
            
            # Update history
            self._update_performance_history(model_id, monitoring_data)
            
            # Check for alerts
            alerts = self._check_prediction_alerts(model_id, metrics)
            if alerts:
                self._handle_alerts(model_id, alerts)
            
            # Record metrics
            self._record_metrics(model_id, metrics)
            
            return monitoring_data
            
        except Exception as e:
            logging.error(f"Error monitoring prediction: {str(e)}")
            raise
    
    def monitor_batch_predictions(
        self,
        model_id: str,
        predictions: List[Any],
        ground_truths: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Monitor a batch of predictions"""
        try:
            batch_metrics = {
                'timestamp': datetime.utcnow(),
                'model_id': model_id,
                'batch_size': len(predictions),
                'metrics': {}
            }
            
            # Calculate batch metrics
            metrics = self._calculate_batch_metrics(
                predictions,
                ground_truths,
                metadata
            )
            batch_metrics['metrics'] = metrics
            
            # Update history
            self._update_performance_history(model_id, batch_metrics)
            
            # Check for alerts
            alerts = self._check_batch_alerts(model_id, metrics)
            if alerts:
                self._handle_alerts(model_id, alerts)
            
            # Record metrics
            self._record_batch_metrics(model_id, metrics)
            
            return batch_metrics
            
        except Exception as e:
            logging.error(f"Error monitoring batch predictions: {str(e)}")
            raise
    
    def _calculate_prediction_metrics(
        self,
        prediction: Any,
        ground_truth: Optional[Any],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate metrics for a single prediction"""
        metrics = {}
        
        # Calculate latency if provided
        if metadata and 'latency' in metadata:
            metrics['latency'] = metadata['latency']
        
        # Calculate accuracy if ground truth is provided
        if ground_truth is not None:
            metrics['accuracy'] = float(prediction == ground_truth)
        
        # Calculate confidence if available
        if isinstance(prediction, dict) and 'confidence' in prediction:
            metrics['confidence'] = prediction['confidence']
        
        # Calculate memory usage
        if torch.cuda.is_available():
            metrics['gpu_memory_used'] = torch.cuda.memory_allocated()
        
        return metrics
    
    def _calculate_batch_metrics(
        self,
        predictions: List[Any],
        ground_truths: Optional[List[Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate metrics for a batch of predictions"""
        metrics = {}
        
        # Calculate average latency if provided
        if metadata and 'latencies' in metadata:
            metrics['avg_latency'] = np.mean(metadata['latencies'])
            metrics['p95_latency'] = np.percentile(metadata['latencies'], 95)
            metrics['p99_latency'] = np.percentile(metadata['latencies'], 99)
        
        # Calculate batch accuracy if ground truths are provided
        if ground_truths is not None:
            correct = sum(p == gt for p, gt in zip(predictions, ground_truths))
            metrics['batch_accuracy'] = correct / len(predictions)
        
        # Calculate average confidence if available
        if all(isinstance(p, dict) and 'confidence' in p for p in predictions):
            confidences = [p['confidence'] for p in predictions]
            metrics['avg_confidence'] = np.mean(confidences)
        
        # Calculate memory usage
        if torch.cuda.is_available():
            metrics['gpu_memory_used'] = torch.cuda.memory_allocated()
        
        return metrics
    
    def _update_performance_history(
        self,
        model_id: str,
        monitoring_data: Dict[str, Any]
    ):
        """Update performance history for the model"""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        # Keep last 1000 records
        if len(self.performance_history[model_id]) >= 1000:
            self.performance_history[model_id].pop(0)
        
        self.performance_history[model_id].append(monitoring_data)
    
    def _check_prediction_alerts(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check for alerts based on prediction metrics"""
        alerts = []
        
        # Check accuracy
        if 'accuracy' in metrics and metrics['accuracy'] < self.accuracy_threshold:
            alerts.append({
                'level': 'warning',
                'message': f'Low prediction accuracy: {metrics["accuracy"]:.3f}'
            })
        
        # Check latency
        if 'latency' in metrics and metrics['latency'] > self.latency_threshold:
            alerts.append({
                'level': 'warning',
                'message': f'High prediction latency: {metrics["latency"]:.1f}ms'
            })
        
        # Check memory usage
        if 'gpu_memory_used' in metrics:
            memory_usage = metrics['gpu_memory_used'] / torch.cuda.get_device_properties(0).total_memory
            if memory_usage > self.memory_threshold:
                alerts.append({
                    'level': 'warning',
                    'message': f'High GPU memory usage: {memory_usage:.1%}'
                })
        
        return alerts
    
    def _check_batch_alerts(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check for alerts based on batch metrics"""
        alerts = []
        
        # Check batch accuracy
        if 'batch_accuracy' in metrics and metrics['batch_accuracy'] < self.accuracy_threshold:
            alerts.append({
                'level': 'warning',
                'message': f'Low batch accuracy: {metrics["batch_accuracy"]:.3f}'
            })
        
        # Check average latency
        if 'avg_latency' in metrics and metrics['avg_latency'] > self.latency_threshold:
            alerts.append({
                'level': 'warning',
                'message': f'High average latency: {metrics["avg_latency"]:.1f}ms'
            })
        
        return alerts
    
    def _handle_alerts(self, model_id: str, alerts: List[Dict[str, Any]]):
        """Handle and record alerts"""
        if model_id not in self.alert_history:
            self.alert_history[model_id] = []
        
        for alert in alerts:
            self.alert_history[model_id].append({
                'timestamp': datetime.utcnow(),
                'level': alert['level'],
                'message': alert['message']
            })
            
            # Record alert metric
            self.metrics_collector.record_alert(
                model_id=model_id,
                alert_level=alert['level']
            )
            
            logging.warning(f"Model {model_id} alert: {alert['message']}")
    
    def _record_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Record metrics to monitoring system"""
        for metric_name, value in metrics.items():
            self.metrics_collector.record_model_metric(
                model_id=model_id,
                metric_name=metric_name,
                value=value
            )
    
    def _record_batch_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Record batch metrics to monitoring system"""
        for metric_name, value in metrics.items():
            self.metrics_collector.record_batch_metric(
                model_id=model_id,
                metric_name=metric_name,
                value=value
            ) 