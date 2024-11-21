from typing import Dict, Any, Optional, List
import time
import psutil
import torch
import logging
from prometheus_client import Counter, Gauge, Histogram
from .custom_metrics import MetricsCollector

class MonitoringService:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        
        # Initialize Prometheus metrics
        self.prediction_counter = Counter(
            'thunderai_predictions_total',
            'Total number of predictions',
            ['model_name', 'version']
        )
        self.prediction_latency = Histogram(
            'thunderai_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name', 'version']
        )
        self.model_accuracy = Gauge(
            'thunderai_model_accuracy',
            'Model accuracy',
            ['model_name', 'version']
        )
        self.gpu_memory_usage = Gauge(
            'thunderai_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['device']
        )
        
        # Start system monitoring
        self.start_system_monitoring()
    
    def record_prediction(
        self,
        model_name: str,
        version: str,
        latency: float,
        success: bool
    ):
        """Record prediction metrics"""
        self.prediction_counter.labels(
            model_name=model_name,
            version=version
        ).inc()
        
        self.prediction_latency.labels(
            model_name=model_name,
            version=version
        ).observe(latency)
        
        self.metrics_collector.record_prediction_metric(
            model_name=model_name,
            version=version,
            latency=latency,
            success=success
        )
    
    def update_model_accuracy(
        self,
        model_name: str,
        version: str,
        accuracy: float
    ):
        """Update model accuracy metrics"""
        self.model_accuracy.labels(
            model_name=model_name,
            version=version
        ).set(accuracy)
        
        self.metrics_collector.record_model_metric(
            model_name=model_name,
            version=version,
            metric_name="accuracy",
            value=accuracy
        )
    
    def start_system_monitoring(self):
        """Start monitoring system resources"""
        def monitor_resources():
            while True:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics_collector.record_system_metric(
                    "cpu_usage",
                    cpu_percent
                )
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics_collector.record_system_metric(
                    "memory_usage",
                    memory.percent
                )
                
                # GPU usage if available
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i)
                        self.gpu_memory_usage.labels(device=f"gpu_{i}").set(
                            memory_allocated
                        )
                        self.metrics_collector.record_system_metric(
                            f"gpu_{i}_memory",
                            memory_allocated
                        )
                
                time.sleep(15)  # Update every 15 seconds
        
        # Start monitoring in a separate thread
        import threading
        monitoring_thread = threading.Thread(
            target=monitor_resources,
            daemon=True
        )
        monitoring_thread.start()
    
    def monitor_batch_processing(
        self,
        batch_size: int,
        processing_time: float,
        success: bool
    ):
        """Monitor batch processing metrics"""
        self.metrics_collector.record_batch_metric(
            batch_size=batch_size,
            processing_time=processing_time,
            success=success
        )
    
    def monitor_model_loading(
        self,
        model_name: str,
        version: str,
        loading_time: float
    ):
        """Monitor model loading metrics"""
        self.metrics_collector.record_model_metric(
            model_name=model_name,
            version=version,
            metric_name="loading_time",
            value=loading_time
        ) 