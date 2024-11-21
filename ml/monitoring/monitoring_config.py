from typing import Dict, Any, List, Optional
from pydantic import BaseSettings
import logging
from prometheus_client import Counter, Gauge, Histogram, Summary
import time

class MonitoringConfig(BaseSettings):
    # Metrics Settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8000
    METRICS_PATH: str = "/metrics"
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = "thunderai.log"
    
    # Alerting Settings
    ALERT_THRESHOLDS: Dict[str, float] = {
        "model_accuracy": 0.9,
        "model_latency": 1.0,  # seconds
        "error_rate": 0.01,
        "memory_usage": 0.9,  # 90% of available memory
        "cpu_usage": 0.8  # 80% of CPU
    }
    
    # Tracing Settings
    ENABLE_TRACING: bool = True
    TRACE_SAMPLE_RATE: float = 0.1
    
    class Config:
        env_file = ".env"

class MetricsService:
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Initialize Prometheus metrics
        self.model_latency = Histogram(
            'model_prediction_latency_seconds',
            'Model prediction latency in seconds',
            ['model_name', 'version']
        )
        
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of predictions',
            ['model_name', 'version', 'status']
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy',
            ['model_name', 'version']
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.request_duration = Summary(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint']
        )
    
    def record_prediction(
        self,
        model_name: str,
        version: str,
        latency: float,
        success: bool
    ):
        """Record prediction metrics"""
        self.model_latency.labels(
            model_name=model_name,
            version=version
        ).observe(latency)
        
        self.prediction_counter.labels(
            model_name=model_name,
            version=version,
            status="success" if success else "failure"
        ).inc()
    
    def update_model_accuracy(
        self,
        model_name: str,
        version: str,
        accuracy: float
    ):
        """Update model accuracy metric"""
        self.model_accuracy.labels(
            model_name=model_name,
            version=version
        ).set(accuracy)
    
    def record_resource_usage(
        self,
        memory_bytes: int,
        cpu_percent: float
    ):
        """Record system resource usage"""
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_percent)
    
    def time_request(
        self,
        method: str,
        endpoint: str
    ):
        """Decorator to time HTTP requests"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.request_duration.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
                    return result
                except Exception as e:
                    logging.error(f"Request failed: {str(e)}")
                    raise
            return wrapper
        return decorator 