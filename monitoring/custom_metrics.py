from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import GPUtil
import logging
from typing import Dict, Any

# System Metrics
CPU_USAGE = Gauge('thunderai_cpu_usage_percent', 'CPU Usage Percentage')
RAM_USAGE = Gauge('thunderai_ram_usage_bytes', 'RAM Usage in Bytes')
DISK_USAGE = Gauge('thunderai_disk_usage_bytes', 'Disk Usage in Bytes')
GPU_USAGE = Gauge('thunderai_gpu_usage_percent', 'GPU Usage Percentage', ['gpu_id'])
GPU_MEMORY = Gauge('thunderai_gpu_memory_used_bytes', 'GPU Memory Usage', ['gpu_id'])

# Model Performance Metrics
MODEL_ACCURACY = Gauge('thunderai_model_accuracy', 'Model Accuracy', ['model_name', 'version'])
MODEL_LATENCY = Summary('thunderai_model_latency_seconds', 'Model Latency', ['model_name', 'version'])
PREDICTION_DISTRIBUTION = Counter('thunderai_prediction_distribution', 'Distribution of Predictions', ['model_name', 'class'])

# Business Metrics
USER_REQUESTS = Counter('thunderai_user_requests_total', 'Total User Requests', ['user_id'])
API_USAGE = Counter('thunderai_api_usage_total', 'API Usage by Endpoint', ['endpoint', 'method'])

class MetricsCollector:
    @staticmethod
    def collect_system_metrics():
        # CPU and RAM
        CPU_USAGE.set(psutil.cpu_percent())
        RAM_USAGE.set(psutil.virtual_memory().used)
        DISK_USAGE.set(psutil.disk_usage('/').used)
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                GPU_USAGE.labels(gpu_id=gpu.id).set(gpu.load * 100)
                GPU_MEMORY.labels(gpu_id=gpu.id).set(gpu.memoryUsed)
        except Exception as e:
            logging.warning(f"Failed to collect GPU metrics: {e}")

    @staticmethod
    def record_prediction(model_name: str, version: str, prediction_class: str, latency: float):
        PREDICTION_DISTRIBUTION.labels(model_name=model_name, class=prediction_class).inc()
        MODEL_LATENCY.labels(model_name=model_name, version=version).observe(latency)

    @staticmethod
    def update_model_accuracy(model_name: str, version: str, accuracy: float):
        MODEL_ACCURACY.labels(model_name=model_name, version=version).set(accuracy) 