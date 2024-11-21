from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUEST_COUNT = Counter('thunderai_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('thunderai_request_latency_seconds', 'Request latency', ['method', 'endpoint'])

# Model metrics
MODEL_PREDICTION_TIME = Histogram('thunderai_model_prediction_seconds', 'Model prediction time', ['model_name', 'version'])
MODEL_PREDICTION_COUNT = Counter('thunderai_model_predictions_total', 'Total model predictions', ['model_name', 'version'])
MODEL_ERROR_COUNT = Counter('thunderai_model_errors_total', 'Total model errors', ['model_name', 'version', 'error_type'])

# Cache metrics
CACHE_HIT_COUNT = Counter('thunderai_cache_hits_total', 'Total cache hits')
CACHE_MISS_COUNT = Counter('thunderai_cache_misses_total', 'Total cache misses')

# System metrics
MEMORY_USAGE = Gauge('thunderai_memory_usage_bytes', 'Memory usage in bytes')
GPU_MEMORY_USAGE = Gauge('thunderai_gpu_memory_usage_bytes', 'GPU memory usage in bytes')
MODEL_LOADING_TIME = Histogram('thunderai_model_loading_seconds', 'Model loading time', ['model_name', 'version']) 