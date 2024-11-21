from prometheus_client import Summary

# Define a Prometheus metric to track model training time
MODEL_TRAINING_TIME = Summary('model_training_time_seconds', 'Time spent training model', ['model_name', 'version']) 