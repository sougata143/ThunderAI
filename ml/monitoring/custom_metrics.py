from prometheus_client import Counter, Histogram, Gauge
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Training metrics
        self.training_steps = Counter(
            'model_training_steps_total',
            'Total number of training steps',
            ['model_name']
        )
        
        self.training_loss = Histogram(
            'model_training_loss',
            'Training loss values',
            ['model_name'],
            buckets=(0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float('inf'))
        )
        
        self.training_accuracy = Histogram(
            'model_training_accuracy',
            'Training accuracy values',
            ['model_name'],
            buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        )
        
        self.epoch_duration = Histogram(
            'model_epoch_duration_seconds',
            'Duration of training epochs',
            ['model_name'],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float('inf'))
        )
        
        self.model_parameters = Gauge(
            'model_parameters_total',
            'Total number of model parameters',
            ['model_name']
        )

    def record_training_step(self):
        """Record a training step"""
        self.training_steps.labels(model_name=self.model_name).inc()

    def record_loss(self, loss: float):
        """Record training loss"""
        self.training_loss.labels(model_name=self.model_name).observe(loss)

    def record_accuracy(self, accuracy: float):
        """Record training accuracy"""
        self.training_accuracy.labels(model_name=self.model_name).observe(accuracy)

    def record_epoch_duration(self, duration: float):
        """Record epoch duration"""
        self.epoch_duration.labels(model_name=self.model_name).observe(duration)

    def set_parameter_count(self, count: int):
        """Set total parameter count"""
        self.model_parameters.labels(model_name=self.model_name).set(count)

    def record_batch_metrics(self, metrics: dict):
        """Record multiple metrics from a training batch"""
        try:
            if 'loss' in metrics:
                self.record_loss(metrics['loss'])
            if 'accuracy' in metrics:
                self.record_accuracy(metrics['accuracy'])
            self.record_training_step()
        except Exception as e:
            logger.error(f"Error recording batch metrics: {str(e)}") 