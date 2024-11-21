from typing import Dict, Any, Optional, List
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import numpy as np
from ..monitoring.custom_metrics import MetricsCollector
from ..versioning.model_registry import ModelRegistry
from ..preprocessing.data_validator import DataValidator, ValidationConfig
from ..preprocessing.feature_extraction import FeatureExtractionService
import logging

class RetrainingOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.model_registry = ModelRegistry()
        self.data_validator = DataValidator(ValidationConfig())
        self.feature_extractor = FeatureExtractionService(config.get('feature_config', {}))
        
        # Configure retraining triggers
        self.performance_threshold = config.get('performance_threshold', 0.9)
        self.data_drift_threshold = config.get('data_drift_threshold', 0.1)
        self.min_samples_required = config.get('min_samples_required', 1000)
    
    @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
    def check_retraining_triggers(
        self,
        model_id: str,
        current_metrics: Dict[str, float]
    ) -> bool:
        """Check if retraining is needed based on various triggers"""
        triggers = []
        
        # Check performance degradation
        if current_metrics.get('accuracy', 1.0) < self.performance_threshold:
            triggers.append('performance_degradation')
        
        # Check data drift
        if current_metrics.get('data_drift', 0.0) > self.data_drift_threshold:
            triggers.append('data_drift')
        
        # Check new data volume
        if current_metrics.get('new_samples', 0) >= self.min_samples_required:
            triggers.append('sufficient_new_data')
        
        # Log triggers
        if triggers:
            logging.info(f"Retraining triggers activated for model {model_id}: {triggers}")
            self.metrics_collector.record_retraining_metric(
                'triggers_activated',
                len(triggers)
            )
        
        return len(triggers) > 0
    
    @task
    def prepare_training_data(
        self,
        new_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare and validate training data"""
        try:
            # Combine new and historical data if available
            if historical_data:
                combined_data = {
                    'texts': historical_data['texts'] + new_data['texts'],
                    'labels': historical_data['labels'] + new_data['labels']
                }
            else:
                combined_data = new_data
            
            # Validate data
            validation_report = self.data_validator.validate_dataset(
                combined_data['texts'],
                combined_data['labels']
            )
            
            if not validation_report.is_valid:
                raise ValueError(
                    f"Data validation failed: {validation_report.issues}"
                )
            
            # Extract features
            features = self.feature_extractor.extract_features(
                combined_data['texts'],
                methods=['transformer', 'tfidf']
            )
            
            return {
                'features': features,
                'labels': combined_data['labels'],
                'texts': combined_data['texts']
            }
            
        except Exception as e:
            logging.error(f"Error preparing training data: {str(e)}")
            raise
    
    @task
    def train_model(
        self,
        model_id: str,
        training_data: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Train model with prepared data"""
        try:
            # Get model class and create instance
            model_class = self.model_registry.get_model_class(model_id)
            model = model_class(hyperparameters or {})
            
            # Train model
            metrics = model.train(training_data)
            
            # Register new model version
            new_version = self.model_registry.register_model(
                model=model,
                name=model_id,
                metrics=metrics
            )
            
            # Log metrics
            self.metrics_collector.record_training_metrics(
                model_id=model_id,
                version=new_version,
                metrics=metrics
            )
            
            return new_version
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
    
    @task
    def evaluate_model(
        self,
        model_id: str,
        version: str,
        test_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate newly trained model"""
        try:
            model = self.model_registry.load_model(model_id, version)
            metrics = model.evaluate(test_data)
            
            # Log evaluation metrics
            self.metrics_collector.record_evaluation_metrics(
                model_id=model_id,
                version=version,
                metrics=metrics
            )
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise
    
    @task
    def deploy_model(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Deploy new model version if it meets criteria"""
        try:
            # Check if new model performs better
            current_version = self.model_registry.get_latest_version(model_id)
            current_metrics = self.model_registry.get_metrics(model_id, current_version)
            
            if metrics['accuracy'] > current_metrics['accuracy']:
                # Deploy new version
                self.model_registry.deploy_model(model_id, version)
                
                logging.info(
                    f"Deployed new version {version} of model {model_id}"
                )
                return True
            
            logging.info(
                f"New version {version} of model {model_id} not deployed: "
                f"performance not better than current version"
            )
            return False
            
        except Exception as e:
            logging.error(f"Error deploying model: {str(e)}")
            raise
    
    @flow(name="model_retraining_pipeline")
    def run_retraining_pipeline(
        self,
        model_id: str,
        new_data: Dict[str, Any],
        current_metrics: Dict[str, float]
    ) -> Optional[str]:
        """Run complete retraining pipeline"""
        try:
            # Check if retraining is needed
            if not self.check_retraining_triggers(model_id, current_metrics):
                logging.info(f"No retraining needed for model {model_id}")
                return None
            
            # Prepare training data
            historical_data = self.model_registry.get_training_data(model_id)
            training_data = self.prepare_training_data(new_data, historical_data)
            
            # Train new model version
            new_version = self.train_model(
                model_id,
                training_data,
                self.config.get('hyperparameters')
            )
            
            # Evaluate new model
            metrics = self.evaluate_model(model_id, new_version, training_data)
            
            # Deploy if better
            deployed = self.deploy_model(model_id, new_version, metrics)
            
            if deployed:
                return new_version
            return None
            
        except Exception as e:
            logging.error(f"Retraining pipeline failed: {str(e)}")
            self.metrics_collector.record_retraining_metric(
                'pipeline_failure',
                1
            )
            raise 