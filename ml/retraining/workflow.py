from typing import Dict, Any, Optional, List
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import logging
from .orchestrator import RetrainingOrchestrator
from .monitoring import RetrainingMonitor
from .model_testing import ModelTester
from ..monitoring.custom_metrics import MetricsCollector

class RetrainingWorkflow:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator = RetrainingOrchestrator(config)
        self.monitor = RetrainingMonitor(config)
        self.model_tester = ModelTester(config)
        self.metrics_collector = MetricsCollector()
    
    @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=24))
    def check_retraining_need(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Check if retraining is needed based on metrics"""
        return self.orchestrator.check_retraining_triggers(model_id, metrics)
    
    @task
    def prepare_data(
        self,
        model_id: str,
        pipeline_run_id: str
    ) -> Dict[str, Any]:
        """Prepare and validate training data"""
        try:
            # Start monitoring
            self.monitor.monitor_retraining_pipeline(model_id, pipeline_run_id)
            
            # Prepare data
            data = self.orchestrator.prepare_training_data(
                new_data=self._fetch_new_data(model_id),
                historical_data=self._fetch_historical_data(model_id)
            )
            
            # Update monitoring
            self.monitor.update_stage_status(
                pipeline_run_id,
                'data_preparation',
                'completed',
                {'data_size': len(data['texts'])}
            )
            
            return data
            
        except Exception as e:
            self.monitor.update_stage_status(
                pipeline_run_id,
                'data_preparation',
                'failed',
                {'error': str(e)}
            )
            raise
    
    @task
    def train_model(
        self,
        model_id: str,
        pipeline_run_id: str,
        training_data: Dict[str, Any]
    ) -> str:
        """Train model with prepared data"""
        try:
            # Train model
            new_version = self.orchestrator.train_model(
                model_id,
                training_data,
                self.config.get('hyperparameters')
            )
            
            # Update monitoring
            self.monitor.update_stage_status(
                pipeline_run_id,
                'training',
                'completed',
                {'model_version': new_version}
            )
            
            return new_version
            
        except Exception as e:
            self.monitor.update_stage_status(
                pipeline_run_id,
                'training',
                'failed',
                {'error': str(e)}
            )
            raise
    
    @task
    def test_model(
        self,
        model_id: str,
        version: str,
        pipeline_run_id: str,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test newly trained model"""
        try:
            # Run tests
            test_results = self.model_tester.run_test_suite(
                model_id,
                version,
                test_data
            )
            
            # Update monitoring
            self.monitor.update_stage_status(
                pipeline_run_id,
                'testing',
                test_results['status'],
                test_results['tests']
            )
            
            return test_results
            
        except Exception as e:
            self.monitor.update_stage_status(
                pipeline_run_id,
                'testing',
                'failed',
                {'error': str(e)}
            )
            raise
    
    @task
    def deploy_model(
        self,
        model_id: str,
        version: str,
        pipeline_run_id: str,
        test_results: Dict[str, Any]
    ) -> bool:
        """Deploy model if tests pass"""
        try:
            if test_results['status'] == 'passed':
                deployed = self.orchestrator.deploy_model(
                    model_id,
                    version,
                    test_results['tests']['performance']['metrics']
                )
                
                status = 'completed' if deployed else 'skipped'
                self.monitor.update_stage_status(
                    pipeline_run_id,
                    'deployment',
                    status,
                    {'deployed': deployed}
                )
                
                return deployed
            
            return False
            
        except Exception as e:
            self.monitor.update_stage_status(
                pipeline_run_id,
                'deployment',
                'failed',
                {'error': str(e)}
            )
            raise
    
    @flow(name="model_retraining_workflow")
    def run_workflow(
        self,
        model_id: str,
        current_metrics: Dict[str, float]
    ) -> Optional[str]:
        """Run complete retraining workflow"""
        try:
            # Generate unique pipeline run ID
            pipeline_run_id = f"retraining_{model_id}_{int(time.time())}"
            
            # Check if retraining is needed
            if not self.check_retraining_need(model_id, current_metrics):
                logging.info(f"No retraining needed for model {model_id}")
                return None
            
            # Run workflow
            training_data = self.prepare_data(model_id, pipeline_run_id)
            new_version = self.train_model(model_id, pipeline_run_id, training_data)
            test_results = self.test_model(
                model_id,
                new_version,
                pipeline_run_id,
                self._prepare_test_data(training_data)
            )
            deployed = self.deploy_model(
                model_id,
                new_version,
                pipeline_run_id,
                test_results
            )
            
            # Complete monitoring
            self.monitor.complete_pipeline_monitoring(
                pipeline_run_id,
                'completed',
                {
                    'deployed': deployed,
                    'version': new_version,
                    **test_results['tests']['performance']['metrics']
                }
            )
            
            return new_version if deployed else None
            
        except Exception as e:
            logging.error(f"Retraining workflow failed: {str(e)}")
            if 'pipeline_run_id' in locals():
                self.monitor.complete_pipeline_monitoring(
                    pipeline_run_id,
                    'failed',
                    {'error': str(e)}
                )
            raise
    
    def _fetch_new_data(self, model_id: str) -> Dict[str, Any]:
        """Fetch new training data"""
        # Implement data fetching logic
        pass
    
    def _fetch_historical_data(self, model_id: str) -> Dict[str, Any]:
        """Fetch historical training data"""
        # Implement data fetching logic
        pass
    
    def _prepare_test_data(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare test dataset from training data"""
        # Implement test data preparation logic
        pass 