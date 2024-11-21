from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from ..monitoring.custom_metrics import MetricsCollector
from ..versioning.model_registry import ModelRegistry
import logging

class ModelTester:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.model_registry = ModelRegistry()
        
        # Configure test thresholds
        self.min_accuracy = config.get('min_accuracy', 0.9)
        self.max_latency = config.get('max_latency', 100)  # ms
        self.min_samples = config.get('min_test_samples', 1000)
    
    def run_test_suite(
        self,
        model_id: str,
        version: str,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run comprehensive test suite on retrained model"""
        try:
            test_results = {
                'model_id': model_id,
                'version': version,
                'status': 'running',
                'tests': {}
            }
            
            # Load model
            model = self.model_registry.load_model(model_id, version)
            
            # Run tests
            test_results['tests']['basic_validation'] = self._run_basic_validation(
                model,
                test_data
            )
            
            test_results['tests']['performance'] = self._test_performance(
                model,
                test_data
            )
            
            test_results['tests']['robustness'] = self._test_robustness(
                model,
                test_data
            )
            
            test_results['tests']['latency'] = self._test_latency(
                model,
                test_data
            )
            
            # Determine overall status
            test_results['status'] = self._determine_test_status(test_results['tests'])
            
            # Record test metrics
            self._record_test_metrics(test_results)
            
            return test_results
            
        except Exception as e:
            logging.error(f"Error running test suite: {str(e)}")
            raise
    
    def _run_basic_validation(
        self,
        model: Any,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run basic model validation tests"""
        results = {
            'status': 'running',
            'checks': {}
        }
        
        # Check input handling
        results['checks']['input_validation'] = self._validate_inputs(
            model,
            test_data
        )
        
        # Check output format
        results['checks']['output_validation'] = self._validate_outputs(
            model,
            test_data
        )
        
        # Check model attributes
        results['checks']['model_validation'] = self._validate_model(model)
        
        results['status'] = 'passed' if all(
            check['status'] == 'passed'
            for check in results['checks'].values()
        ) else 'failed'
        
        return results
    
    def _test_performance(
        self,
        model: Any,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test model performance metrics"""
        predictions = model.predict(test_data['features'])
        
        metrics = {
            'accuracy': accuracy_score(test_data['labels'], predictions),
            'precision': None,
            'recall': None,
            'f1': None
        }
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_data['labels'],
            predictions,
            average='weighted'
        )
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        return {
            'status': 'passed' if metrics['accuracy'] >= self.min_accuracy else 'failed',
            'metrics': metrics
        }
    
    def _test_robustness(
        self,
        model: Any,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test model robustness"""
        results = {
            'noise_tolerance': self._test_noise_tolerance(model, test_data),
            'adversarial_resistance': self._test_adversarial_resistance(model, test_data),
            'edge_cases': self._test_edge_cases(model, test_data)
        }
        
        return {
            'status': 'passed' if all(
                test['status'] == 'passed'
                for test in results.values()
            ) else 'failed',
            'results': results
        }
    
    def _test_latency(
        self,
        model: Any,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test model inference latency"""
        import time
        
        latencies = []
        for _ in range(100):  # Run 100 inference tests
            start_time = time.time()
            model.predict(test_data['features'][:1])  # Test single sample
            latencies.append((time.time() - start_time) * 1000)  # Convert to ms
        
        metrics = {
            'mean_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }
        
        return {
            'status': 'passed' if metrics['p95_latency'] <= self.max_latency else 'failed',
            'metrics': metrics
        }
    
    def _determine_test_status(self, test_results: Dict[str, Any]) -> str:
        """Determine overall test suite status"""
        return 'passed' if all(
            test['status'] == 'passed'
            for test in test_results.values()
        ) else 'failed'
    
    def _record_test_metrics(self, test_results: Dict[str, Any]):
        """Record test metrics"""
        for test_name, test_data in test_results['tests'].items():
            if 'metrics' in test_data:
                for metric_name, value in test_data['metrics'].items():
                    self.metrics_collector.record_testing_metric(
                        f'{test_name}_{metric_name}',
                        value,
                        model_id=test_results['model_id'],
                        version=test_results['version']
                    ) 