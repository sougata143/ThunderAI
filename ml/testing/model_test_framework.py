from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
from ..monitoring.custom_metrics import MetricsCollector
from ..versioning.model_registry import ModelRegistry
import logging
from dataclasses import dataclass

@dataclass
class TestCase:
    """Test case configuration"""
    name: str
    inputs: List[Any]
    expected_outputs: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    test_type: str = 'functional'  # functional, performance, robustness
    timeout: float = 30.0  # seconds

@dataclass
class TestResult:
    """Test execution result"""
    test_case: TestCase
    passed: bool
    actual_outputs: Optional[List[Any]] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    execution_time: float = 0.0

class ModelTestFramework:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.model_registry = ModelRegistry()
        
        # Test configuration
        self.performance_threshold = config.get('performance_threshold', 0.9)
        self.latency_threshold = config.get('latency_threshold', 100)  # ms
        self.memory_threshold = config.get('memory_threshold', 0.9)
        
        # Initialize test suites
        self.test_suites = {
            'functional': self._create_functional_suite(),
            'performance': self._create_performance_suite(),
            'robustness': self._create_robustness_suite(),
            'integration': self._create_integration_suite()
        }
    
    def run_test_suite(
        self,
        model_id: str,
        suite_name: str,
        custom_test_cases: Optional[List[TestCase]] = None
    ) -> Dict[str, Any]:
        """Run a complete test suite"""
        try:
            # Load model
            model = self.model_registry.load_model(model_id)
            
            # Get test cases
            test_cases = self.test_suites[suite_name]
            if custom_test_cases:
                test_cases.extend(custom_test_cases)
            
            # Run tests
            results = []
            for test_case in test_cases:
                result = self._run_test_case(model, test_case)
                results.append(result)
                
                # Record metrics
                self._record_test_metrics(model_id, result)
            
            # Generate report
            report = self._generate_test_report(results)
            
            return report
            
        except Exception as e:
            logging.error(f"Error running test suite: {str(e)}")
            raise
    
    def _create_functional_suite(self) -> List[TestCase]:
        """Create functional test cases"""
        return [
            TestCase(
                name="basic_prediction",
                inputs=["This is a test input"],
                expected_outputs=[1],
                test_type="functional"
            ),
            TestCase(
                name="empty_input",
                inputs=[""],
                test_type="functional"
            ),
            TestCase(
                name="long_input",
                inputs=["a" * 1000],
                test_type="functional"
            )
        ]
    
    def _create_performance_suite(self) -> List[TestCase]:
        """Create performance test cases"""
        return [
            TestCase(
                name="latency_test",
                inputs=["Test input"] * 100,
                test_type="performance",
                metadata={"batch_size": 1}
            ),
            TestCase(
                name="throughput_test",
                inputs=["Test input"] * 1000,
                test_type="performance",
                metadata={"batch_size": 32}
            ),
            TestCase(
                name="memory_usage_test",
                inputs=["Test input"] * 500,
                test_type="performance",
                metadata={"monitor_memory": True}
            )
        ]
    
    def _create_robustness_suite(self) -> List[TestCase]:
        """Create robustness test cases"""
        return [
            TestCase(
                name="noise_test",
                inputs=self._generate_noisy_inputs(["Test input"], noise_level=0.1),
                test_type="robustness"
            ),
            TestCase(
                name="adversarial_test",
                inputs=self._generate_adversarial_inputs(["Test input"]),
                test_type="robustness"
            ),
            TestCase(
                name="edge_case_test",
                inputs=self._generate_edge_cases(),
                test_type="robustness"
            )
        ]
    
    def _create_integration_suite(self) -> List[TestCase]:
        """Create integration test cases"""
        return [
            TestCase(
                name="pipeline_test",
                inputs=["Test input"],
                test_type="integration",
                metadata={"test_preprocessing": True}
            ),
            TestCase(
                name="api_test",
                inputs=["Test input"],
                test_type="integration",
                metadata={"test_api": True}
            )
        ]
    
    def _run_test_case(
        self,
        model: Any,
        test_case: TestCase
    ) -> TestResult:
        """Run a single test case"""
        import time
        start_time = time.time()
        
        try:
            # Execute test based on type
            if test_case.test_type == "functional":
                result = self._run_functional_test(model, test_case)
            elif test_case.test_type == "performance":
                result = self._run_performance_test(model, test_case)
            elif test_case.test_type == "robustness":
                result = self._run_robustness_test(model, test_case)
            else:
                result = self._run_integration_test(model, test_case)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            return TestResult(
                test_case=test_case,
                passed=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _run_functional_test(
        self,
        model: Any,
        test_case: TestCase
    ) -> TestResult:
        """Run functional test case"""
        outputs = []
        for input_data in test_case.inputs:
            output = model.predict(input_data)
            outputs.append(output)
        
        passed = True
        if test_case.expected_outputs:
            passed = all(
                self._compare_outputs(actual, expected)
                for actual, expected in zip(outputs, test_case.expected_outputs)
            )
        
        return TestResult(
            test_case=test_case,
            passed=passed,
            actual_outputs=outputs
        )
    
    def _run_performance_test(
        self,
        model: Any,
        test_case: TestCase
    ) -> TestResult:
        """Run performance test case"""
        import time
        batch_size = test_case.metadata.get('batch_size', 1)
        
        latencies = []
        memory_usage = []
        
        for i in range(0, len(test_case.inputs), batch_size):
            batch = test_case.inputs[i:i + batch_size]
            
            start_time = time.time()
            model.predict(batch)
            latencies.append((time.time() - start_time) * 1000)  # ms
            
            if test_case.metadata.get('monitor_memory', False):
                memory_usage.append(self._get_memory_usage())
        
        metrics = {
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }
        
        if memory_usage:
            metrics['max_memory_usage'] = max(memory_usage)
        
        passed = (
            metrics['p95_latency'] <= self.latency_threshold and
            (not memory_usage or max(memory_usage) <= self.memory_threshold)
        )
        
        return TestResult(
            test_case=test_case,
            passed=passed,
            metrics=metrics
        )
    
    def _run_robustness_test(
        self,
        model: Any,
        test_case: TestCase
    ) -> TestResult:
        """Run robustness test case"""
        original_outputs = [model.predict(input_data) for input_data in test_case.inputs]
        perturbed_outputs = []
        
        for input_data in test_case.inputs:
            # Generate perturbations
            perturbations = self._generate_perturbations(input_data)
            
            # Test model stability
            outputs = []
            for perturbed_input in perturbations:
                output = model.predict(perturbed_input)
                outputs.append(output)
            
            perturbed_outputs.append(outputs)
        
        # Calculate stability metrics
        stability_scores = []
        for orig, perturbed in zip(original_outputs, perturbed_outputs):
            stability = sum(
                self._compare_outputs(orig, pert)
                for pert in perturbed
            ) / len(perturbed)
            stability_scores.append(stability)
        
        metrics = {
            'avg_stability': np.mean(stability_scores),
            'min_stability': min(stability_scores)
        }
        
        passed = metrics['avg_stability'] >= self.config.get('stability_threshold', 0.8)
        
        return TestResult(
            test_case=test_case,
            passed=passed,
            metrics=metrics
        )
    
    def _run_integration_test(
        self,
        model: Any,
        test_case: TestCase
    ) -> TestResult:
        """Run integration test case"""
        # Implement integration test logic
        pass
    
    def _generate_test_report(
        self,
        results: List[TestResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results if r.passed),
            'failed_tests': sum(1 for r in results if not r.passed),
            'total_execution_time': sum(r.execution_time for r in results),
            'test_results': []
        }
        
        for result in results:
            test_result = {
                'name': result.test_case.name,
                'type': result.test_case.test_type,
                'passed': result.passed,
                'execution_time': result.execution_time
            }
            
            if result.metrics:
                test_result['metrics'] = result.metrics
            if result.error:
                test_result['error'] = result.error
                
            report['test_results'].append(test_result)
        
        return report
    
    def _record_test_metrics(
        self,
        model_id: str,
        result: TestResult
    ):
        """Record test metrics"""
        if result.metrics:
            for metric_name, value in result.metrics.items():
                self.metrics_collector.record_testing_metric(
                    model_id=model_id,
                    metric_name=f"{result.test_case.name}_{metric_name}",
                    value=value
                ) 