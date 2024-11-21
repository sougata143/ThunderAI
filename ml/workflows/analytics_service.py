from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..monitoring.custom_metrics import MetricsCollector
import logging

class WorkflowAnalytics:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        
        # Initialize analytics storage
        self.workflow_history: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.resource_usage: Dict[str, Dict[str, List[float]]] = {}
    
    def analyze_workflow_execution(
        self,
        workflow_id: str,
        execution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a single workflow execution"""
        try:
            # Calculate execution metrics
            execution_time = (
                execution_data['end_time'] - execution_data['start_time']
            ).total_seconds()
            
            step_metrics = self._analyze_steps(execution_data['steps'])
            resource_metrics = self._analyze_resource_usage(execution_data)
            performance_metrics = self._analyze_performance(execution_data)
            
            # Store metrics for historical analysis
            self._store_execution_metrics(
                workflow_id,
                execution_time,
                step_metrics,
                resource_metrics,
                performance_metrics
            )
            
            return {
                'execution_time': execution_time,
                'step_metrics': step_metrics,
                'resource_metrics': resource_metrics,
                'performance_metrics': performance_metrics,
                'bottlenecks': self._identify_bottlenecks(step_metrics),
                'optimization_suggestions': self._generate_optimization_suggestions(
                    step_metrics,
                    resource_metrics
                )
            }
            
        except Exception as e:
            logging.error(f"Error analyzing workflow execution: {str(e)}")
            raise
    
    def _analyze_steps(
        self,
        steps: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze workflow steps performance"""
        step_metrics = {}
        
        for step_name, step_data in steps.items():
            duration = (
                step_data['end_time'] - step_data['start_time']
            ).total_seconds()
            
            step_metrics[step_name] = {
                'duration': duration,
                'status': step_data['status'],
                'resource_usage': step_data.get('resource_usage', {}),
                'success_rate': 1.0 if step_data['status'] == 'completed' else 0.0,
                'retries': step_data.get('retries', 0)
            }
        
        return step_metrics
    
    def _analyze_resource_usage(
        self,
        execution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        resource_metrics = {
            'cpu': [],
            'memory': [],
            'gpu': []
        }
        
        for step in execution_data['steps'].values():
            if 'resource_usage' in step:
                for resource, usage in step['resource_usage'].items():
                    if resource in resource_metrics:
                        resource_metrics[resource].append(usage)
        
        return {
            resource: {
                'mean': np.mean(values) if values else 0,
                'max': max(values) if values else 0,
                'p95': np.percentile(values, 95) if values else 0
            }
            for resource, values in resource_metrics.items()
        }
    
    def _analyze_performance(
        self,
        execution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze workflow performance metrics"""
        return {
            'success_rate': sum(
                1 for step in execution_data['steps'].values()
                if step['status'] == 'completed'
            ) / len(execution_data['steps']),
            'error_rate': sum(
                1 for step in execution_data['steps'].values()
                if step['status'] == 'failed'
            ) / len(execution_data['steps']),
            'total_retries': sum(
                step.get('retries', 0)
                for step in execution_data['steps'].values()
            )
        }
    
    def _identify_bottlenecks(
        self,
        step_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify workflow bottlenecks"""
        bottlenecks = []
        total_duration = sum(
            metrics['duration'] for metrics in step_metrics.values()
        )
        
        for step_name, metrics in step_metrics.items():
            # Check for time-consuming steps
            if metrics['duration'] > total_duration * 0.3:  # 30% threshold
                bottlenecks.append({
                    'step': step_name,
                    'type': 'duration',
                    'impact': 'high',
                    'metric': metrics['duration'],
                    'suggestion': 'Consider optimizing or parallelizing this step'
                })
            
            # Check for resource bottlenecks
            for resource, usage in metrics['resource_usage'].items():
                if usage > 0.8:  # 80% threshold
                    bottlenecks.append({
                        'step': step_name,
                        'type': f'{resource}_usage',
                        'impact': 'medium',
                        'metric': usage,
                        'suggestion': f'Consider increasing {resource} allocation'
                    })
            
            # Check for retry patterns
            if metrics['retries'] > 2:
                bottlenecks.append({
                    'step': step_name,
                    'type': 'reliability',
                    'impact': 'medium',
                    'metric': metrics['retries'],
                    'suggestion': 'Investigate failure patterns and add error handling'
                })
        
        return bottlenecks
    
    def _generate_optimization_suggestions(
        self,
        step_metrics: Dict[str, Any],
        resource_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate workflow optimization suggestions"""
        suggestions = []
        
        # Analyze step durations
        long_steps = [
            step for step, metrics in step_metrics.items()
            if metrics['duration'] > 300  # 5 minutes threshold
        ]
        if long_steps:
            suggestions.append(
                f"Consider parallelizing or optimizing long-running steps: {', '.join(long_steps)}"
            )
        
        # Analyze resource usage
        for resource, metrics in resource_metrics.items():
            if metrics['p95'] > 0.9:  # 90% threshold
                suggestions.append(
                    f"High {resource} usage detected. Consider scaling {resource} allocation"
                )
            elif metrics['p95'] < 0.3:  # 30% threshold
                suggestions.append(
                    f"Low {resource} utilization. Consider reducing {resource} allocation"
                )
        
        return suggestions
    
    def _store_execution_metrics(
        self,
        workflow_id: str,
        execution_time: float,
        step_metrics: Dict[str, Any],
        resource_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ):
        """Store execution metrics for historical analysis"""
        if workflow_id not in self.workflow_history:
            self.workflow_history[workflow_id] = []
            self.performance_metrics[workflow_id] = {
                'execution_time': [],
                'success_rate': [],
                'error_rate': []
            }
            self.resource_usage[workflow_id] = {
                'cpu': [],
                'memory': [],
                'gpu': []
            }
        
        # Store execution data
        self.workflow_history[workflow_id].append({
            'timestamp': datetime.utcnow(),
            'execution_time': execution_time,
            'step_metrics': step_metrics,
            'resource_metrics': resource_metrics,
            'performance_metrics': performance_metrics
        })
        
        # Update performance metrics
        self.performance_metrics[workflow_id]['execution_time'].append(execution_time)
        self.performance_metrics[workflow_id]['success_rate'].append(
            performance_metrics['success_rate']
        )
        self.performance_metrics[workflow_id]['error_rate'].append(
            performance_metrics['error_rate']
        )
        
        # Update resource usage
        for resource, metrics in resource_metrics.items():
            if resource in self.resource_usage[workflow_id]:
                self.resource_usage[workflow_id][resource].append(metrics['mean']) 