from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..monitoring.custom_metrics import MetricsCollector
import logging

class WorkflowAnalytics:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
    
    def analyze_workflow_performance(
        self,
        workflow_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Analyze workflow performance metrics"""
        try:
            # Get workflow execution data
            workflow_data = self._get_workflow_data(workflow_id, time_range)
            
            # Calculate performance metrics
            performance_metrics = {
                'execution_time': self._analyze_execution_time(workflow_data),
                'success_rate': self._analyze_success_rate(workflow_data),
                'resource_utilization': self._analyze_resource_usage(workflow_data),
                'bottlenecks': self._identify_bottlenecks(workflow_data),
                'error_patterns': self._analyze_error_patterns(workflow_data)
            }
            
            # Record analytics metrics
            self._record_analytics_metrics(workflow_id, performance_metrics)
            
            return performance_metrics
            
        except Exception as e:
            logging.error(f"Error analyzing workflow performance: {str(e)}")
            raise
    
    def generate_workflow_insights(
        self,
        workflow_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate insights from workflow execution data"""
        insights = []
        
        # Analyze execution patterns
        execution_patterns = self._analyze_execution_patterns(workflow_data)
        if execution_patterns:
            insights.extend(execution_patterns)
        
        # Analyze resource efficiency
        resource_insights = self._analyze_resource_efficiency(workflow_data)
        if resource_insights:
            insights.extend(resource_insights)
        
        # Analyze failure patterns
        failure_insights = self._analyze_failure_patterns(workflow_data)
        if failure_insights:
            insights.extend(failure_insights)
        
        return insights
    
    def _analyze_execution_time(
        self,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze workflow execution time metrics"""
        execution_times = []
        step_times = {}
        
        for execution in workflow_data['executions']:
            duration = (execution['end_time'] - execution['start_time']).total_seconds()
            execution_times.append(duration)
            
            # Analyze step durations
            for step, step_data in execution['steps'].items():
                if step not in step_times:
                    step_times[step] = []
                step_times[step].append(step_data['duration'])
        
        return {
            'mean_duration': np.mean(execution_times),
            'p95_duration': np.percentile(execution_times, 95),
            'step_durations': {
                step: {
                    'mean': np.mean(times),
                    'p95': np.percentile(times, 95)
                }
                for step, times in step_times.items()
            }
        }
    
    def _analyze_success_rate(
        self,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze workflow success rate metrics"""
        total_executions = len(workflow_data['executions'])
        successful_executions = sum(
            1 for e in workflow_data['executions']
            if e['status'] == 'completed'
        )
        
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        return {
            'success_rate': success_rate,
            'total_executions': total_executions,
            'successful_executions': successful_executions
        }
    
    def _analyze_resource_usage(
        self,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze resource utilization patterns"""
        resource_metrics = {
            'cpu': [],
            'memory': [],
            'gpu': []
        }
        
        for execution in workflow_data['executions']:
            for metric in execution.get('resource_metrics', []):
                for resource, value in metric.items():
                    if resource in resource_metrics:
                        resource_metrics[resource].append(value)
        
        return {
            resource: {
                'mean': np.mean(values),
                'max': np.max(values),
                'p95': np.percentile(values, 95)
            }
            for resource, values in resource_metrics.items()
            if values
        }
    
    def _identify_bottlenecks(
        self,
        workflow_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify workflow bottlenecks"""
        bottlenecks = []
        step_metrics = {}
        
        # Analyze step execution patterns
        for execution in workflow_data['executions']:
            for step, step_data in execution['steps'].items():
                if step not in step_metrics:
                    step_metrics[step] = []
                step_metrics[step].append({
                    'duration': step_data['duration'],
                    'resource_usage': step_data.get('resource_usage', {}),
                    'status': step_data['status']
                })
        
        # Identify potential bottlenecks
        for step, metrics in step_metrics.items():
            durations = [m['duration'] for m in metrics]
            mean_duration = np.mean(durations)
            p95_duration = np.percentile(durations, 95)
            
            if p95_duration > mean_duration * 2:
                bottlenecks.append({
                    'step': step,
                    'mean_duration': mean_duration,
                    'p95_duration': p95_duration,
                    'impact': 'high' if p95_duration > mean_duration * 3 else 'medium'
                })
        
        return bottlenecks
    
    def _analyze_error_patterns(
        self,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze workflow error patterns"""
        error_patterns = {}
        
        for execution in workflow_data['executions']:
            if execution['status'] == 'failed':
                error = execution.get('error', {})
                error_type = error.get('type', 'unknown')
                
                if error_type not in error_patterns:
                    error_patterns[error_type] = {
                        'count': 0,
                        'steps': set(),
                        'examples': []
                    }
                
                error_patterns[error_type]['count'] += 1
                error_patterns[error_type]['steps'].add(error.get('step'))
                error_patterns[error_type]['examples'].append(error.get('message'))
        
        return error_patterns
    
    def _record_analytics_metrics(
        self,
        workflow_id: str,
        metrics: Dict[str, Any]
    ):
        """Record workflow analytics metrics"""
        for metric_type, values in metrics.items():
            if isinstance(values, dict):
                for name, value in values.items():
                    if isinstance(value, (int, float)):
                        self.metrics_collector.record_workflow_metric(
                            workflow_id=workflow_id,
                            metric_name=f"{metric_type}_{name}",
                            value=value
                        ) 