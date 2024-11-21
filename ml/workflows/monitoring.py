from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from ..monitoring.custom_metrics import MetricsCollector
from prometheus_client import Counter, Gauge, Histogram

@dataclass
class WorkflowMetrics:
    """Workflow metrics data"""
    total_duration: float
    step_durations: Dict[str, float]
    success_rate: float
    error_count: int
    resource_usage: Dict[str, float]
    custom_metrics: Dict[str, float]

class WorkflowMonitor:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        
        # Initialize Prometheus metrics
        self.workflow_duration = Histogram(
            'workflow_duration_seconds',
            'Workflow execution duration in seconds',
            ['workflow_type', 'workflow_id']
        )
        
        self.workflow_step_duration = Histogram(
            'workflow_step_duration_seconds',
            'Workflow step execution duration in seconds',
            ['workflow_type', 'workflow_id', 'step_name']
        )
        
        self.workflow_success = Counter(
            'workflow_success_total',
            'Number of successful workflow executions',
            ['workflow_type']
        )
        
        self.workflow_failure = Counter(
            'workflow_failure_total',
            'Number of failed workflow executions',
            ['workflow_type', 'error_type']
        )
        
        self.workflow_active = Gauge(
            'workflow_active',
            'Number of currently active workflows',
            ['workflow_type']
        )
    
    def start_workflow_monitoring(
        self,
        workflow_id: str,
        workflow_type: str
    ) -> Dict[str, Any]:
        """Start monitoring a workflow execution"""
        monitoring_data = {
            'workflow_id': workflow_id,
            'workflow_type': workflow_type,
            'start_time': datetime.utcnow(),
            'steps': {},
            'metrics': {},
            'status': 'running'
        }
        
        self.workflow_active.labels(workflow_type=workflow_type).inc()
        
        return monitoring_data
    
    def record_step_execution(
        self,
        monitoring_data: Dict[str, Any],
        step_name: str,
        duration: float,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Record execution metrics for a workflow step"""
        monitoring_data['steps'][step_name] = {
            'duration': duration,
            'metrics': metrics or {},
            'timestamp': datetime.utcnow()
        }
        
        # Record step duration metric
        self.workflow_step_duration.labels(
            workflow_type=monitoring_data['workflow_type'],
            workflow_id=monitoring_data['workflow_id'],
            step_name=step_name
        ).observe(duration)
        
        # Record custom metrics
        if metrics:
            for metric_name, value in metrics.items():
                self.metrics_collector.record_workflow_metric(
                    workflow_type=monitoring_data['workflow_type'],
                    workflow_id=monitoring_data['workflow_id'],
                    metric_name=f"{step_name}_{metric_name}",
                    value=value
                )
    
    def complete_workflow_monitoring(
        self,
        monitoring_data: Dict[str, Any],
        status: str = 'completed',
        error: Optional[str] = None
    ) -> WorkflowMetrics:
        """Complete workflow monitoring and generate metrics"""
        end_time = datetime.utcnow()
        monitoring_data['end_time'] = end_time
        monitoring_data['status'] = status
        
        if error:
            monitoring_data['error'] = error
        
        # Calculate total duration
        duration = (end_time - monitoring_data['start_time']).total_seconds()
        
        # Record workflow completion metrics
        if status == 'completed':
            self.workflow_success.labels(
                workflow_type=monitoring_data['workflow_type']
            ).inc()
        else:
            self.workflow_failure.labels(
                workflow_type=monitoring_data['workflow_type'],
                error_type=error or 'unknown'
            ).inc()
        
        self.workflow_duration.labels(
            workflow_type=monitoring_data['workflow_type'],
            workflow_id=monitoring_data['workflow_id']
        ).observe(duration)
        
        self.workflow_active.labels(
            workflow_type=monitoring_data['workflow_type']
        ).dec()
        
        # Generate workflow metrics
        metrics = WorkflowMetrics(
            total_duration=duration,
            step_durations={
                step: data['duration']
                for step, data in monitoring_data['steps'].items()
            },
            success_rate=1.0 if status == 'completed' else 0.0,
            error_count=1 if error else 0,
            resource_usage=self._get_resource_usage(),
            custom_metrics=monitoring_data.get('metrics', {})
        )
        
        return metrics
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage metrics"""
        import psutil
        
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        usage = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used,
            'memory_available': memory.available
        }
        
        if torch.cuda.is_available():
            usage['gpu_memory_used'] = torch.cuda.memory_allocated()
            usage['gpu_memory_cached'] = torch.cuda.memory_reserved()
        
        return usage
    
    def get_workflow_metrics(
        self,
        workflow_type: str,
        time_range: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Get aggregated workflow metrics for a time range"""
        # Query metrics from monitoring system
        start_time = datetime.utcnow() - time_range
        
        metrics = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_duration': 0.0,
            'step_metrics': {},
            'error_distribution': {},
            'resource_usage': {
                'cpu': [],
                'memory': [],
                'gpu': []
            }
        }
        
        # Implement metric collection logic
        
        return metrics
    
    def generate_workflow_report(
        self,
        workflow_id: str,
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """Generate a detailed workflow execution report"""
        report = {
            'workflow_id': workflow_id,
            'execution_summary': {},
            'step_details': {},
            'resource_utilization': {},
            'errors': []
        }
        
        if include_metrics:
            report['metrics'] = {}
        
        # Implement report generation logic
        
        return report 