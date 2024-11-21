from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
from ..monitoring.custom_metrics import MetricsCollector
from ..versioning.model_registry import ModelRegistry
import logging

class RetrainingMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.model_registry = ModelRegistry()
        
        # Configure monitoring thresholds
        self.performance_threshold = config.get('performance_threshold', 0.9)
        self.drift_threshold = config.get('drift_threshold', 0.1)
        self.latency_threshold = config.get('latency_threshold', 100)  # ms
        
        # Initialize monitoring state
        self.monitoring_state = {}
    
    def monitor_retraining_pipeline(
        self,
        model_id: str,
        pipeline_run_id: str
    ) -> Dict[str, Any]:
        """Monitor a retraining pipeline execution"""
        try:
            monitoring_data = {
                'start_time': datetime.utcnow(),
                'status': 'running',
                'metrics': {},
                'alerts': [],
                'stages': {}
            }
            
            self.monitoring_state[pipeline_run_id] = monitoring_data
            
            # Record pipeline start
            self.metrics_collector.record_retraining_metric(
                'pipeline_start',
                pipeline_run_id=pipeline_run_id,
                model_id=model_id
            )
            
            return monitoring_data
            
        except Exception as e:
            logging.error(f"Error initializing pipeline monitoring: {str(e)}")
            raise
    
    def update_stage_status(
        self,
        pipeline_run_id: str,
        stage: str,
        status: str,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Update status and metrics for a pipeline stage"""
        try:
            if pipeline_run_id not in self.monitoring_state:
                raise ValueError(f"Unknown pipeline run: {pipeline_run_id}")
            
            monitoring_data = self.monitoring_state[pipeline_run_id]
            
            # Update stage information
            monitoring_data['stages'][stage] = {
                'status': status,
                'completion_time': datetime.utcnow(),
                'metrics': metrics or {}
            }
            
            # Record stage metrics
            if metrics:
                for metric_name, value in metrics.items():
                    self.metrics_collector.record_retraining_metric(
                        f'{stage}_{metric_name}',
                        value,
                        pipeline_run_id=pipeline_run_id
                    )
            
            # Check for alerts
            self._check_stage_alerts(pipeline_run_id, stage, metrics)
            
        except Exception as e:
            logging.error(f"Error updating stage status: {str(e)}")
            raise
    
    def complete_pipeline_monitoring(
        self,
        pipeline_run_id: str,
        status: str,
        final_metrics: Optional[Dict[str, float]] = None
    ):
        """Complete pipeline monitoring and generate report"""
        try:
            if pipeline_run_id not in self.monitoring_state:
                raise ValueError(f"Unknown pipeline run: {pipeline_run_id}")
            
            monitoring_data = self.monitoring_state[pipeline_run_id]
            monitoring_data['status'] = status
            monitoring_data['end_time'] = datetime.utcnow()
            
            if final_metrics:
                monitoring_data['metrics'].update(final_metrics)
                
                # Record final metrics
                for metric_name, value in final_metrics.items():
                    self.metrics_collector.record_retraining_metric(
                        f'final_{metric_name}',
                        value,
                        pipeline_run_id=pipeline_run_id
                    )
            
            # Generate and store report
            report = self._generate_monitoring_report(pipeline_run_id)
            
            # Cleanup monitoring state
            del self.monitoring_state[pipeline_run_id]
            
            return report
            
        except Exception as e:
            logging.error(f"Error completing pipeline monitoring: {str(e)}")
            raise
    
    def _check_stage_alerts(
        self,
        pipeline_run_id: str,
        stage: str,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Check for alerts based on stage metrics"""
        monitoring_data = self.monitoring_state[pipeline_run_id]
        
        if metrics:
            alerts = []
            
            # Check performance
            if 'accuracy' in metrics and metrics['accuracy'] < self.performance_threshold:
                alerts.append({
                    'level': 'warning',
                    'message': f'Low accuracy in {stage}: {metrics["accuracy"]:.3f}'
                })
            
            # Check data drift
            if 'drift_score' in metrics and metrics['drift_score'] > self.drift_threshold:
                alerts.append({
                    'level': 'warning',
                    'message': f'High data drift detected: {metrics["drift_score"]:.3f}'
                })
            
            # Check latency
            if 'latency' in metrics and metrics['latency'] > self.latency_threshold:
                alerts.append({
                    'level': 'warning',
                    'message': f'High latency in {stage}: {metrics["latency"]:.1f}ms'
                })
            
            # Add alerts to monitoring data
            monitoring_data['alerts'].extend(alerts)
            
            # Record alerts
            for alert in alerts:
                self.metrics_collector.record_retraining_metric(
                    'alert',
                    1,
                    pipeline_run_id=pipeline_run_id,
                    alert_level=alert['level']
                )
    
    def _generate_monitoring_report(self, pipeline_run_id: str) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        monitoring_data = self.monitoring_state[pipeline_run_id]
        
        # Calculate pipeline duration
        duration = (monitoring_data['end_time'] - monitoring_data['start_time']).total_seconds()
        
        # Compile stage metrics
        stage_metrics = {}
        for stage, data in monitoring_data['stages'].items():
            stage_metrics[stage] = {
                'status': data['status'],
                'duration': (data['completion_time'] - monitoring_data['start_time']).total_seconds(),
                'metrics': data.get('metrics', {})
            }
        
        # Generate report
        report = {
            'pipeline_run_id': pipeline_run_id,
            'status': monitoring_data['status'],
            'duration': duration,
            'start_time': monitoring_data['start_time'].isoformat(),
            'end_time': monitoring_data['end_time'].isoformat(),
            'metrics': monitoring_data['metrics'],
            'stages': stage_metrics,
            'alerts': monitoring_data['alerts'],
            'summary': {
                'total_stages': len(stage_metrics),
                'successful_stages': sum(1 for s in stage_metrics.values() if s['status'] == 'completed'),
                'failed_stages': sum(1 for s in stage_metrics.values() if s['status'] == 'failed'),
                'total_alerts': len(monitoring_data['alerts'])
            }
        }
        
        return report 