from typing import Dict, Any, List, Optional
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
import logging
from datetime import datetime, timedelta

from ..monitoring.model_performance_monitor import ModelPerformanceMonitor
from ..retraining.orchestrator import RetrainingOrchestrator
from ..monitoring.custom_metrics import MetricsCollector
from ..versioning.model_registry import ModelRegistry

class WorkflowOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.model_registry = ModelRegistry()
        self.performance_monitor = ModelPerformanceMonitor({})
        self.retraining_orchestrator = RetrainingOrchestrator({})
        
        # Initialize workflow states
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
    
    @task
    def initialize_workflow(
        self,
        workflow_id: str,
        workflow_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize a new workflow"""
        try:
            workflow_info = {
                'id': workflow_id,
                'type': workflow_type,
                'status': 'initializing',
                'start_time': datetime.utcnow(),
                'config': config,
                'steps': [],
                'metrics': {}
            }
            
            self.active_workflows[workflow_id] = workflow_info
            logging.info(f"Initialized workflow {workflow_id} of type {workflow_type}")
            
            return workflow_info
            
        except Exception as e:
            logging.error(f"Error initializing workflow: {str(e)}")
            raise
    
    @task
    def execute_workflow_step(
        self,
        workflow_id: str,
        step_name: str,
        step_func: callable,
        step_args: Dict[str, Any]
    ) -> Any:
        """Execute a single workflow step"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Record step start
            step_info = {
                'name': step_name,
                'status': 'running',
                'start_time': datetime.utcnow()
            }
            workflow['steps'].append(step_info)
            
            # Execute step
            result = step_func(**step_args)
            
            # Update step status
            step_info.update({
                'status': 'completed',
                'end_time': datetime.utcnow(),
                'result': result
            })
            
            return result
            
        except Exception as e:
            step_info.update({
                'status': 'failed',
                'end_time': datetime.utcnow(),
                'error': str(e)
            })
            logging.error(f"Error executing workflow step: {str(e)}")
            raise
    
    @task
    def complete_workflow(
        self,
        workflow_id: str,
        status: str = 'completed'
    ) -> Dict[str, Any]:
        """Complete a workflow and record its history"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Update workflow status
            workflow.update({
                'status': status,
                'end_time': datetime.utcnow()
            })
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            # Record metrics
            self._record_workflow_metrics(workflow)
            
            return workflow
            
        except Exception as e:
            logging.error(f"Error completing workflow: {str(e)}")
            raise
    
    @flow
    def create_monitoring_workflow(
        self,
        model_id: str,
        schedule: Optional[str] = None
    ) -> str:
        """Create a monitoring workflow for a model"""
        workflow_id = f"monitoring_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize workflow
        workflow = self.initialize_workflow(
            workflow_id=workflow_id,
            workflow_type='monitoring',
            config={'model_id': model_id}
        )
        
        # Define workflow steps
        steps = [
            {
                'name': 'check_performance',
                'func': self.performance_monitor.get_current_metrics,
                'args': {'model_id': model_id}
            },
            {
                'name': 'evaluate_metrics',
                'func': self.performance_monitor.evaluate_metrics,
                'args': {'model_id': model_id}
            },
            {
                'name': 'update_dashboards',
                'func': self.metrics_collector.update_dashboard_metrics,
                'args': {'model_id': model_id}
            }
        ]
        
        # Execute steps
        for step in steps:
            self.execute_workflow_step(
                workflow_id=workflow_id,
                step_name=step['name'],
                step_func=step['func'],
                step_args=step['args']
            )
        
        # Complete workflow
        self.complete_workflow(workflow_id)
        
        # Create deployment if schedule is provided
        if schedule:
            self._create_deployment(
                workflow_id=workflow_id,
                schedule=schedule
            )
        
        return workflow_id
    
    @flow
    def create_retraining_workflow(
        self,
        model_id: str,
        training_config: Dict[str, Any]
    ) -> str:
        """Create a retraining workflow for a model"""
        workflow_id = f"retraining_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize workflow
        workflow = self.initialize_workflow(
            workflow_id=workflow_id,
            workflow_type='retraining',
            config={
                'model_id': model_id,
                'training_config': training_config
            }
        )
        
        # Define workflow steps
        steps = [
            {
                'name': 'prepare_data',
                'func': self.retraining_orchestrator.prepare_training_data,
                'args': {'model_id': model_id}
            },
            {
                'name': 'train_model',
                'func': self.retraining_orchestrator.train_model,
                'args': {
                    'model_id': model_id,
                    'config': training_config
                }
            },
            {
                'name': 'evaluate_model',
                'func': self.retraining_orchestrator.evaluate_model,
                'args': {'model_id': model_id}
            },
            {
                'name': 'deploy_model',
                'func': self.retraining_orchestrator.deploy_model,
                'args': {'model_id': model_id}
            }
        ]
        
        # Execute steps
        for step in steps:
            self.execute_workflow_step(
                workflow_id=workflow_id,
                step_name=step['name'],
                step_func=step['func'],
                step_args=step['args']
            )
        
        # Complete workflow
        self.complete_workflow(workflow_id)
        
        return workflow_id
    
    def _create_deployment(
        self,
        workflow_id: str,
        schedule: str
    ):
        """Create a Prefect deployment for the workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        deployment = Deployment.build_from_flow(
            flow=self.create_monitoring_workflow,
            name=f"deployment_{workflow_id}",
            schedule=CronSchedule(cron=schedule),
            tags=[workflow['type'], workflow_id]
        )
        deployment.apply()
    
    def _record_workflow_metrics(self, workflow: Dict[str, Any]):
        """Record workflow metrics"""
        duration = (workflow['end_time'] - workflow['start_time']).total_seconds()
        
        self.metrics_collector.record_workflow_metric(
            workflow_type=workflow['type'],
            workflow_id=workflow['id'],
            duration=duration,
            status=workflow['status']
        )
        
        # Record step metrics
        for step in workflow['steps']:
            if 'start_time' in step and 'end_time' in step:
                step_duration = (step['end_time'] - step['start_time']).total_seconds()
                self.metrics_collector.record_workflow_metric(
                    workflow_type=workflow['type'],
                    workflow_id=workflow['id'],
                    step_name=step['name'],
                    duration=step_duration,
                    status=step['status']
                ) 