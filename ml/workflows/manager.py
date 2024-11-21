from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from .templates import WorkflowTemplate, WorkflowTemplateManager
from .analytics import WorkflowAnalytics
from .monitoring import WorkflowMonitor
from ..monitoring.custom_metrics import MetricsCollector

class WorkflowManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.template_manager = WorkflowTemplateManager()
        self.analytics = WorkflowAnalytics()
        self.monitor = WorkflowMonitor()
        self.metrics_collector = MetricsCollector()
        
        # Initialize workflow states
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.optimization_cache: Dict[str, Any] = {}
    
    async def create_workflow(
        self,
        template_name: str,
        workflow_config: Dict[str, Any]
    ) -> str:
        """Create a new workflow from template"""
        try:
            # Get template
            template = self.template_manager.get_template(template_name)
            if not template:
                raise ValueError(f"Template {template_name} not found")
            
            # Generate workflow ID
            workflow_id = f"{template_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize workflow
            workflow = {
                'id': workflow_id,
                'template': template_name,
                'config': workflow_config,
                'status': 'initializing',
                'start_time': datetime.utcnow(),
                'steps': {},
                'metrics': {}
            }
            
            # Apply optimizations
            optimized_config = self._optimize_workflow(template, workflow_config)
            workflow['optimized_config'] = optimized_config
            
            # Store workflow
            self.active_workflows[workflow_id] = workflow
            
            # Start monitoring
            await self.monitor.start_workflow_monitoring(workflow_id)
            
            return workflow_id
            
        except Exception as e:
            logging.error(f"Error creating workflow: {str(e)}")
            raise
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a workflow"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow['status'] = 'running'
            template = self.template_manager.get_template(workflow['template'])
            
            # Execute steps
            for step in template.steps:
                try:
                    step_result = await self._execute_step(
                        workflow_id,
                        step,
                        input_data
                    )
                    workflow['steps'][step.name] = step_result
                    
                except Exception as e:
                    workflow['steps'][step.name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    workflow['status'] = 'failed'
                    raise
            
            workflow['status'] = 'completed'
            workflow['end_time'] = datetime.utcnow()
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            # Generate analytics
            analytics = await self.analytics.analyze_workflow_performance(workflow_id)
            
            return {
                'workflow_id': workflow_id,
                'status': workflow['status'],
                'steps': workflow['steps'],
                'analytics': analytics
            }
            
        except Exception as e:
            logging.error(f"Error executing workflow: {str(e)}")
            raise
    
    async def _execute_step(
        self,
        workflow_id: str,
        step: Any,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        start_time = datetime.utcnow()
        
        step_result = {
            'name': step.name,
            'start_time': start_time,
            'status': 'running'
        }
        
        try:
            # Execute step function
            result = await step.function(
                input_data,
                **step.args
            )
            
            step_result.update({
                'status': 'completed',
                'end_time': datetime.utcnow(),
                'result': result
            })
            
        except Exception as e:
            step_result.update({
                'status': 'failed',
                'end_time': datetime.utcnow(),
                'error': str(e)
            })
            raise
        
        finally:
            # Record metrics
            duration = (step_result['end_time'] - start_time).total_seconds()
            self.metrics_collector.record_workflow_metric(
                workflow_id=workflow_id,
                step_name=step.name,
                duration=duration,
                status=step_result['status']
            )
        
        return step_result
    
    def _optimize_workflow(
        self,
        template: WorkflowTemplate,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize workflow configuration based on historical data"""
        template_id = template.name
        
        if template_id in self.optimization_cache:
            cached_optimization = self.optimization_cache[template_id]
            if self._is_optimization_valid(cached_optimization):
                return cached_optimization['config']
        
        # Analyze historical executions
        historical_data = self._get_historical_executions(template_id)
        if not historical_data:
            return config
        
        # Optimize parameters
        optimized_config = self._optimize_parameters(
            config,
            historical_data
        )
        
        # Cache optimization
        self.optimization_cache[template_id] = {
            'config': optimized_config,
            'timestamp': datetime.utcnow()
        }
        
        return optimized_config
    
    def _optimize_parameters(
        self,
        config: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize workflow parameters based on historical performance"""
        optimized_config = config.copy()
        
        # Analyze successful executions
        successful_executions = [
            execution for execution in historical_data
            if execution['status'] == 'completed'
        ]
        
        if not successful_executions:
            return config
        
        # Find optimal parameters
        param_performance = {}
        for execution in successful_executions:
            for param, value in execution['config'].items():
                if param not in param_performance:
                    param_performance[param] = []
                param_performance[param].append({
                    'value': value,
                    'duration': execution['duration'],
                    'success_rate': execution['success_rate']
                })
        
        # Optimize each parameter
        for param, performances in param_performance.items():
            if param in config:
                optimal_value = self._find_optimal_value(performances)
                optimized_config[param] = optimal_value
        
        return optimized_config
    
    def _find_optimal_value(
        self,
        performances: List[Dict[str, Any]]
    ) -> Any:
        """Find optimal parameter value based on performance metrics"""
        # Sort by success rate and duration
        sorted_performances = sorted(
            performances,
            key=lambda x: (x['success_rate'], -x['duration']),
            reverse=True
        )
        
        if sorted_performances:
            return sorted_performances[0]['value']
        return None
    
    def _is_optimization_valid(
        self,
        optimization: Dict[str, Any]
    ) -> bool:
        """Check if cached optimization is still valid"""
        cache_age = datetime.utcnow() - optimization['timestamp']
        return cache_age < timedelta(hours=24)
    
    def _get_historical_executions(
        self,
        template_id: str
    ) -> List[Dict[str, Any]]:
        """Get historical workflow executions for a template"""
        return [
            workflow for workflow in self.workflow_history
            if workflow['template'] == template_id
        ] 