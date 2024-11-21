from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import logging
from croniter import croniter
from .scheduler import WorkflowScheduler
from .template_system import WorkflowTemplate
from ..monitoring.custom_metrics import MetricsCollector

class AdvancedWorkflowScheduler(WorkflowScheduler):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics_collector = MetricsCollector()
        
        # Additional scheduling features
        self.dependencies: Dict[str, List[str]] = {}
        self.retries: Dict[str, Dict[str, Any]] = {}
        self.priorities: Dict[str, int] = {}
        self.resource_limits: Dict[str, Dict[str, Any]] = {}
    
    async def schedule_with_dependencies(
        self,
        template_name: str,
        schedule: str,
        dependencies: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Schedule workflow with dependencies"""
        await self.schedule_workflow(template_name, schedule, parameters)
        self.dependencies[template_name] = dependencies
        
        logging.info(
            f"Scheduled workflow {template_name} with dependencies: {dependencies}"
        )
    
    async def schedule_with_retry(
        self,
        template_name: str,
        schedule: str,
        retry_config: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Schedule workflow with retry policy"""
        await self.schedule_workflow(template_name, schedule, parameters)
        self.retries[template_name] = retry_config
        
        logging.info(
            f"Scheduled workflow {template_name} with retry config: {retry_config}"
        )
    
    async def schedule_with_priority(
        self,
        template_name: str,
        schedule: str,
        priority: int,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Schedule workflow with priority"""
        await self.schedule_workflow(template_name, schedule, parameters)
        self.priorities[template_name] = priority
        
        logging.info(
            f"Scheduled workflow {template_name} with priority: {priority}"
        )
    
    async def schedule_with_resources(
        self,
        template_name: str,
        schedule: str,
        resource_limits: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Schedule workflow with resource limits"""
        await self.schedule_workflow(template_name, schedule, parameters)
        self.resource_limits[template_name] = resource_limits
        
        logging.info(
            f"Scheduled workflow {template_name} with resource limits: {resource_limits}"
        )
    
    async def _check_schedules(self):
        """Enhanced schedule checking with dependencies and priorities"""
        current_time = datetime.utcnow()
        scheduled_workflows = []
        
        # Check all templates
        for template_name, template in self.template_manager.templates.items():
            if template.schedule:
                last_run = self.scheduled_workflows.get(
                    template_name,
                    {}
                ).get('last_run')
                
                if self._should_run(template.schedule, last_run, current_time):
                    # Check dependencies
                    if self._check_dependencies(template_name):
                        scheduled_workflows.append({
                            'template_name': template_name,
                            'priority': self.priorities.get(template_name, 0),
                            'resource_limits': self.resource_limits.get(template_name, {})
                        })
        
        # Sort by priority
        scheduled_workflows.sort(key=lambda x: x['priority'], reverse=True)
        
        # Execute workflows
        for workflow in scheduled_workflows:
            await self._execute_workflow(workflow)
    
    def _check_dependencies(self, template_name: str) -> bool:
        """Check if all dependencies are satisfied"""
        if template_name not in self.dependencies:
            return True
        
        for dep in self.dependencies[template_name]:
            last_run = self.scheduled_workflows.get(dep, {}).get('last_run')
            if not last_run:
                return False
        
        return True
    
    async def _execute_workflow(self, workflow: Dict[str, Any]):
        """Execute workflow with retry logic and resource limits"""
        template_name = workflow['template_name']
        retry_config = self.retries.get(template_name, {
            'max_retries': 3,
            'retry_delay': 60
        })
        
        retries = 0
        while retries <= retry_config['max_retries']:
            try:
                # Check resource limits
                if self._check_resources(workflow['resource_limits']):
                    # Create and execute workflow
                    workflow_id = await self.workflow_manager.create_workflow(
                        template_name,
                        self.template_manager.get_template(template_name).parameters
                    )
                    
                    self.scheduled_workflows[template_name] = {
                        'last_run': datetime.utcnow(),
                        'workflow_id': workflow_id
                    }
                    
                    # Execute workflow asynchronously
                    asyncio.create_task(
                        self.workflow_manager.execute_workflow(workflow_id)
                    )
                    
                    self.metrics_collector.record_scheduler_metric(
                        'workflow_scheduled',
                        1,
                        {'template': template_name}
                    )
                    break
                    
            except Exception as e:
                logging.error(
                    f"Error executing workflow {template_name}: {str(e)}"
                )
                retries += 1
                if retries <= retry_config['max_retries']:
                    await asyncio.sleep(retry_config['retry_delay'])
                else:
                    self.metrics_collector.record_scheduler_metric(
                        'workflow_failed',
                        1,
                        {'template': template_name}
                    )
    
    def _check_resources(self, resource_limits: Dict[str, Any]) -> bool:
        """Check if required resources are available"""
        # Implement resource checking logic
        return True
    
    def get_workflow_status(self, template_name: str) -> Dict[str, Any]:
        """Get detailed workflow status"""
        status = {
            'schedule': self.template_manager.get_template(template_name).schedule,
            'last_run': self.scheduled_workflows.get(
                template_name,
                {}
            ).get('last_run'),
            'dependencies': self.dependencies.get(template_name, []),
            'priority': self.priorities.get(template_name, 0),
            'resource_limits': self.resource_limits.get(template_name, {}),
            'retry_config': self.retries.get(template_name, {})
        }
        
        return status 