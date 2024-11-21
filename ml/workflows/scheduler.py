from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from croniter import croniter
from .template_system import TemplateManager, WorkflowTemplate
from .manager import WorkflowManager
from ..monitoring.custom_metrics import MetricsCollector

class WorkflowScheduler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.template_manager = TemplateManager()
        self.workflow_manager = WorkflowManager(config)
        self.metrics_collector = MetricsCollector()
        
        self.scheduled_workflows: Dict[str, Dict[str, Any]] = {}
        self.running = False
    
    async def start(self):
        """Start the workflow scheduler"""
        self.running = True
        await self._schedule_loop()
    
    async def stop(self):
        """Stop the workflow scheduler"""
        self.running = False
    
    async def _schedule_loop(self):
        """Main scheduling loop"""
        while self.running:
            try:
                await self._check_schedules()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Error in scheduling loop: {str(e)}")
                self.metrics_collector.record_scheduler_metric(
                    'scheduler_error',
                    1
                )
    
    async def _check_schedules(self):
        """Check and trigger scheduled workflows"""
        current_time = datetime.utcnow()
        
        for template_name, template in self.template_manager.templates.items():
            if template.schedule:
                last_run = self.scheduled_workflows.get(
                    template_name,
                    {}
                ).get('last_run')
                
                if self._should_run(template.schedule, last_run, current_time):
                    try:
                        # Create and execute workflow
                        workflow_id = await self.workflow_manager.create_workflow(
                            template_name,
                            template.parameters
                        )
                        
                        self.scheduled_workflows[template_name] = {
                            'last_run': current_time,
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
                        
                    except Exception as e:
                        logging.error(
                            f"Error scheduling workflow for template {template_name}: {str(e)}"
                        )
                        self.metrics_collector.record_scheduler_metric(
                            'scheduling_error',
                            1,
                            {'template': template_name}
                        )
    
    def _should_run(
        self,
        schedule: str,
        last_run: Optional[datetime],
        current_time: datetime
    ) -> bool:
        """Check if workflow should be run based on schedule"""
        if not last_run:
            return True
        
        try:
            cron = croniter(schedule, last_run)
            next_run = cron.get_next(datetime)
            return current_time >= next_run
        except Exception as e:
            logging.error(f"Error parsing schedule: {str(e)}")
            return False
    
    async def schedule_workflow(
        self,
        template_name: str,
        schedule: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Schedule a workflow with custom schedule"""
        template = self.template_manager.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        # Validate schedule
        if not croniter.is_valid(schedule):
            raise ValueError("Invalid schedule format")
        
        # Update template schedule
        template.schedule = schedule
        if parameters:
            template.parameters.update(parameters)
        
        self.template_manager.save_template(template)
        
        logging.info(f"Scheduled workflow {template_name} with schedule: {schedule}")
    
    async def unschedule_workflow(self, template_name: str):
        """Remove workflow schedule"""
        template = self.template_manager.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        template.schedule = None
        self.template_manager.save_template(template)
        
        if template_name in self.scheduled_workflows:
            del self.scheduled_workflows[template_name]
        
        logging.info(f"Unscheduled workflow {template_name}")
    
    def get_scheduled_workflows(self) -> List[Dict[str, Any]]:
        """Get list of scheduled workflows"""
        scheduled = []
        for template_name, template in self.template_manager.templates.items():
            if template.schedule:
                scheduled.append({
                    'template_name': template_name,
                    'schedule': template.schedule,
                    'last_run': self.scheduled_workflows.get(
                        template_name,
                        {}
                    ).get('last_run'),
                    'parameters': template.parameters
                })
        return scheduled 