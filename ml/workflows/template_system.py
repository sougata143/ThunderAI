from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yaml
import json
import logging
from pathlib import Path

@dataclass
class WorkflowStep:
    """Definition of a workflow step"""
    name: str
    function: str
    args: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_interval": 60
    })
    timeout: int = 3600
    resources: Dict[str, Any] = field(default_factory=lambda: {
        "cpu": "1",
        "memory": "1Gi"
    })

@dataclass
class WorkflowTemplate:
    """Workflow template definition"""
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    schedule: Optional[str] = None
    timeout: Optional[int] = None
    notifications: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'steps': [
                {
                    'name': step.name,
                    'function': step.function,
                    'args': step.args,
                    'dependencies': step.dependencies,
                    'retry_policy': step.retry_policy,
                    'timeout': step.timeout,
                    'resources': step.resources
                }
                for step in self.steps
            ],
            'schedule': self.schedule,
            'timeout': self.timeout,
            'notifications': self.notifications,
            'tags': self.tags,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowTemplate':
        """Create template from dictionary"""
        steps = [
            WorkflowStep(**step_data)
            for step_data in data.pop('steps', [])
        ]
        return cls(steps=steps, **data)

class TemplateManager:
    def __init__(self, template_dir: str = 'workflows/templates'):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from files"""
        for file_path in self.template_dir.glob('*.yaml'):
            try:
                with open(file_path, 'r') as f:
                    template_data = yaml.safe_load(f)
                    template = WorkflowTemplate.from_dict(template_data)
                    self.templates[template.name] = template
            except Exception as e:
                logging.error(f"Error loading template {file_path}: {str(e)}")
    
    def save_template(self, template: WorkflowTemplate):
        """Save template to file"""
        file_path = self.template_dir / f"{template.name.lower()}.yaml"
        try:
            with open(file_path, 'w') as f:
                yaml.dump(template.to_dict(), f)
            self.templates[template.name] = template
        except Exception as e:
            logging.error(f"Error saving template {template.name}: {str(e)}")
            raise
    
    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get template by name"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        return [
            {
                'name': template.name,
                'description': template.description,
                'version': template.version,
                'tags': template.tags
            }
            for template in self.templates.values()
        ]
    
    def validate_template(self, template: WorkflowTemplate) -> List[str]:
        """Validate template configuration"""
        errors = []
        
        # Check required fields
        if not template.name:
            errors.append("Template name is required")
        if not template.steps:
            errors.append("Template must have at least one step")
        
        # Validate steps
        step_names = set()
        for step in template.steps:
            if not step.name:
                errors.append("Step name is required")
            if not step.function:
                errors.append(f"Function is required for step {step.name}")
            if step.name in step_names:
                errors.append(f"Duplicate step name: {step.name}")
            step_names.add(step.name)
            
            # Validate dependencies
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(
                        f"Invalid dependency {dep} for step {step.name}"
                    )
        
        # Validate schedule if provided
        if template.schedule:
            try:
                from croniter import croniter
                if not croniter.is_valid(template.schedule):
                    errors.append("Invalid schedule format")
            except ImportError:
                logging.warning("croniter not installed, skipping schedule validation")
        
        return errors
    
    def create_template_from_workflow(
        self,
        workflow_id: str,
        workflow_data: Dict[str, Any]
    ) -> WorkflowTemplate:
        """Create template from successful workflow execution"""
        steps = []
        for step_data in workflow_data['steps'].values():
            step = WorkflowStep(
                name=step_data['name'],
                function=step_data.get('function', ''),
                args=step_data.get('args', {}),
                dependencies=step_data.get('dependencies', []),
                retry_policy=step_data.get('retry_policy', {}),
                timeout=step_data.get('timeout', 3600),
                resources=step_data.get('resources', {})
            )
            steps.append(step)
        
        template = WorkflowTemplate(
            name=f"template_from_{workflow_id}",
            description=f"Template created from workflow {workflow_id}",
            version="1.0.0",
            steps=steps,
            parameters=workflow_data.get('config', {})
        )
        
        return template
    
    def export_template(
        self,
        template_name: str,
        format: str = 'yaml'
    ) -> str:
        """Export template to specified format"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        template_dict = template.to_dict()
        
        if format == 'yaml':
            return yaml.dump(template_dict)
        elif format == 'json':
            return json.dumps(template_dict, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}") 