from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import yaml
import logging

@dataclass
class WorkflowStep:
    """Definition of a workflow step"""
    name: str
    function: str
    args: Dict[str, Any]
    retry_policy: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    dependencies: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'function': self.function,
            'args': self.args,
            'retry_policy': self.retry_policy,
            'timeout': self.timeout,
            'dependencies': self.dependencies or []
        }

@dataclass
class WorkflowTemplate:
    """Workflow template definition"""
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    schedule: Optional[str] = None
    timeout: Optional[int] = None
    notifications: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'steps': [step.to_dict() for step in self.steps],
            'schedule': self.schedule,
            'timeout': self.timeout,
            'notifications': self.notifications,
            'tags': self.tags or []
        }

class WorkflowTemplateManager:
    def __init__(self, template_dir: str = 'workflows/templates'):
        self.template_dir = template_dir
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load workflow templates from YAML files"""
        try:
            import os
            for filename in os.listdir(self.template_dir):
                if filename.endswith('.yaml'):
                    template_path = os.path.join(self.template_dir, filename)
                    with open(template_path, 'r') as f:
                        template_dict = yaml.safe_load(f)
                        template = self._parse_template(template_dict)
                        self.templates[template.name] = template
        except Exception as e:
            logging.error(f"Error loading templates: {str(e)}")
    
    def _parse_template(self, template_dict: Dict[str, Any]) -> WorkflowTemplate:
        """Parse template dictionary into WorkflowTemplate object"""
        steps = []
        for step_dict in template_dict.get('steps', []):
            step = WorkflowStep(
                name=step_dict['name'],
                function=step_dict['function'],
                args=step_dict.get('args', {}),
                retry_policy=step_dict.get('retry_policy'),
                timeout=step_dict.get('timeout'),
                dependencies=step_dict.get('dependencies')
            )
            steps.append(step)
        
        return WorkflowTemplate(
            name=template_dict['name'],
            description=template_dict['description'],
            version=template_dict['version'],
            steps=steps,
            schedule=template_dict.get('schedule'),
            timeout=template_dict.get('timeout'),
            notifications=template_dict.get('notifications'),
            tags=template_dict.get('tags')
        )
    
    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get workflow template by name"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available workflow templates"""
        return [
            {
                'name': template.name,
                'description': template.description,
                'version': template.version,
                'tags': template.tags
            }
            for template in self.templates.values()
        ]
    
    def create_template(self, template_dict: Dict[str, Any]) -> WorkflowTemplate:
        """Create a new workflow template"""
        template = self._parse_template(template_dict)
        self.templates[template.name] = template
        
        # Save template to file
        template_path = os.path.join(
            self.template_dir,
            f"{template.name.lower().replace(' ', '_')}.yaml"
        )
        with open(template_path, 'w') as f:
            yaml.dump(template.to_dict(), f)
        
        return template
    
    def update_template(
        self,
        name: str,
        template_dict: Dict[str, Any]
    ) -> Optional[WorkflowTemplate]:
        """Update an existing workflow template"""
        if name not in self.templates:
            return None
        
        template = self._parse_template(template_dict)
        self.templates[name] = template
        
        # Update template file
        template_path = os.path.join(
            self.template_dir,
            f"{name.lower().replace(' ', '_')}.yaml"
        )
        with open(template_path, 'w') as f:
            yaml.dump(template.to_dict(), f)
        
        return template
    
    def delete_template(self, name: str) -> bool:
        """Delete a workflow template"""
        if name not in self.templates:
            return False
        
        del self.templates[name]
        
        # Delete template file
        template_path = os.path.join(
            self.template_dir,
            f"{name.lower().replace(' ', '_')}.yaml"
        )
        try:
            os.remove(template_path)
            return True
        except Exception as e:
            logging.error(f"Error deleting template file: {str(e)}")
            return False
    
    def validate_template(self, template: WorkflowTemplate) -> List[str]:
        """Validate workflow template"""
        errors = []
        
        # Check for required fields
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
            if step.dependencies:
                for dep in step.dependencies:
                    if dep not in step_names:
                        errors.append(
                            f"Invalid dependency {dep} for step {step.name}"
                        )
        
        return errors 