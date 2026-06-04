"""
Configuration management for CHPC Job Manager
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Manages configuration loading and access"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.task_types = self._load_task_types()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please copy config.yaml.example to config.yaml and configure it."
            )
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_task_types(self) -> Dict[str, Any]:
        """Load task types configuration"""
        task_types_path = self.config.get('task_types_config', 'task_types.yaml')
        if not os.path.isabs(task_types_path):
            task_types_path = os.path.join(
                os.path.dirname(os.path.abspath(self.config_path)),
                task_types_path,
            )
        
        if not os.path.exists(task_types_path):
            raise FileNotFoundError(
                f"Task types config not found: {task_types_path}\n"
                f"Please copy task_types.yaml.example to task_types.yaml and configure it."
            )
        
        with open(task_types_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(key, default)
    
    def get_task_type(self, task_type: str) -> Dict[str, Any]:
        """Get configuration for a specific task type"""
        if task_type not in self.task_types:
            raise ValueError(f"Unknown task type: {task_type}")
        return self.task_types[task_type]
    
    def get_completion_file(self, task_type: str, arguments: Dict[str, Any]) -> str:
        """
        Generate completion file path for a task by filling in the template
        with task arguments
        """
        task_config = self.get_task_type(task_type)
        template = task_config.get('completion_file_template')
        
        if not template:
            return None
        
        try:
            return template.format(**arguments)
        except KeyError as e:
            raise ValueError(
                f"Completion file template requires argument {e} "
                f"but it was not provided in task arguments"
            )
    
    @property
    def username(self) -> str:
        """Get CHPC username"""
        return self.config['username']
    
    @property
    def database_path(self) -> str:
        """Get database path"""
        return self.config.get('database_path', './tasks.db')
    
    @property
    def partitions(self) -> list:
        """Get partition configurations"""
        return self.config.get('partitions', [])
    
    @property
    def manager_settings(self) -> Dict[str, Any]:
        """Get manager daemon settings"""
        return self.config.get('manager', {})
    
    @property
    def task_executor_settings(self) -> Dict[str, Any]:
        """Get task executor settings"""
        return self.config.get('task_executor', {})
