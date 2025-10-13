import os
import yaml
from typing import Any, Dict

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            # Get project root
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(src_dir)
            config_path = os.path.join(project_root, 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation: 'simulation.n_sim'"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


    @property
    def feature_creation(self) -> Dict[str, Any]:
        return self.config['feature_creation']
    
    @property
    def simulation(self) -> Dict[str, Any]:
        return self.config['simulation']

    @property
    def data(self) -> Dict[str, Any]:
        return self.config['data']
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.config['evaluation']

