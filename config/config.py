"""Configuration loader"""
import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml"):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load default config on import 
config = load_config()