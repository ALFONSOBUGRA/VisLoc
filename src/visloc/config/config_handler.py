"""
Configuration handler for the Visual Localization (VisLoc) project.

This module provides functionality for loading and accessing configuration settings.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from visloc.utils.constants import DEFAULT_CONFIG_PATH, DEFAULT_ENV_VAR_NAME


class ConfigHandler:
    """
    Configuration handler for loading and accessing configuration values.
    
    This class provides methods to load configuration from YAML files and
    retrieve values using dot notation for nested dictionaries.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the configuration handler.
        
        Args:
            config_path: Path to the configuration file. If None, try environment variable
                         or default path.
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Configuration dictionary loaded from the file or empty dictionary if file not found.
        """
        if config_path is None:
            config_path = os.environ.get(DEFAULT_ENV_VAR_NAME)
        
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else {}
        else:
            print(f"Config file {config_path} not found. Using default settings.")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the configuration using dot notation.
        
        Args:
            key: Key to retrieve, can use dot notation for nested dictionaries (e.g., "section.key").
            default: Default value to return if key not found.
            
        Returns:
            Value from the configuration or default if not found.
        """
        if not self.config:
            return default
            
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            key: Key to update, can use dot notation for nested dictionaries.
            value: New value to set.
        """
        if '.' in key:
            parts = key.split('.')
            config = self.config
            
            # Navigate to the innermost dictionary
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            
            # Set the value
            config[parts[-1]] = value
        else:
            self.config[key] = value
    
    def save(self, config_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration. If None, use the original path.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"Configuration saved to {config_path}")
    
    def merge(self, other_config: Dict[str, Any]) -> None:
        """
        Merge another configuration dictionary into this one.
        
        Args:
            other_config: Another configuration dictionary to merge.
        """
        def _recursive_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in source.items():
                if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                    destination[key] = _recursive_merge(value, destination[key])
                else:
                    destination[key] = value
            return destination
        
        self.config = _recursive_merge(other_config, self.config) 