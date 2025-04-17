"""
Utility functions for SKAI-NotiAssistance.

This module provides utility functions used throughout the system.
"""

import os
import yaml
import json
import datetime
from typing import Any, Dict, List, Optional, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML file contents
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.debug(f"Loaded YAML from {file_path}")
        return config or {}
    except Exception as e:
        logger.error(f"Error loading YAML from {file_path}: {str(e)}")
        return {}


def save_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to a YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the YAML file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        logger.debug(f"Saved YAML to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving YAML to {file_path}: {str(e)}")
        return False


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON file contents
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        return {}


def save_json(data: Union[Dict[str, Any], List[Any]], file_path: str, pretty: bool = True) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary or list to save
        file_path: Path to save the JSON file
        pretty: Whether to format the JSON with indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            if pretty:
                json.dump(data, file, indent=2, ensure_ascii=False)
            else:
                json.dump(data, file, ensure_ascii=False)
        logger.debug(f"Saved JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary to override base values
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in base_config and 
            isinstance(base_config[key], dict) and 
            isinstance(value, dict)
        ):
            merged_config[key] = merge_configs(base_config[key], value)
        else:
            merged_config[key] = value
            
    return merged_config


def ensure_directory(directory_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {str(e)}")
        return False


def format_timestamp(timestamp: Optional[Union[str, datetime.datetime]] = None) -> str:
    """
    Format a timestamp for logging and display.
    
    Args:
        timestamp: Timestamp to format, current time if None
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            try:
                timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                logger.warning(f"Could not parse timestamp: {timestamp}")
                return timestamp
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S") 