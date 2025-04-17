"""
Logging functionality for SKAI-NotiAssistance.

This module provides centralized logging configuration and access.
"""

import os
import yaml
import logging
import logging.config
from typing import Optional, Dict, Any

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "config", "logging_config.yaml")

# Dictionary to store loggers
_loggers = {}


def setup_logging(
    config_path: Optional[str] = None,
    default_level: int = logging.INFO,
    env_key: str = "SKAI_LOGGING_CONFIG"
) -> None:
    """
    Set up logging configuration from a YAML file.
    
    Args:
        config_path: Path to the logging configuration file
        default_level: Default logging level if config file not found
        env_key: Environment variable to check for config file path
    """
    path = config_path or os.getenv(env_key, DEFAULT_CONFIG_PATH)
    
    if os.path.exists(path):
        try:
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
            
            # Create log directory if needed
            for handler in config.get('handlers', {}).values():
                if 'filename' in handler:
                    log_dir = os.path.dirname(handler['filename'])
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)
            
            logging.config.dictConfig(config)
            
            root_logger = logging.getLogger()
            root_logger.info(f"Logging configured using {path}")
            
        except Exception as e:
            print(f"Error in logging configuration: {e}")
            logging.basicConfig(level=default_level)
            logging.error(f"Error in logging configuration: {e}")
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging config file not found at {path}, using basic configuration")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.
    
    This function returns a cached logger if one exists for the given name,
    or creates a new one if needed.
    
    Args:
        name: Name for the logger
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    
    return _loggers[name]


def set_log_level(level: int, logger_name: Optional[str] = None) -> None:
    """
    Set the log level for a specific logger or all loggers.
    
    Args:
        level: Logging level to set
        logger_name: Name of specific logger to set level for, or None for all
    """
    if logger_name:
        logger = get_logger(logger_name)
        logger.setLevel(level)
    else:
        # Set level for all existing loggers
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        for logger in _loggers.values():
            logger.setLevel(level)


# Initialize logging with default configuration
setup_logging() 