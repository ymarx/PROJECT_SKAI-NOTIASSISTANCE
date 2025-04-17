"""
Base classes for the SKAI-NotiAssistance system.

This module defines the core abstractions and interfaces used throughout the system.
"""

import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseNode(ABC):
    """Base class for all processing nodes in the agent system."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a base node.
        
        Args:
            name: Optional name for this node. If not provided, a UUID will be generated.
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"{self.__class__.__name__}_{self.id[:8]}"
        self.logger = get_logger(f"{__name__}.{self.name}")
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and update the agent state.
        
        Args:
            inputs: Input data for this node
            state: Current agent state
            
        Returns:
            Dictionary of outputs from this node
        """
        pass
    
    def __call__(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the node callable directly.
        
        Args:
            inputs: Input data for this node
            state: Current agent state
            
        Returns:
            Dictionary of outputs from this node
        """
        self.logger.debug(f"Processing inputs in node {self.name}")
        try:
            result = self.process(inputs, state)
            self.logger.debug(f"Node {self.name} processing completed")
            return result
        except Exception as e:
            self.logger.error(f"Error in node {self.name}: {str(e)}", exc_info=True)
            raise


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, config_path: Optional[str] = None):
        """
        Initialize a base agent.
        
        Args:
            name: Name for this agent
            config_path: Optional path to agent configuration file
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
        self.nodes: Dict[str, BaseNode] = {}
        self.state: Dict[str, Any] = {"agent_id": self.id, "agent_name": self.name}
        
        if config_path and os.path.exists(config_path):
            from .utils import load_yaml
            self.config = load_yaml(config_path)
            self.logger.info(f"Loaded configuration from {config_path}")
        else:
            self.config = {}
            if config_path:
                self.logger.warning(f"Configuration file {config_path} not found")
    
    def add_node(self, node: BaseNode, name: Optional[str] = None) -> str:
        """
        Add a processing node to this agent.
        
        Args:
            node: The node to add
            name: Optional name override for the node
            
        Returns:
            The ID of the added node
        """
        node_name = name or node.name
        self.nodes[node_name] = node
        self.logger.debug(f"Added node {node_name} to agent {self.name}")
        return node_name
    
    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent's workflow with the given inputs.
        
        Args:
            inputs: Input data for the agent
            
        Returns:
            Dictionary of agent outputs
        """
        pass
    
    def reset_state(self):
        """Reset the agent state to initial values."""
        self.state = {"agent_id": self.id, "agent_name": self.name}
        self.logger.debug(f"Reset state for agent {self.name}")
        
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the agent callable directly.
        
        Args:
            inputs: Input data for the agent
            
        Returns:
            Dictionary of agent outputs
        """
        return self.run(inputs) 