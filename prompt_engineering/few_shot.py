"""
Few-shot example handling for SKAI-NotiAssistance.

This module provides functions for managing few-shot examples in prompts.
"""

import os
from typing import Any, Dict, List, Optional, Union

from ..lib.utils import load_yaml
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Default template file path
DEFAULT_TEMPLATES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   "config", "prompt_templates.yaml")


def get_few_shot_examples(
    task: str,
    template_file: Optional[str] = None,
    num_examples: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Get few-shot examples for a specific task.
    
    Args:
        task: Task name (e.g., notification_analysis)
        template_file: Optional path to the template file
        num_examples: Optional maximum number of examples to return
        
    Returns:
        List of example dictionaries with 'input' and 'output' keys
    """
    file_path = template_file or DEFAULT_TEMPLATES_PATH
    
    try:
        templates = load_yaml(file_path)
        
        if not templates:
            logger.error(f"No templates found in {file_path}")
            return []
        
        if "few_shot_examples" not in templates:
            logger.error(f"few_shot_examples section not found in {file_path}")
            return []
        
        if task not in templates["few_shot_examples"]:
            logger.warning(f"No few-shot examples found for task '{task}'")
            return []
        
        examples = templates["few_shot_examples"][task]
        
        if num_examples is not None and num_examples > 0:
            examples = examples[:num_examples]
        
        logger.debug(f"Loaded {len(examples)} few-shot examples for task '{task}'")
        return examples
        
    except Exception as e:
        logger.error(f"Error loading few-shot examples for task '{task}': {str(e)}")
        return []


def create_few_shot_examples(
    task: str,
    template_file: Optional[str] = None,
    num_examples: Optional[int] = None,
    format_type: str = "basic"
) -> str:
    """
    Create a formatted string of few-shot examples for inclusion in prompts.
    
    Args:
        task: Task name (e.g., notification_analysis)
        template_file: Optional path to the template file
        num_examples: Optional maximum number of examples to return
        format_type: Format style for the examples ("basic", "markdown", "qa")
        
    Returns:
        Formatted string containing the few-shot examples
    """
    examples = get_few_shot_examples(task, template_file, num_examples)
    
    if not examples:
        return ""
    
    formatted_examples = []
    
    if format_type == "basic":
        for i, example in enumerate(examples):
            formatted_examples.append(f"Example {i+1}:\n")
            formatted_examples.append(f"Input:\n{example['input']}\n")
            formatted_examples.append(f"Output:\n{example['output']}\n")
    
    elif format_type == "markdown":
        for i, example in enumerate(examples):
            formatted_examples.append(f"### Example {i+1}")
            formatted_examples.append(f"**Input:**\n```\n{example['input']}\n```\n")
            formatted_examples.append(f"**Output:**\n```\n{example['output']}\n```\n")
    
    elif format_type == "qa":
        for i, example in enumerate(examples):
            formatted_examples.append(f"User: {example['input']}")
            formatted_examples.append(f"Assistant: {example['output']}\n")
    
    else:
        logger.warning(f"Unknown format_type '{format_type}', using basic format")
        return create_few_shot_examples(task, template_file, num_examples, "basic")
    
    return "\n".join(formatted_examples)


def add_few_shot_to_prompt(
    prompt: str,
    task: str,
    template_file: Optional[str] = None,
    num_examples: Optional[int] = None,
    format_type: str = "basic"
) -> str:
    """
    Add few-shot examples to a prompt.
    
    Args:
        prompt: The original prompt
        task: Task name (e.g., notification_analysis)
        template_file: Optional path to the template file
        num_examples: Optional maximum number of examples to return
        format_type: Format style for the examples
        
    Returns:
        Prompt with few-shot examples added
    """
    few_shot_examples = create_few_shot_examples(
        task=task,
        template_file=template_file,
        num_examples=num_examples,
        format_type=format_type
    )
    
    if not few_shot_examples:
        return prompt
    
    # Add a header for the examples
    examples_header = "\n\nHere are some examples to guide your response:\n\n"
    
    # Add a separator after the examples
    examples_footer = "\n\nNow, analyze the following case:\n\n"
    
    return prompt + examples_header + few_shot_examples + examples_footer 