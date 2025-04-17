"""
Prompt engineering module for SKAI-NotiAssistance.

This package contains functions and utilities for prompt creation and management.
"""

from .templates import load_template, format_prompt, get_system_prompt
from .few_shot import create_few_shot_examples, get_few_shot_examples

__all__ = [
    'load_template',
    'format_prompt',
    'get_system_prompt',
    'create_few_shot_examples',
    'get_few_shot_examples'
] 