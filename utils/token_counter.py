"""
Token counting utilities for SKAI-NotiAssistance.

This module provides functions for estimating and counting tokens for LLM interactions.
"""

import re
from typing import Union, Dict, List, Any, Optional

from .logger import get_logger

logger = get_logger(__name__)

# Default tokens per character estimate (approximation)
DEFAULT_TOKENS_PER_CHAR = 0.25

# Cache for tiktoken encoders
_tiktoken_encoders = {}


def estimate_tokens(text: str, tokens_per_char: float = DEFAULT_TOKENS_PER_CHAR) -> int:
    """
    Estimate the number of tokens in a text string using character count heuristic.
    This is a fast but rough approximation when precise counting is not needed.
    
    Args:
        text: Input text to estimate token count for
        tokens_per_char: Estimated ratio of tokens per character
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Count characters and apply the ratio
    return int(len(text) * tokens_per_char)


def count_tokens(
    text: Union[str, List, Dict], 
    model: str = "gpt-4"
) -> int:
    """
    Count the number of tokens in a text string or structured data using tiktoken.
    
    Args:
        text: Input text or data structure to count tokens for
        model: Model name to use for tokenization
        
    Returns:
        Token count
    """
    if not text:
        return 0
    
    try:
        import tiktoken
        
        # Convert structured data to string if needed
        if not isinstance(text, str):
            import json
            text = json.dumps(text)
        
        # Get or create encoder for the model
        if model not in _tiktoken_encoders:
            try:
                _tiktoken_encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to cl100k_base encoding for unknown models
                _tiktoken_encoders[model] = tiktoken.get_encoding("cl100k_base")
                logger.warning(f"Unknown model: {model}, using cl100k_base encoding")
        
        encoder = _tiktoken_encoders[model]
        
        # Count tokens
        tokens = encoder.encode(text)
        return len(tokens)
        
    except ImportError:
        logger.warning("tiktoken package not installed, falling back to character-based estimation")
        return estimate_tokens(text)
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return estimate_tokens(text)


def truncate_text_to_token_limit(
    text: str, 
    max_tokens: int, 
    model: str = "gpt-4",
    truncation_marker: str = "... [text truncated]"
) -> str:
    """
    Truncate text to fit within a token limit.
    
    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        model: Model name to use for tokenization
        truncation_marker: Text to append when truncation occurs
        
    Returns:
        Truncated text
    """
    if not text:
        return text
    
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return text
    
    try:
        import tiktoken
        
        # Get encoder
        if model not in _tiktoken_encoders:
            try:
                _tiktoken_encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                _tiktoken_encoders[model] = tiktoken.get_encoding("cl100k_base")
        
        encoder = _tiktoken_encoders[model]
        
        # Encode and truncate
        tokens = encoder.encode(text)
        marker_tokens = encoder.encode(truncation_marker)
        
        # Calculate available tokens (reserving space for marker)
        available_tokens = max_tokens - len(marker_tokens)
        
        # Truncate and add marker
        truncated_tokens = tokens[:available_tokens]
        truncated_text = encoder.decode(truncated_tokens) + truncation_marker
        
        return truncated_text
        
    except ImportError:
        # Fallback to character-based truncation
        logger.warning("tiktoken package not installed, falling back to character-based truncation")
        
        # Estimate ratio of characters to tokens
        chars_per_token = 1.0 / DEFAULT_TOKENS_PER_CHAR
        
        # Calculate target character count
        marker_token_estimate = estimate_tokens(truncation_marker)
        available_tokens = max_tokens - marker_token_estimate
        target_chars = int(available_tokens * chars_per_token)
        
        # Truncate and add marker
        return text[:target_chars] + truncation_marker 