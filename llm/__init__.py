"""
LLM module - Unified interface for multiple LLM providers.

Supported providers:
- ollama: Local models (qwen, mistral, llama, etc.)
- anthropic: Claude models
- openai: GPT models
- google: Gemini models
- ntnu: NTNU IDUN local models (OpenAI-compatible)

Usage:
    from src.llm import LLMFactory
    
    # Create an LLM instance
    llm = LLMFactory.create("ollama", model="llama3.3")
    
    # Generate completion
    response = await llm.complete("What is machine learning?")
    
    # With system prompt
    response = await llm.complete(
        "Summarize this paper...",
        system="You are a research assistant."
    )
"""

from .base import BaseLLM, LLMFactory

# Import providers to register them
from . import ollama
from . import anthropic_llm
from . import openai_llm
from . import google_llm
from . import ntnu_llm

__all__ = ['BaseLLM', 'LLMFactory']
