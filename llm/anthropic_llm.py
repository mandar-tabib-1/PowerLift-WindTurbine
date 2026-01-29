"""
Anthropic Claude provider.
"""
import os
from typing import Optional
from .base import BaseLLM, LLMFactory

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@LLMFactory.register("anthropic")
class AnthropicLLM(BaseLLM):
    """Anthropic Claude provider."""
    
    def __init__(
        self, 
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY env var.")
        
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    async def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion using Claude."""
        
        # Use config defaults if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system:
            kwargs["system"] = system
        
        if temperature > 0:
            kwargs["temperature"] = temperature
        
        response = await self.client.messages.create(**kwargs)
        
        # Extract text from response
        return response.content[0].text
