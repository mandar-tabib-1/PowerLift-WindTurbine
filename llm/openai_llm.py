"""
OpenAI GPT provider.
"""
import os
from typing import Optional
from .base import BaseLLM, LLMFactory

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@LLMFactory.register("openai")
class OpenAILLM(BaseLLM):
    """OpenAI GPT provider."""
    
    def __init__(
        self, 
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var.")
        
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    async def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion using GPT."""
        
        # Use config defaults if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
