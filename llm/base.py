"""
Base LLM interface and factory for all providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar, Any
from pydantic import BaseModel
import json

T = TypeVar('T', bound=BaseModel)


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    async def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """Generate a completion for the given prompt."""
        pass
    
    async def complete_structured(
        self, 
        prompt: str, 
        schema: Type[T],
        system: Optional[str] = None,
        temperature: float = 0.3
    ) -> T:
        """Generate a structured response matching the given Pydantic schema."""
        schema_json = schema.model_json_schema()
        
        structured_prompt = f"""{prompt}

Respond with a valid JSON object matching this schema:
```json
{json.dumps(schema_json, indent=2)}
```

Return ONLY the JSON object, no additional text."""

        response = await self.complete(
            structured_prompt, 
            system=system,
            temperature=temperature
        )
        
        # Parse JSON from response
        json_str = self._extract_json(response)
        return schema.model_validate_json(json_str)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from a response that might contain markdown."""
        text = text.strip()
        
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        
        # Try to find raw JSON
        if text.startswith("{") or text.startswith("["):
            return text
        
        # Last resort: find first { to last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        
        raise ValueError(f"Could not extract JSON from response: {text[:200]}...")

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class LLMFactory:
    """Factory for creating LLM instances."""
    
    _providers: dict[str, Type[BaseLLM]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an LLM provider."""
        def decorator(provider_class: Type[BaseLLM]) -> Type[BaseLLM]:
            cls._providers[name] = provider_class
            return provider_class
        return decorator
    
    @classmethod
    def create(cls, provider: str, model: str, **kwargs) -> BaseLLM:
        """Create an LLM instance for the given provider."""
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider: {provider}. Available: {available}")
        
        return cls._providers[provider](model=model, **kwargs)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered providers."""
        return list(cls._providers.keys())


# Import providers to register them
from . import ollama, anthropic_llm, openai_llm, google_llm

__all__ = ['BaseLLM', 'LLMFactory']
