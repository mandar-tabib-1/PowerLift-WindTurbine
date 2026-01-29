"""
Ollama provider for local LLM inference.
Supports: qwen, mistral, llama, and other Ollama-compatible models.
"""
import httpx
from typing import Optional
from .base import BaseLLM, LLMFactory


@LLMFactory.register("ollama")
class OllamaLLM(BaseLLM):
    """Ollama provider for local model inference."""
    
    def __init__(
        self, 
        model: str, 
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    async def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion using Ollama API."""
        
        # Use config defaults if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        timeout_config = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
        
        return result["message"]["content"]
    
    async def list_models(self) -> list[str]:
        """List available models in Ollama."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            result = response.json()
        
        return [model["name"] for model in result.get("models", [])]
    
    async def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
