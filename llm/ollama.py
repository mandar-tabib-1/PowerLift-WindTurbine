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
        max_tokens: Optional[int] = None,
        retries: int = 3
    ) -> str:
        """Generate completion using Ollama API with retry logic."""
        
        # Use config defaults if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        messages = []
        if system:
            messages.append({"role": "system", "content": "You are an expert in predictive maintenance. Answer the following question in detail."})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
                response = await self.client.post("/completions", json=payload)
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == retries - 1:
                    raise e
    
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
