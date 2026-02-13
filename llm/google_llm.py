"""
Google Gemini provider.
"""
import os
from typing import Optional
from .base import BaseLLM, LLMFactory

try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False


@LLMFactory.register("google")
class GoogleLLM(BaseLLM):
    """Google Gemini provider."""
    
    def __init__(
        self, 
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        if not HAS_GOOGLE:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY env var.")
        
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    async def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3
    ) -> str:
        """Generate completion using Gemini with retry logic."""
        
        # Use config defaults if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        if system:
            config.system_instruction = "You are an expert in predictive maintenance. Answer the following question in detail."
        
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == retries - 1:
                    raise e
