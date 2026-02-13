"""
NTNU IDUN LLM provider - OpenAI-compatible API for NTNU's local models.
"""
import os
from typing import Optional
from .base import BaseLLM, LLMFactory

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@LLMFactory.register("ntnu")
class NTNULLM(BaseLLM):
    """NTNU IDUN LLM provider using OpenAI-compatible API."""
    
    def __init__(
        self, 
        model: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
        base_url: str = "https://llm.hpc.ntnu.no/v1",
        timeout: float = 300.0,  # 5 minutes timeout for NTNU's potentially slower models
        temperature: float = 0.7,
        max_tokens: int = 16000,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("NTNU_API_KEY")
        if not self.api_key:
            raise ValueError("NTNU API key not provided. Set NTNU_API_KEY env var or pass api_key.")
        
        self.base_url = base_url
        self.timeout = timeout
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        
        # Create OpenAI client with extended timeout for NTNU
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,  # Extended timeout for slower models
            max_retries=2  # Add retries for network issues
        )
    
    @property
    def provider_name(self) -> str:
        return "ntnu"
    
    async def complete(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3
    ) -> str:
        """Generate completion using NTNU models with retry logic."""
        
        # Use config defaults if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Enhanced system message for predictive maintenance
        messages = []
        if system:
            enhanced_system = "You are an expert in predictive maintenance. Answer the following question in detail. " + system
            messages.append({"role": "system", "content": enhanced_system})
        else:
            messages.append({"role": "system", "content": "You are an expert in predictive maintenance and wind turbine analysis. Provide detailed, technical answers."})
        
        messages.append({"role": "user", "content": prompt})
        
        import time
        for attempt in range(retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                answer = response.choices[0].message.content
                
                # Validate response - check for empty, "none", or too short responses
                if answer and answer.strip() and answer.strip().lower() not in ['none', 'n/a', 'null']:
                    if len(answer.strip()) > 20:  # Ensure substantive answer
                        return answer
                
                # If response is invalid and we have retries left, try again
                if attempt < retries - 1:
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                    
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)  # Longer delay after error before retry
                    continue
                else:
                    raise e
        
        return "Unable to get a response from the LLM. Please try again."