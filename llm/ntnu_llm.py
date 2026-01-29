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
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion using NTNU models."""
        
        # Use config defaults if not provided
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except openai.APITimeoutError as e:
            raise RuntimeError(
                f"NTNU API timeout after {self.timeout}s. This could be due to:\n"
                f"1. Network connectivity issues (Are you on NTNU network or VPN?)\n"
                f"2. NTNU server being overloaded\n"
                f"3. Large model '{self.model}' taking longer than expected\n"
                f"Original error: {str(e)}"
            )
        except openai.APIConnectionError as e:
            raise RuntimeError(
                f"NTNU API connection error. Please check:\n"
                f"1. You are connected to NTNU network or VPN\n"
                f"2. The base URL '{self.base_url}' is accessible\n"
                f"3. Your API key is correct\n"
                f"Original error: {str(e)}"
            )
        except openai.AuthenticationError as e:
            raise RuntimeError(
                f"NTNU API authentication error. Please check:\n"
                f"1. Your API key format - NTNU expects keys starting with 'sk-'\n"
                f"2. For personal API: Contact help@hpc.ntnu.no to get an API key starting with 'sk-'\n"
                f"3. For shared access: Use 'sk-IDUN-NTNU-LLM-API-KEY' (if available)\n"
                f"4. Ensure you have access to the NTNU LLM service\n"
                f"Current API key: {self.api_key[:10]}... (first 10 chars)\n"
                f"Original error: {str(e)}"
            )
        except Exception as e:
            raise RuntimeError(f"NTNU API error: {str(e)}")