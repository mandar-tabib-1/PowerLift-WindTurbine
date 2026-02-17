"""
Wind Turbine Multi-Agent Orchestrator - GUI Version

A Streamlit-based graphical interface for the wind turbine analysis system.
Run with: streamlit run wind_turbine_gui.py

& ".\.venv\Scripts\python.exe" -m streamlit run wind_turbine_gui.py --server.port 8506
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import time
import os
import sys
import torch
import math
import asyncio
from pathlib import Path
from datetime import datetime
import openai
import requests
import pydeck as pdk

# Try to import geopy for Agent 2D geographic calculations
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    # geopy not available, will use fallback distance calculation

#To do,
#I get the error ⚠️ Agent 2B: Could not connect to LLM. Error: query_local_llm() got an unexpected keyword argument 'temperature' . I have added a new folder called LLM that could be useful with ntnu_llm.py and base.py , and it could be useful for connecting to the specfic Ntnu server where the local llm is. 

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULT_ML_YAW = SCRIPT_DIR.parent
PROJECT_ROOT = RESULT_ML_YAW.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Load environment variables from .env file (for local development / VS Code)
from dotenv import load_dotenv
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# Import the LLM factory
from llm import LLMFactory


def get_api_config(provider: str) -> tuple:
    """
    Get API base URL and key for a given LLM provider.

    Loads secrets from (in priority order):
    1. Streamlit secrets (st.secrets) - for Streamlit Cloud deployment
    2. Environment variables (.env file) - for local VS Code development

    Returns:
        tuple: (api_base, api_key)
    """
    def _get_secret(key: str, fallback: str = "") -> str:
        """Try Streamlit secrets first, then env vars."""
        try:
            return st.secrets[key]
        except (KeyError, FileNotFoundError, AttributeError):
            return os.getenv(key, fallback)

    if provider == "NTNU":
        api_base = _get_secret("NTNU_API_BASE", "https://llm.hpc.ntnu.no/v1")
        api_key = _get_secret("NTNU_API_KEY", "")
    elif provider == "OpenAI":
        api_base = "https://api.openai.com/v1"
        api_key = _get_secret("OPENAI_API_KEY", "")
    elif provider == "Ollama":
        api_base = "http://localhost:11434/v1"
        api_key = "ollama"
    elif provider == "Google":
        api_base = "https://generativelanguage.googleapis.com/v1beta"
        api_key = _get_secret("GOOGLE_API_KEY", "")
    else:  # Anthropic
        api_base = "https://api.anthropic.com/v1"
        api_key = _get_secret("ANTHROPIC_API_KEY", "")

    return api_base, api_key


class GlobalLLMConfig:
    """
    Global LLM configuration manager with persistent local storage.
    Handles configuration priority: GUI session state → local storage → YAML → defaults
    """
    
    PROVIDER_MODELS = {
        "NTNU": [
            "moonshotai/Kimi-K2.5",
            "openai/gpt-oss-120b",
            "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4",
            "zai-org/GLM-4.7-FP8",
            "NorwAI/NorwAI-Magistral-24B-reasoning",
            "Qwen/Qwen3-Coder-Next-FP8",
            "zai-org/GLM-Image"
        ],
        "OpenAI": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini"
        ],
        "Ollama": [
            "llama3.2",
            "mistral",
            "codellama",
            "phi3",
            "qwen2.5"
        ],
        "Google": [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro"
        ],
        "Anthropic": [
            "claude-3.5-sonnet",
            "claude-3-opus",
            "claude-3-haiku"
        ]
    }
    
    DEFAULT_PROVIDER = "NTNU"
    DEFAULT_TEMPERATURE = 0.1
    
    # Maps provider names to their environment variable key names
    PROVIDER_API_KEY_NAMES = {
        "NTNU": "NTNU_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "Ollama": None,  # No key needed
    }

    @classmethod
    def initialize_session_state(cls):
        """Initialize session state with default or persisted values."""
        if 'global_llm_config_initialized' not in st.session_state:
            # Try to load from local storage first
            saved_config = cls._load_from_local_storage()

            # Set default values with persistence
            st.session_state.global_llm_provider = saved_config.get('provider', cls.DEFAULT_PROVIDER)
            st.session_state.global_llm_model = saved_config.get('model', cls.PROVIDER_MODELS[cls.DEFAULT_PROVIDER][0])
            st.session_state.global_llm_temperature = saved_config.get('temperature', cls.DEFAULT_TEMPERATURE)
            st.session_state.global_llm_config_initialized = True

            # Initialize user-provided API keys dict
            if 'user_api_keys' not in st.session_state:
                st.session_state.user_api_keys = {}

            # Backward compatibility - sync with existing session state
            if hasattr(st.session_state, 'llm_provider'):
                st.session_state.global_llm_provider = st.session_state.llm_provider
            if hasattr(st.session_state, 'selected_model'):
                st.session_state.global_llm_model = st.session_state.selected_model

        # Ensure user_api_keys exists even after config is initialized
        if 'user_api_keys' not in st.session_state:
            st.session_state.user_api_keys = {}
    
    @classmethod
    def get_provider(cls) -> str:
        """Get current LLM provider."""
        cls.initialize_session_state()
        return st.session_state.global_llm_provider
    
    @classmethod
    def get_model(cls) -> str:
        """Get current LLM model."""
        cls.initialize_session_state()
        return st.session_state.global_llm_model
    
    @classmethod
    def get_temperature(cls) -> float:
        """Get current temperature setting."""
        cls.initialize_session_state()
        return st.session_state.global_llm_temperature
    
    @classmethod
    def set_provider(cls, provider: str):
        """Set LLM provider and save to local storage."""
        st.session_state.global_llm_provider = provider
        # Reset model to first available for new provider
        if provider in cls.PROVIDER_MODELS:
            st.session_state.global_llm_model = cls.PROVIDER_MODELS[provider][0]
        cls._save_to_local_storage()
    
    @classmethod
    def set_model(cls, model: str):
        """Set LLM model and save to local storage."""
        st.session_state.global_llm_model = model
        cls._save_to_local_storage()
    
    @classmethod
    def set_temperature(cls, temperature: float):
        """Set LLM temperature and save to local storage."""
        st.session_state.global_llm_temperature = temperature
        cls._save_to_local_storage()
    
    @classmethod
    def set_user_api_key(cls, provider: str, api_key: str):
        """Set a user-provided API key for a provider (stored in session only)."""
        if 'user_api_keys' not in st.session_state:
            st.session_state.user_api_keys = {}
        st.session_state.user_api_keys[provider] = api_key

    @classmethod
    def get_user_api_key(cls, provider: str) -> str:
        """Get user-provided API key for a provider, or empty string if none."""
        if 'user_api_keys' not in st.session_state:
            st.session_state.user_api_keys = {}
        return st.session_state.user_api_keys.get(provider, "")

    @classmethod
    def get_api_config(cls) -> tuple:
        """Get API base URL and key for current provider.
        Priority: user-provided key (GUI) → env/.env → Streamlit secrets
        """
        provider = cls.get_provider()
        api_base, api_key = get_api_config(provider)

        # Override with user-provided key from GUI if available
        user_key = cls.get_user_api_key(provider)
        if user_key:
            api_key = user_key

        return api_base, api_key
    
    @classmethod
    def create_llm_instance(cls):
        """Create LLM instance using current global configuration."""
        provider = cls.get_provider().lower()
        model = cls.get_model()
        
        # Get API configuration
        api_base, api_key = cls.get_api_config()
        
        try:
            # Provider name mapping for LLMFactory
            provider_map = {
                "ntnu": "ntnu",
                "openai": "openai", 
                "ollama": "ollama",
                "google": "google",
                "anthropic": "anthropic"
            }
            
            factory_provider = provider_map.get(provider, provider)
            
            return LLMFactory.create(
                factory_provider,
                model=model,
                api_key=api_key,
                base_url=api_base
            )
        except Exception as e:
            st.error(f"Failed to create LLM instance: {e}")
            return None
    
    @classmethod
    def _save_to_local_storage(cls):
        """Save current configuration to browser local storage."""
        try:
            config = {
                'provider': cls.get_provider(),
                'model': cls.get_model(),
                'temperature': cls.get_temperature()
            }
            
            # Use Streamlit's component to save to localStorage
            st.session_state._llm_config_save_needed = config
            
        except Exception as e:
            # Fail silently for local storage issues
            pass
    
    @classmethod
    def _load_from_local_storage(cls) -> dict:
        """Load configuration from browser local storage."""
        try:
            # In a real implementation, this would use a Streamlit component
            # For now, return empty dict to fall back to defaults
            return {}
        except Exception:
            return {}


def test_llm_connection(provider: str = None, model: str = None) -> dict:
    """
    Test LLM connection using global configuration or provided parameters.
    
    Args:
        provider: LLM provider name (uses global config if None)
        model: Model name to test (uses global config if None)
    
    Returns:
        dict with keys: 'success', 'status', 'message', 'error_details'
    """
    # Use global config if not provided
    if provider is None:
        provider = GlobalLLMConfig.get_provider()
    if model is None:
        model = GlobalLLMConfig.get_model()
    import requests
    import json
    
    # Use GlobalLLMConfig to pick up user-provided keys from GUI
    api_base, api_key = GlobalLLMConfig.get_api_config()

    result = {
        'success': False,
        'status': 'unknown',
        'message': '',
        'error_details': ''
    }

    # Check if API key exists
    if not api_key or api_key == "":
        if provider == "Ollama":
            result['status'] = 'no_key_required'
        else:
            result['status'] = 'no_api_key'
            result['message'] = f"❌ No API key found for {provider}"
            result['error_details'] = "Enter API key above, or configure in .env file / Streamlit secrets"
            return result
    
    try:
        if provider == "Ollama":
            # Test Ollama local server
            response = requests.get(f"{api_base.replace('/v1', '')}/api/tags", timeout=5)
            if response.status_code == 200:
                result['success'] = True
                result['status'] = 'connected'
                result['message'] = f"✅ Ollama server running at {api_base}"
            else:
                result['status'] = 'server_error'
                result['message'] = f"❌ Ollama server error (status {response.status_code})"
                result['error_details'] = "Is Ollama running? Try: ollama serve"
                
        elif provider == "NTNU":
            # Test NTNU API with minimal request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result['success'] = True
                result['status'] = 'connected'
                result['message'] = f"✅ Connected to NTNU LLM ({model})"
            elif response.status_code == 401:
                result['status'] = 'invalid_key'
                result['message'] = "❌ Invalid API key"
                result['error_details'] = "Check your NTNU_API_KEY in .env"
            elif response.status_code == 404:
                result['status'] = 'model_not_found'
                result['message'] = f"❌ Model '{model}' not available"
                result['error_details'] = "Try a different model from the dropdown"
            elif response.status_code == 429:
                result['status'] = 'rate_limit'
                result['message'] = "⚠️ Rate limit exceeded"
                result['error_details'] = "Wait a moment and try again"
            else:
                result['status'] = 'api_error'
                result['message'] = f"❌ API error (status {response.status_code})"
                result['error_details'] = response.text[:200]
                
        elif provider == "OpenAI":
            # Test OpenAI API
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result['success'] = True
                result['status'] = 'connected'
                result['message'] = f"✅ Connected to OpenAI ({model})"
            elif response.status_code == 401:
                result['status'] = 'invalid_key'
                result['message'] = "❌ Invalid OpenAI API key"
                result['error_details'] = "Check your OPENAI_API_KEY"
            elif response.status_code == 404:
                result['status'] = 'model_not_found'
                result['message'] = f"❌ Model '{model}' not found"
                result['error_details'] = "You may not have access to this model"
            elif response.status_code == 429:
                result['status'] = 'rate_limit'
                result['message'] = "⚠️ Rate limit or quota exceeded"
                result['error_details'] = "Check your OpenAI billing/limits"
            else:
                result['status'] = 'api_error'
                result['message'] = f"❌ OpenAI API error ({response.status_code})"
                result['error_details'] = response.text[:200]
                
        elif provider == "Google":
            # Test Google Gemini API
            response = requests.post(
                f"{api_base}/models/{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": "test"}]}]},
                timeout=10
            )
            
            if response.status_code == 200:
                result['success'] = True
                result['status'] = 'connected'
                result['message'] = f"✅ Connected to Google Gemini ({model})"
            elif response.status_code == 400:
                error_data = response.json()
                if "API key not valid" in str(error_data):
                    result['status'] = 'invalid_key'
                    result['message'] = "❌ Invalid Google API key"
                    result['error_details'] = "Check your GOOGLE_API_KEY"
                else:
                    result['status'] = 'api_error'
                    result['message'] = f"❌ Google API error"
                    result['error_details'] = str(error_data)[:200]
            else:
                result['status'] = 'api_error'
                result['message'] = f"❌ API error (status {response.status_code})"
                result['error_details'] = response.text[:200]
                
        elif provider == "Anthropic":
            # Test Anthropic Claude API
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            response = requests.post(
                f"{api_base}/messages",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result['success'] = True
                result['status'] = 'connected'
                result['message'] = f"✅ Connected to Anthropic ({model})"
            elif response.status_code == 401:
                result['status'] = 'invalid_key'
                result['message'] = "❌ Invalid Anthropic API key"
                result['error_details'] = "Check your ANTHROPIC_API_KEY"
            elif response.status_code == 429:
                result['status'] = 'rate_limit'
                result['message'] = "⚠️ Rate limit exceeded"
                result['error_details'] = "Wait and try again"
            else:
                result['status'] = 'api_error'
                result['message'] = f"❌ API error (status {response.status_code})"
                result['error_details'] = response.text[:200]
                
    except requests.exceptions.ConnectionError:
        result['status'] = 'connection_error'
        result['message'] = "❌ Cannot connect to API server"
        result['error_details'] = "Check your internet connection or firewall"
    except requests.exceptions.Timeout:
        result['status'] = 'timeout'
        result['message'] = "❌ Connection timeout"
        result['error_details'] = "Server not responding, try again"
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"❌ Unexpected error: {type(e).__name__}"
        result['error_details'] = str(e)[:200]
    
    return result


# Page configuration
st.set_page_config(
    page_title="Wind Turbine Analysis System",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .agent-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #28a745;
    }
    .info-box {
        background-color: #e7f3ff;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Initialize Session State
# =============================================================================
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = 'NTNU'
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'moonshotai/Kimi-K2.5'
if 'turbine_locations' not in st.session_state:
    st.session_state.turbine_locations = {}
if 'last_searched_farm' not in st.session_state:
    st.session_state.last_searched_farm = ''
if 'show_turbine_map' not in st.session_state:
    st.session_state.show_turbine_map = False
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Predictive Maintenance'
if 'pdm_results' not in st.session_state:
    st.session_state.pdm_results = None
if 'pdm_models' not in st.session_state:
    st.session_state.pdm_models = None
if 'whatif_mode_enabled' not in st.session_state:
    st.session_state.whatif_mode_enabled = False
if 'whatif_wind_speed' not in st.session_state:
    st.session_state.whatif_wind_speed = 8.5  # Default above cut-in
if 'whatif_wind_direction' not in st.session_state:
    st.session_state.whatif_wind_direction = 270.0  # Default westerly


# =============================================================================
# Norwegian Wind Farms Database
# =============================================================================
NORWEGIAN_WIND_FARMS = {
    "Bessaker Wind Farm": {
        "location": "Bessakerfjellet, Åfjord, Trøndelag",
        "latitude": 64.1833,
        "longitude": 10.1167,
        "capacity_mw": 57.5,
        "turbines": 25,
        "description": "Located at Bessakerfjellet in Åfjord municipality, featuring 25 Enercon E-70 turbines."
    },
    "Tonstad Wind Farm": {
        "location": "Sirdal, Agder",
        "latitude": 58.6667,
        "longitude": 6.7333,
        "capacity_mw": 208,
        "turbines": 51,
        "description": "One of Norway's largest onshore wind farms, located in Southern Norway."
    },
    "Smøla Wind Farm": {
        "location": "Smøla, Møre og Romsdal",
        "latitude": 63.4000,
        "longitude": 8.0167,
        "capacity_mw": 150,
        "turbines": 68,
        "description": "One of Norway's oldest and largest wind farms, operational since 2002."
    },
    "Roan Wind Farm": {
        "location": "Roan, Trøndelag",
        "latitude": 64.1500,
        "longitude": 10.3000,
        "capacity_mw": 255.6,
        "turbines": 71,
        "description": "Part of the Fosen Vind project, one of Europe's largest onshore wind farms."
    },
    "Raggovidda Wind Farm": {
        "location": "Berlevåg, Finnmark",
        "latitude": 70.5333,
        "longitude": 29.1000,
        "capacity_mw": 45,
        "turbines": 15,
        "description": "Norway's northernmost wind farm, located above the Arctic Circle."
    },
}


# =============================================================================
# Agent Imports (lazy loading)
# =============================================================================
@st.cache_resource
def load_agents():
    """Load all agents once and cache them."""
    from rotor_power_agent import RotorPowerAgent
    from tt_opinf_inference_agent import WakeFlowAgent
    
    # Initialize agents
    model_path = os.path.join(SCRIPT_DIR, "rotor_power_gp_model.joblib")
    power_agent = RotorPowerAgent(model_path)
    wake_agent = WakeFlowAgent()
    
    return power_agent, wake_agent


# Import reviewer agent (not cached - initialized per analysis with user settings)
try:
    from reviewer_agent import WindTurbineReviewerAgent
    REVIEWER_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ReviewerAgent not available: {e}")
    REVIEWER_AGENT_AVAILABLE = False


# =============================================================================
# Predictive Maintenance (PdM) Functions
# =============================================================================

def load_pdm_models_subprocess():
    """
    Load PdM models using separate Python environment with compatible scikit-learn.
    
    This function creates a subprocess with a dedicated virtual environment (.venv_pdm)
    that has scikit-learn 1.3.0 installed, avoiding version conflicts with the main
    GUI environment which uses scikit-learn 1.8.0.
    
    Returns:
        dict or None: Models and test data, or None if loading fails
    """
    import subprocess
    import tempfile
    import pickle
    
    try:
        # Path to PdM-specific Python interpreter
        pdm_python = os.path.join(SCRIPT_DIR, '.venv_pdm', 'Scripts', 'python.exe')
        
        # Check if required files exist
        rul_dir = os.path.join(SCRIPT_DIR, 'RUL')
        save_models_path = os.path.join(rul_dir, 'save_models.py')
        wind_turbine_pm_path = os.path.join(rul_dir, 'wind_turbine_pm_fuhrlander.py')
        
        if not os.path.exists(pdm_python):
            st.error("❌ PdM environment not found at `.venv_pdm`")
            
            with st.expander("🔧 Setup Instructions - Create PdM Environment"):
                st.markdown("""
                **To create the PdM environment with compatible scikit-learn:**
                
                Open PowerShell in your project directory and run:
                
                ```powershell
                # Create new virtual environment
                python -m venv .venv_pdm
                
                # Activate it
                & "\.venv_pdm\Scripts\Activate.ps1"
                
                # Install compatible scikit-learn + dependencies
                pip install scikit-learn==1.3.0 numpy pandas joblib matplotlib
                
                # Verify installation
                python -c "import sklearn; print(f'sklearn version: {sklearn.__version__}')"
                
                # Deactivate when done
                deactivate
                ```
                
                **What this does:**
                - Creates `.venv_pdm/` with scikit-learn 1.3.0 (compatible with saved models)
                - Your main `.venv/` keeps scikit-learn 1.8.0 (for other tasks)
                - No conflicts between versions!
                
                **After setup, click "Load PdM Models" again.**
                """)
            
            return None
        
        # Check if required Python modules exist
        if not os.path.exists(save_models_path):
            st.error(f"❌ Required file not found: `RUL/save_models.py`")
            st.warning("Please ensure the RUL directory contains all necessary files.")
            return None
            
        if not os.path.exists(wind_turbine_pm_path):
            st.error(f"❌ Required file not found: `RUL/wind_turbine_pm_fuhrlander.py`")
            st.warning("Please ensure the RUL directory contains all necessary files.")
            return None
        
        # Create Python script to run in separate process
        loader_script = """
import sys
import os

# Add both parent directory and RUL directory to path
parent_path = r'""" + str(SCRIPT_DIR).replace('\\', '\\\\') + """'
rul_path = os.path.join(parent_path, 'RUL')

# Ensure paths are absolute and exist
if os.path.exists(rul_path):
    sys.path.insert(0, rul_path)
if os.path.exists(parent_path):
    sys.path.insert(0, parent_path)

# Change to RUL directory to ensure relative imports work
os.chdir(rul_path)

# Import and load models
from save_models import load_all_models, load_test_data
import pickle
import tempfile

# Load models using compatible sklearn
models = load_all_models()
test_data = load_test_data()

# Serialize to temporary file
with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
    pickle.dump({
        'models': models, 
        'test_data': test_data,
        'sklearn_version': __import__('sklearn').__version__
    }, f)
    print(f.name)  # Output temp file path to stdout
"""
        
        # Run loader script in separate process
        result = subprocess.run(
            [pdm_python, '-c', loader_script],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode != 0:
            st.error("❌ Model loading subprocess failed")
            st.code(result.stderr, language="text")
            
            with st.expander("🔍 Troubleshooting"):
                st.markdown("""
                **Common Issues:**
                
                1. **ImportError: No module named 'save_models' or 'wind_turbine_pm_fuhrlander'**
                   - Check that `RUL/save_models.py` exists
                   - Check that `RUL/wind_turbine_pm_fuhrlander.py` exists
                   - Verify `.venv_pdm` has all dependencies installed
                   - Ensure the RUL directory structure is intact
                
                2. **FileNotFoundError: Model files not found**
                   - Ensure `RUL/saved_models/` directory exists
                   - Run training script if models haven't been generated:
                     ```powershell
                     & "\.venv_pdm\Scripts\python.exe" RUL/wind_turbine_pm_fuhrlander.py
                     ```
                
                3. **Pickle/joblib errors**
                   - Models may have been saved with incompatible sklearn
                   - Retrain models with `.venv_pdm` environment
                   
                4. **Path/import errors**
                   - Verify the RUL directory is in the correct location
                   - Check file permissions on the RUL directory
                """)
            
            return None
        
        # Get temp file path from stdout (last line)
        stdout_lines = result.stdout.strip().splitlines()
        temp_file_path = stdout_lines[-1].strip() if stdout_lines else ""
        
        if not temp_file_path or not os.path.exists(temp_file_path):
            st.error("❌ Subprocess succeeded but temp file not found")
            st.code(f"Stdout: {result.stdout}\nStderr: {result.stderr}", language="text")
            return None
        
        # Load serialized data in main process
        with open(temp_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Cleanup temp file
        try:
            os.unlink(temp_file_path)
        except:
            pass  # Ignore cleanup errors
        
        # Verify loaded data structure
        if not isinstance(data, dict) or 'models' not in data or 'test_data' not in data:
            st.error("❌ Loaded data has unexpected structure")
            return None
        
        # Check test data validity
        test_data = data['test_data']
        if not isinstance(test_data, dict) or 'X_test' not in test_data:
            st.error("❌ Test data missing required keys (X_test, y_test)")
            return None
        
        n_samples = len(test_data['X_test'])
        n_features = test_data['X_test'].shape[1] if len(test_data['X_test'].shape) > 1 else 0
        sklearn_version = data.get('sklearn_version', 'unknown')
        
        st.success(f'✅ Models loaded via subprocess with sklearn {sklearn_version}')
        st.info(f'📊 Loaded {n_samples:,} samples with {n_features} features')
        
        return {
            'models': data['models'],
            'test_data': test_data
        }
        
    except subprocess.TimeoutExpired:
        st.error("❌ Model loading timed out (>60 seconds)")
        st.warning("""**Possible causes:**
        - Model files are very large
        - Slow disk I/O
        - Python environment has issues
        
        **Try:**
        - Restart Streamlit
        - Check system resources
        - Verify `.venv_pdm` is not corrupted
        """)
        return None
        
    except Exception as e:
        st.error(f"❌ Unexpected error in subprocess model loading: {str(e)}")
        import traceback
        with st.expander("🔍 Full Error Trace"):
            st.code(traceback.format_exc())
        return None


@st.cache_resource
def load_pdm_models():
    """
    Load Fuhrlander FL2500 predictive maintenance models.
    
    Loads models directly using the main environment's scikit-learn 1.8.0.
    Models have been retrained with sklearn 1.8.0 for compatibility.
    
    Returns:
        dict: Dictionary with 'models' and 'test_data'
    """
    try:
        import sys
        
        # Add RUL directory to path
        rul_path = os.path.join(SCRIPT_DIR, 'RUL')
        if rul_path not in sys.path:
            sys.path.insert(0, rul_path)
        
        with st.spinner('📦 Loading PdM models (sklearn 1.8.0)...'):
            from save_models import load_all_models, load_test_data
            
            models = load_all_models()
            test_data = load_test_data()
        
        # Display version info
        sklearn_version = models.get('sklearn_version', 'unknown')
        st.success(f'✅ Models loaded successfully (sklearn {sklearn_version})')
        
        return {
            'models': models,
            'test_data': test_data
        }
        
    except ImportError as e:
        st.error(f"❌ Import error: {e}")
        st.info("💡 Make sure RUL folder contains save_models.py")
        return None
        
    except Exception as e:
        st.error(f'❌ Failed to load PdM models: {str(e)}')
        
        # Check if it's a version mismatch error
        error_str = str(e).lower()
        if 'pickle' in error_str or 'joblib' in error_str or 'sklearn' in error_str or 'pyx_unpickle' in error_str:
            with st.expander("💡 Solution: Retrain Models"):
                st.markdown("""
                **This error is due to scikit-learn version mismatch.**
                
                The saved models were created with an older sklearn version (1.4.2), 
                but your current environment has sklearn 1.8.0.
                
                **Solution: Retrain models with current sklearn version**
                ```powershell
                # Navigate to project directory
                cd PowerLift
                
                # Activate main environment
                & "\.venv\Scripts\activate"
                
                # Retrain models (~10-15 minutes)
                cd RUL
                python wind_turbine_pm_fuhrlander.py
                ```
                
                This will regenerate all models compatible with sklearn 1.8.0.
                The retrained models will have identical performance.
                """)
        
        with st.expander("🔍 Click for detailed error trace"):
            import traceback
            st.code(traceback.format_exc())
        
        return None


def run_pdm_inference(models_dict, n_samples=1000, sample_offset=0):
    """Run predictive maintenance inference on test data."""
    import numpy as np
    import sys
    rul_path = os.path.join(SCRIPT_DIR, 'RUL')
    if rul_path not in sys.path:
        sys.path.insert(0, rul_path)
    
    from wind_turbine_pm_fuhrlander import predict_rul
    
    models = models_dict['models']
    test_data = models_dict['test_data']
    
    # Extract subset of test data
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    test_df = test_data['test_df']
    
    # Apply offset and limit to n_samples
    end_idx = min(sample_offset + n_samples, len(X_test))
    X_subset = X_test[sample_offset:end_idx]
    y_subset = y_test[sample_offset:end_idx]
    df_subset = test_df.iloc[sample_offset:end_idx]
    
    # Run inference
    autoencoder = models['autoencoder']
    gmm = models['gmm']
    binary_clf = models['binary_clf']
    multi_clf = models['multi_clf']
    lstm_models = models['lstm_models']
    failure_threshold = models['failure_threshold']
    state_order_map = models['state_order_map']
    
    # 1. Health Indicator
    hi = autoencoder.get_health_indicator(X_subset)
    
    # 2. Health States (apply ordering map to raw GMM predictions)
    raw_states = gmm.predict(hi.reshape(-1, 1))
    states = np.array([state_order_map[s] for s in raw_states])
    
    # Verify state ordering (diagnostic)
    if len(hi) > 0:
        state_hi_means = {}
        for state in [0, 1, 2]:
            mask = states == state
            if np.any(mask):
                state_hi_means[state] = np.mean(hi[mask])
        print(f"[PDM Inference] Mean HI per state: {state_hi_means}")
        print(f"[PDM Inference] State order map used: {state_order_map}")
    
    # 3. Binary fault prediction
    binary_pred = binary_clf.predict(X_subset)
    binary_prob = binary_clf.predict_proba(X_subset)
    
    # 4. Multi-class fault prediction
    multi_pred = multi_clf.predict(X_subset)
    multi_prob = multi_clf.predict_proba(X_subset)
    
    # 5. RUL prediction
    rul = predict_rul(hi, states, lstm_models, failure_threshold, seq_length=24)
    
    return {
        'X': X_subset,
        'y_true': y_subset,
        'df': df_subset,
        'health_indicator': hi,
        'health_states': states,
        'binary_predictions': binary_pred,
        'binary_probabilities': binary_prob,
        'multi_predictions': multi_pred,
        'multi_probabilities': multi_prob,
        'rul_predictions': rul,
        'failure_threshold': failure_threshold,
        'feature_names': models['feature_names'],
        'n_samples': len(X_subset),
        'sample_offset': sample_offset,
    }


def display_pdm_results(pdm_results, models_dict):
    """Display PdM inference results in the GUI."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sys
    
    rul_path = os.path.join(SCRIPT_DIR, 'RUL')
    if rul_path not in sys.path:
        sys.path.insert(0, rul_path)
    
    st.markdown('## 🏥 Predictive Maintenance Analysis Results')
    st.markdown('---')
    
    # Summary metrics
    st.markdown('### 📊 Overall Health Summary')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_hi = np.nanmean(pdm_results['health_indicator'])
        st.metric('Average Health Indicator', f'{avg_hi:.4f}')
        health_status = 'Healthy' if avg_hi < pdm_results['failure_threshold'] else '⚠️ At Risk'
        st.caption(f'Status: {health_status}')
    
    with col2:
        state_counts = pd.Series(pdm_results['health_states']).value_counts()
        state_labels = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
        dominant_state = state_counts.idxmax()
        st.metric('Dominant Health State', state_labels[dominant_state])
        st.caption(f'Count: {state_counts[dominant_state]}/{len(pdm_results["health_states"])}')
    
    with col3:
        avg_fault_prob = np.nanmean(pdm_results['binary_probabilities'][:, 1])
        st.metric('Avg Fault Probability', f'{avg_fault_prob:.2%}')
        st.caption('Binary classifier')
    
    with col4:
        valid_rul = pdm_results['rul_predictions'][~np.isnan(pdm_results['rul_predictions'])]
        if len(valid_rul) > 0:
            min_rul = np.min(valid_rul)
            st.metric('Min RUL (hours)', f'{min_rul:.1f}')
            st.caption('Time to failure')
        else:
            st.metric('Min RUL', 'N/A')
            st.caption('No valid predictions')
    
    st.markdown('---')
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        '📈 Health Indicator Trend',
        '🎯 State Distribution',
        '⚠️ Fault Probabilities',
        '⏱️ RUL Predictions',
        '🔍 Feature Importance'
    ])
    
    with tab1:
        st.markdown('#### Health Indicator Over Time')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color by health state
        state_colors = {0: '#2ca02c', 1: '#ff7f0e', 2: '#d62728'}
        state_labels_map = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
        
        for state in [0, 1, 2]:
            mask = pdm_results['health_states'] == state
            if np.any(mask):
                indices = np.where(mask)[0]
                ax.scatter(indices, pdm_results['health_indicator'][mask],
                          s=20, alpha=0.6, color=state_colors[state],
                          label=state_labels_map[state])
        
        ax.axhline(y=pdm_results['failure_threshold'], color='red',
                  linestyle='--', linewidth=2,
                  label=f"Failure Threshold ({pdm_results['failure_threshold']:.4f})")
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Health Indicator', fontsize=12)
        ax.set_title('Health Indicator Trend with GMM States', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Statistics
        st.markdown('**Statistics:**')
        st.write(f'- Mean HI: {np.nanmean(pdm_results["health_indicator"]):.4f}')
        st.write(f'- Std HI: {np.nanstd(pdm_results["health_indicator"]):.4f}')
        st.write(f'- Max HI: {np.nanmax(pdm_results["health_indicator"]):.4f}')
        st.write(f'- Samples above threshold: {np.sum(pdm_results["health_indicator"] > pdm_results["failure_threshold"])}/{len(pdm_results["health_indicator"])}')
    
    with tab2:
        st.markdown('#### Health State Distribution')
        
        # Add state ordering verification
        st.markdown("**🔍 GMM State Ordering Verification:**")
        state_labels_map = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
        verification_data = []
        for state in [0, 1, 2]:
            mask = pdm_results['health_states'] == state
            if np.any(mask):
                hi_in_state = pdm_results['health_indicator'][mask]
                verification_data.append({
                    'State': f"{state} ({state_labels_map[state]})",
                    'Samples': int(np.sum(mask)),
                    'HI Mean': f"{np.mean(hi_in_state):.4f}",
                    'HI Min': f"{np.min(hi_in_state):.4f}",
                    'HI Max': f"{np.max(hi_in_state):.4f}"
                })
        
        if verification_data:
            df_verify = pd.DataFrame(verification_data)
            st.dataframe(df_verify, use_container_width=True, hide_index=True)
            
            # Check ordering
            means = [float(row['HI Mean']) for row in verification_data]
            if all(means[i] < means[i+1] for i in range(len(means)-1)):
                st.success("✅ States correctly ordered: Healthy (low HI) → Degrading (medium HI) → Critical (high HI)")
            else:
                st.error("❌ States are NOT correctly ordered! Healthy should have lowest HI, Critical should have highest.")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            state_labels_map = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
            state_counts = pd.Series(pdm_results['health_states']).value_counts().sort_index()
            colors = ['#2ca02c', '#ff7f0e', '#d62728']
            labels = [state_labels_map[i] for i in state_counts.index]
            
            wedges, texts, autotexts = ax.pie(state_counts.values, labels=labels,
                                               autopct='%1.1f%%', colors=colors,
                                               startangle=90, textprops={'size': 12})
            ax.set_title('GMM Health State Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(labels, state_counts.values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Health State', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Health State Counts', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(state_counts.values):
                ax.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            plt.close(fig)
    
    with tab3:
        st.markdown('#### Fault Probability Analysis')
        
        # Binary classifier probabilities
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Binary fault probability over time
        ax1.plot(pdm_results['binary_probabilities'][:, 1],
                linewidth=1, alpha=0.7, color='orange')
        ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
                   label='Decision Threshold (0.5)')
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('P(Anomalous)', fontsize=12)
        ax1.set_title('Binary Fault Probability Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Multi-class probabilities
        class_labels = ['Healthy', 'Pre-Fault', 'Fault']
        colors_multi = ['#2ca02c', '#ff7f0e', '#d62728']
        
        for i, (label, color) in enumerate(zip(class_labels, colors_multi)):
            ax2.plot(pdm_results['multi_probabilities'][:, i],
                    linewidth=1, alpha=0.7, label=label, color=color)
        
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Multi-Class Fault Probabilities', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Statistics
        st.markdown('**Binary Classifier Statistics:**')
        n_anomalous = np.sum(pdm_results['binary_predictions'] == 1)
        st.write(f'- Samples classified as anomalous: {n_anomalous}/{len(pdm_results["binary_predictions"])} ({n_anomalous/len(pdm_results["binary_predictions"])*100:.1f}%)')
        st.write(f'- Average P(anomalous): {np.mean(pdm_results["binary_probabilities"][:, 1]):.3f}')
        
        st.markdown('**Multi-Class Classifier Statistics:**')
        for i, label in enumerate(class_labels):
            count = np.sum(pdm_results['multi_predictions'] == i)
            st.write(f'- {label}: {count}/{len(pdm_results["multi_predictions"])} ({count/len(pdm_results["multi_predictions"])*100:.1f}%)')
    
    with tab4:
        st.markdown('#### Remaining Useful Life (RUL) Predictions')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valid_mask = ~np.isnan(pdm_results['rul_predictions'])
        valid_indices = np.where(valid_mask)[0]
        valid_rul = pdm_results['rul_predictions'][valid_mask]
        
        if len(valid_rul) > 0:
            # Color by urgency
            colors_rul = np.where(valid_rul < 24, 'red',
                                 np.where(valid_rul < 168, 'orange', 'green'))
            
            ax.scatter(valid_indices, valid_rul, c=colors_rul, s=20, alpha=0.6)
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('RUL (hours)', fontsize=12)
            ax.set_title('Remaining Useful Life Predictions', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='> 1 week (safe)'),
                Patch(facecolor='orange', label='1-7 days (warning)'),
                Patch(facecolor='red', label='< 1 day (critical)')
            ]
            ax.legend(handles=legend_elements, loc='best')
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Statistics
            st.markdown('**RUL Statistics:**')
            st.write(f'- Valid predictions: {len(valid_rul)}/{len(pdm_results["rul_predictions"])}')
            st.write(f'- Mean RUL: {np.mean(valid_rul):.1f} hours ({np.mean(valid_rul)/24:.1f} days)')
            st.write(f'- Min RUL: {np.min(valid_rul):.1f} hours')
            st.write(f'- Max RUL: {np.max(valid_rul):.1f} hours')
            
            critical = np.sum(valid_rul < 24)
            warning = np.sum((valid_rul >= 24) & (valid_rul < 168))
            safe = np.sum(valid_rul >= 168)
            st.write(f'- Critical (<24h): {critical}')
            st.write(f'- Warning (1-7 days): {warning}')
            st.write(f'- Safe (>7 days): {safe}')
        else:
            st.warning('No valid RUL predictions available (requires sufficient sequence length)')
    
    with tab5:
        st.markdown('#### Feature Importance (SHAP Analysis)')
        
        try:
            from shap_explainer import SHAPExplainer
            
            # Get a sample for SHAP explanation
            sample_idx = min(100, len(pdm_results['X']) - 1)
            X_sample = pdm_results['X'][sample_idx:sample_idx+1]
            
            with st.spinner('Computing SHAP values...'):
                explainer = SHAPExplainer(
                    binary_clf=models_dict['models']['binary_clf'],
                    multi_clf=models_dict['models']['multi_clf'],
                    feature_names=pdm_results['feature_names']
                )
                
                explanation = explainer.explain_single_sample(
                    X_sample,
                    sample_id=f'sample_{sample_idx}',
                    save_plot=False,
                    verbose=False
                )
            
            st.markdown(f'**Sample {sample_idx} Analysis:**')
            st.write(explanation['explanation_text'])
            
            # Display top features
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('**Top Features Pushing Towards Fault:**')
                for feat, shap_val, feat_val in explanation['top_features_pushing_fault'][:10]:
                    st.write(f'- {feat}: SHAP={shap_val:.4f}, Value={feat_val:.4f}')
            
            with col2:
                st.markdown('**Top Features Pushing Towards Healthy:**')
                for feat, shap_val, feat_val in explanation['top_features_pushing_healthy'][:10]:
                    st.write(f'- {feat}: SHAP={shap_val:.4f}, Value={feat_val:.4f}')
            
        except Exception as e:
            st.warning(f'SHAP analysis not available: {str(e)}')
            st.info('Install SHAP library for detailed feature importance analysis: `pip install shap`')
    
    st.markdown('---')
    st.success('✅ Predictive Maintenance analysis complete!')
    
    # Add fault diagnosis section using SHAP
    st.markdown('---')
    st.markdown('### 🔍 Fault Diagnosis & Root Cause Analysis')
    st.markdown('Analyzing critical and degrading turbines using SHAP values...')
    
    with st.spinner('Computing SHAP-based diagnosis...'):
        try:
            # Get feature data from pdm_results (returned by run_pdm_inference)
            X_data = pdm_results['X']
            diagnosis = diagnose_faults_shap(pdm_results, models_dict, X_data, top_n=5)
            display_fault_diagnosis(diagnosis)
        except Exception as e:
            st.error(f'❌ Fault diagnosis failed: {str(e)}')
            st.info('💡 SHAP analysis requires the shap package. Install with: pip install shap')
            import traceback
            with st.expander('🔍 Error Details'):
                st.code(traceback.format_exc())
    
    # Add downloadable report section
    st.markdown('---')
    st.markdown('### 📄 Download Analysis Report')
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('Generate a comprehensive PDF report with analysis results and recommendations.')
    
    with col2:
        if st.button('📥 Generate Report', use_container_width=True):
            report_content = generate_pdm_report(pdm_results, models_dict)
            st.download_button(
                label='Download Report',
                data=report_content,
                file_name=f'pdm_analysis_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.md',
                mime='text/markdown',
                use_container_width=True
            )
    
    # Add chatbot section
    st.markdown('---')
    pdm_chatbot_section(pdm_results, models_dict)


# =============================================================================
# Fault Diagnosis Mapping (Feature -> Root Cause -> Maintenance Action)
# =============================================================================

FAULT_DIAGNOSIS_MAP = {
    # Gearbox temperatures
    'wtrm_avg_TrmTmp_Gbx': {
        'component': 'Gearbox',
        'high_threshold': 80.0,
        'root_cause': 'High gearbox temperature - lubrication degradation or overload',
        'maintenance_action': '1. Check oil level and quality\n2. Replace oil if contaminated\n3. Check for gear tooth wear\n4. Verify cooling system operation',
        'urgency': 'HIGH',
        'cost_estimate': '€5,000-€15,000'
    },
    'wtrm_avg_TrmTmp_GbxOil': {
        'component': 'Gearbox Oil System',
        'high_threshold': 75.0,
        'root_cause': 'High oil temperature - cooling system failure or oil degradation',
        'maintenance_action': '1. Inspect oil cooler\n2. Check oil filter condition\n3. Verify oil circulation pump\n4. Replace oil if oxidized',
        'urgency': 'HIGH',
        'cost_estimate': '€3,000-€8,000'
    },
    'wtrm_avg_TrmTmp_GbxBrg151': {
        'component': 'Gearbox Bearing 151',
        'high_threshold': 70.0,
        'root_cause': 'High bearing temperature - bearing wear or lubrication failure',
        'maintenance_action': '1. Perform vibration analysis\n2. Inspect bearing condition (endoscopy)\n3. Check lubrication delivery\n4. Plan bearing replacement',
        'urgency': 'CRITICAL',
        'cost_estimate': '€10,000-€30,000'
    },
    'wtrm_avg_TrmTmp_GbxBrg152': {
        'component': 'Gearbox Bearing 152',
        'high_threshold': 70.0,
        'root_cause': 'High bearing temperature - bearing wear or lubrication failure',
        'maintenance_action': '1. Perform vibration analysis\n2. Inspect bearing condition (endoscopy)\n3. Check lubrication delivery\n4. Plan bearing replacement',
        'urgency': 'CRITICAL',
        'cost_estimate': '€10,000-€30,000'
    },
    'wtrm_avg_TrmTmp_GbxBrg450': {
        'component': 'Gearbox Bearing 450',
        'high_threshold': 70.0,
        'root_cause': 'High bearing temperature - bearing wear or lubrication failure',
        'maintenance_action': '1. Perform vibration analysis\n2. Inspect bearing condition (endoscopy)\n3. Check lubrication delivery\n4. Plan bearing replacement',
        'urgency': 'CRITICAL',
        'cost_estimate': '€10,000-€30,000'
    },
    'wtrm_avg_TrmTmp_GnBrgDE': {
        'component': 'Generator Bearing (Drive End)',
        'high_threshold': 75.0,
        'root_cause': 'High generator bearing temperature - bearing degradation',
        'maintenance_action': '1. Vibration analysis\n2. Oil analysis for metal particles\n3. Check alignment with gearbox\n4. Replace bearing if needed',
        'urgency': 'HIGH',
        'cost_estimate': '€8,000-€20,000'
    },
    'wtrm_avg_TrmTmp_GnBrgNDE': {
        'component': 'Generator Bearing (Non-Drive End)',
        'high_threshold': 75.0,
        'root_cause': 'High generator bearing temperature - bearing degradation',
        'maintenance_action': '1. Vibration analysis\n2. Oil analysis for metal particles\n3. Check alignment\n4. Replace bearing if needed',
        'urgency': 'HIGH',
        'cost_estimate': '€8,000-€20,000'
    },
    'wtrm_avg_Gbx_OilPres': {
        'component': 'Gearbox Oil Pressure',
        'low_threshold': 1.5,
        'root_cause': 'Low oil pressure - pump failure, filter blockage, or oil leak',
        'maintenance_action': '1. Inspect oil pump operation\n2. Check oil filters (replace if clogged)\n3. Check for oil leaks\n4. Verify pressure sensor calibration',
        'urgency': 'CRITICAL',
        'cost_estimate': '€2,000-€10,000'
    },
    'wtrm_avg_Brg_OilPres': {
        'component': 'Main Bearing Oil Pressure',
        'low_threshold': 1.2,
        'root_cause': 'Low bearing oil pressure - pump malfunction or leak',
        'maintenance_action': '1. Check bearing lubrication pump\n2. Inspect oil lines for leaks\n3. Replace filters\n4. Verify oil level',
        'urgency': 'CRITICAL',
        'cost_estimate': '€2,000-€8,000'
    },
    'wgen_avg_GnTmp_phsA': {
        'component': 'Generator Windings',
        'high_threshold': 120.0,
        'root_cause': 'High generator temperature - electrical overload or cooling failure',
        'maintenance_action': '1. Check generator cooling system\n2. Inspect stator windings for damage\n3. Verify electrical load balance\n4. Check for insulation degradation',
        'urgency': 'HIGH',
        'cost_estimate': '€15,000-€50,000'
    },
    'wgen_sdv_Spd': {
        'component': 'Generator Speed Variability',
        'high_threshold': 10.0,
        'root_cause': 'High speed variability - mechanical imbalance or blade issues',
        'maintenance_action': '1. Inspect rotor balance\n2. Check blade pitch actuators\n3. Verify coupling condition\n4. Inspect gearbox teeth for wear',
        'urgency': 'MEDIUM',
        'cost_estimate': '€5,000-€20,000'
    },
    'thermal_stress_idx': {
        'component': 'Overall Thermal Stress',
        'high_threshold': 0.75,
        'root_cause': 'High thermal stress - system overheating',
        'maintenance_action': '1. Comprehensive thermal inspection\n2. Check all cooling systems\n3. Reduce load temporarily\n4. Plan major maintenance',
        'urgency': 'HIGH',
        'cost_estimate': '€10,000-€40,000'
    },
    'bearing_stress_idx': {
        'component': 'Overall Bearing Stress',
        'high_threshold': 0.70,
        'root_cause': 'High bearing stress - multiple bearing degradation',
        'maintenance_action': '1. Prioritize bearing inspections\n2. Comprehensive vibration survey\n3. Oil analysis for all bearings\n4. Plan coordinated bearing replacement',
        'urgency': 'CRITICAL',
        'cost_estimate': '€25,000-€80,000'
    },
    'power_efficiency': {
        'component': 'Power Efficiency',
        'low_threshold': 0.25,
        'root_cause': 'Low power efficiency - mechanical losses or grid issues',
        'maintenance_action': '1. Check gearbox efficiency\n2. Inspect blade pitch system\n3. Verify generator performance\n4. Check grid connection quality',
        'urgency': 'MEDIUM',
        'cost_estimate': '€3,000-€15,000'
    },
    'bearing_temp_spread': {
        'component': 'Bearing Temperature Balance',
        'high_threshold': 15.0,
        'root_cause': 'Large bearing temp spread - uneven load distribution or alignment issue',
        'maintenance_action': '1. Check shaft alignment\n2. Inspect bearing mounting\n3. Verify load distribution\n4. Check for structural issues',
        'urgency': 'MEDIUM',
        'cost_estimate': '€5,000-€25,000'
    },
}


def diagnose_faults_shap(pdm_results, models_dict, X_data, top_n=5):
    """
    Fault diagnosis agent using SHAP values for critical and degrading turbines.
    Returns root cause analysis and maintenance recommendations without requiring LLM.
    
    Args:
        pdm_results: Dictionary with analysis results including health_states
        models_dict: Dictionary with trained models (binary_clf, multi_clf)
        X_data: Feature matrix (N_samples, 27 features)
        top_n: Number of top contributing features to analyze
    
    Returns:
        dict with diagnosis for critical and degrading turbines
    """
    import numpy as np
    import pandas as pd
    import sys
    import os
    
    # Add RUL directory to path if not already there
    rul_dir = os.path.join(os.path.dirname(__file__), 'RUL')
    if rul_dir not in sys.path:
        sys.path.insert(0, rul_dir)
    
    try:
        from shap_explainer import SHAPExplainer
    except ImportError:
        return {
            'error': 'SHAP explainer not available',
            'critical_diagnosis': [],
            'degrading_diagnosis': []
        }
    
    # Feature names from Fuhrlander FL2500
    RAW_FEATURES = [
        'wtrm_avg_TrmTmp_Gbx', 'wtrm_avg_TrmTmp_GbxOil',
        'wtrm_avg_TrmTmp_GbxBrg151', 'wtrm_avg_TrmTmp_GbxBrg152',
        'wtrm_avg_TrmTmp_GbxBrg450', 'wtrm_avg_TrmTmp_GnBrgDE',
        'wtrm_avg_TrmTmp_GnBrgNDE', 'wtrm_avg_Gbx_OilPres',
        'wtrm_avg_Brg_OilPres', 'wgen_avg_GnTmp_phsA',
        'wgen_avg_Spd', 'wnac_avg_WSpd1', 'wnac_avg_NacTmp',
        'wgdc_avg_TriGri_PwrAt', 'wgdc_avg_TriGri_A',
        'wtrm_sdv_TrmTmp_Gbx', 'wtrm_sdv_TrmTmp_GbxOil', 'wgen_sdv_Spd'
    ]
    ENGINEERED_FEATURES = [
        'thermal_stress_idx', 'bearing_stress_idx', 'power_efficiency',
        'gbx_temp_trend', 'oil_pressure_ratio', 'bearing_temp_spread',
        'gen_thermal_load', 'oil_temp_trend', 'variability_trend'
    ]
    ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
    
    # Identify critical and degrading samples
    health_states = pdm_results['health_states']
    critical_mask = health_states == 2  # Critical
    degrading_mask = health_states == 1  # Degrading
    
    critical_indices = np.where(critical_mask)[0]
    degrading_indices = np.where(degrading_mask)[0]
    
    diagnosis = {
        'critical_diagnosis': [],
        'degrading_diagnosis': [],
        'critical_count': len(critical_indices),
        'degrading_count': len(degrading_indices)
    }
    
    if len(critical_indices) == 0 and len(degrading_indices) == 0:
        return diagnosis
    
    # Extract models from models_dict structure
    # models_dict has structure: {'models': {...}, 'test_data': {...}}
    models = models_dict.get('models', models_dict)  # Fallback if already unwrapped
    
    # Initialize SHAP explainer
    try:
        explainer = SHAPExplainer(
            binary_clf=models['binary_clf'],
            multi_clf=models['multi_clf'],
            feature_names=ALL_FEATURES,
            output_dir=rul_dir
        )
    except Exception as e:
        diagnosis['error'] = f"Failed to initialize SHAP explainer: {str(e)}"
        return diagnosis
    
    # Analyze CRITICAL samples (take up to 100 random samples for speed)
    if len(critical_indices) > 0:
        sample_size = min(100, len(critical_indices))
        sampled_critical = np.random.choice(critical_indices, sample_size, replace=False)
        
        X_critical = X_data[sampled_critical]
        
        # Compute SHAP values for critical samples
        try:
            shap_result = explainer.compute_shap_values(X_critical, max_samples=sample_size)
            binary_shap = shap_result['binary_shap']
            
            if isinstance(binary_shap, list):
                binary_shap = binary_shap[1]  # Class 1 (anomalous)
            
            # Average SHAP across all critical samples
            mean_shap = np.mean(np.abs(binary_shap), axis=0)
            top_features_idx = np.argsort(mean_shap)[::-1][:top_n]
            
            for rank, feat_idx in enumerate(top_features_idx, 1):
                feat_name = ALL_FEATURES[feat_idx]
                shap_importance = mean_shap[feat_idx]
                avg_feature_value = np.mean(X_critical[:, feat_idx])
                
                # Get diagnosis if feature is known
                if feat_name in FAULT_DIAGNOSIS_MAP:
                    diag = FAULT_DIAGNOSIS_MAP[feat_name].copy()
                    diag['rank'] = rank
                    diag['feature_name'] = feat_name
                    diag['shap_importance'] = float(shap_importance)
                    diag['avg_value'] = float(avg_feature_value)
                    
                    # Check if value exceeds threshold
                    if 'high_threshold' in diag and avg_feature_value > diag['high_threshold']:
                        diag['threshold_exceeded'] = True
                        diag['threshold_type'] = 'HIGH'
                    elif 'low_threshold' in diag and avg_feature_value < diag['low_threshold']:
                        diag['threshold_exceeded'] = True
                        diag['threshold_type'] = 'LOW'
                    else:
                        diag['threshold_exceeded'] = False
                    
                    diagnosis['critical_diagnosis'].append(diag)
        except Exception as e:
            diagnosis['critical_error'] = f"SHAP analysis failed for critical samples: {str(e)}"
    
    # Analyze DEGRADING samples
    if len(degrading_indices) > 0:
        sample_size = min(100, len(degrading_indices))
        sampled_degrading = np.random.choice(degrading_indices, sample_size, replace=False)
        
        X_degrading = X_data[sampled_degrading]
        
        try:
            shap_result = explainer.compute_shap_values(X_degrading, max_samples=sample_size)
            binary_shap = shap_result['binary_shap']
            
            if isinstance(binary_shap, list):
                binary_shap = binary_shap[1]
            
            mean_shap = np.mean(np.abs(binary_shap), axis=0)
            top_features_idx = np.argsort(mean_shap)[::-1][:top_n]
            
            for rank, feat_idx in enumerate(top_features_idx, 1):
                feat_name = ALL_FEATURES[feat_idx]
                shap_importance = mean_shap[feat_idx]
                avg_feature_value = np.mean(X_degrading[:, feat_idx])
                
                if feat_name in FAULT_DIAGNOSIS_MAP:
                    diag = FAULT_DIAGNOSIS_MAP[feat_name].copy()
                    diag['rank'] = rank
                    diag['feature_name'] = feat_name
                    diag['shap_importance'] = float(shap_importance)
                    diag['avg_value'] = float(avg_feature_value)
                    
                    if 'high_threshold' in diag and avg_feature_value > diag['high_threshold']:
                        diag['threshold_exceeded'] = True
                        diag['threshold_type'] = 'HIGH'
                    elif 'low_threshold' in diag and avg_feature_value < diag['low_threshold']:
                        diag['threshold_exceeded'] = True
                        diag['threshold_type'] = 'LOW'
                    else:
                        diag['threshold_exceeded'] = False
                    
                    diagnosis['degrading_diagnosis'].append(diag)
        except Exception as e:
            diagnosis['degrading_error'] = f"SHAP analysis failed for degrading samples: {str(e)}"
    
    return diagnosis


def generate_pdm_report(pdm_results, models_dict):
    """
    Generate a downloadable Markdown report with PdM analysis results and recommendations.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    # Extract key metrics
    avg_hi = np.nanmean(pdm_results['health_indicator'])
    failure_threshold = pdm_results['failure_threshold']
    state_counts = pd.Series(pdm_results['health_states']).value_counts()
    state_labels = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
    avg_fault_prob = np.nanmean(pdm_results['binary_probabilities'][:, 1])
    valid_rul = pdm_results['rul_predictions'][~np.isnan(pdm_results['rul_predictions'])]
    
    report = f"""# Predictive Maintenance Analysis Report
**Fuhrlander FL2500 Wind Turbine**  
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report presents the predictive maintenance analysis for the Fuhrlander FL2500 wind turbine based on {pdm_results['n_samples']} SCADA data samples.

### Overall Health Assessment

- **Average Health Indicator:** {avg_hi:.4f}
- **Failure Threshold:** {failure_threshold:.4f}
- **Health Status:** {'⚠️ AT RISK - Immediate attention required' if avg_hi >= failure_threshold else '✅ HEALTHY - Normal operation'}
- **Average Fault Probability:** {avg_fault_prob:.2%}

### Health State Distribution

"""
    
    for state_id in [0, 1, 2]:
        count = state_counts.get(state_id, 0)
        percentage = (count / pdm_results['n_samples']) * 100
        report += f"- **{state_labels[state_id]}:** {count} samples ({percentage:.1f}%)\n"
    
    report += "\n---\n\n## Detailed Analysis\n\n### 1. Health Indicator Analysis\n\n"
    
    report += f"""The Health Indicator (HI) is computed using an autoencoder neural network that measures reconstruction error:

- **Mean HI:** {np.nanmean(pdm_results['health_indicator']):.4f}
- **Std Deviation:** {np.nanstd(pdm_results['health_indicator']):.4f}
- **Maximum HI:** {np.nanmax(pdm_results['health_indicator']):.4f}
- **Minimum HI:** {np.nanmin(pdm_results['health_indicator']):.4f}
- **Samples Above Threshold:** {np.sum(pdm_results['health_indicator'] > failure_threshold)} / {len(pdm_results['health_indicator'])} ({np.sum(pdm_results['health_indicator'] > failure_threshold)/len(pdm_results['health_indicator'])*100:.1f}%)

**Interpretation:** Values below {failure_threshold:.4f} indicate normal operation. Higher values suggest anomalous behavior requiring investigation.

### 2. Fault Probability Assessment\n
#### Binary Classification (Healthy vs Anomalous):
"""
    
    n_anomalous = np.sum(pdm_results['binary_predictions'] == 1)
    report += f"""- **Anomalous Samples:** {n_anomalous} / {len(pdm_results['binary_predictions'])} ({n_anomalous/len(pdm_results['binary_predictions'])*100:.1f}%)
- **Average P(Anomalous):** {np.mean(pdm_results['binary_probabilities'][:, 1]):.3f}

#### Multi-Class Classification:

"""
    
    class_labels = ['Healthy', 'Pre-Fault', 'Fault']
    for i, label in enumerate(class_labels):
        count = np.sum(pdm_results['multi_predictions'] == i)
        percentage = (count / len(pdm_results['multi_predictions'])) * 100
        report += f"- **{label}:** {count} samples ({percentage:.1f}%)\n"
    
    report += "\n### 3. Remaining Useful Life (RUL) Predictions\n\n"
    
    if len(valid_rul) > 0:
        min_rul = np.min(valid_rul)
        mean_rul = np.mean(valid_rul)
        critical = np.sum(valid_rul < 24)
        warning = np.sum((valid_rul >= 24) & (valid_rul < 168))
        safe = np.sum(valid_rul >= 168)
        
        report += f"""- **Valid Predictions:** {len(valid_rul)} / {len(pdm_results['rul_predictions'])}
- **Mean RUL:** {mean_rul:.1f} hours ({mean_rul/24:.1f} days)
- **Minimum RUL:** {min_rul:.1f} hours
- **Maximum RUL:** {np.max(valid_rul):.1f} hours

#### RUL Distribution by Urgency:

- **Critical (<24 hours):** {critical} samples
- **Warning (1-7 days):** {warning} samples
- **Safe (>7 days):** {safe} samples

"""
    else:
        report += "No valid RUL predictions available (requires sufficient sequence length).\n\n"
    
    report += """---

## Recommendations

### Immediate Actions

"""
    
    # Generate recommendations based on analysis
    recommendations = []
    
    if avg_hi >= failure_threshold:
        recommendations.append("🔴 **URGENT:** Average Health Indicator exceeds failure threshold. Schedule immediate inspection of critical components (gearbox, bearings, generator).")
    
    if avg_fault_prob > 0.5:
        recommendations.append("🔴 **URGENT:** High fault probability detected. Perform comprehensive diagnostic check within 24 hours.")
    
    if len(valid_rul) > 0 and np.min(valid_rul) < 24:
        recommendations.append(f"🔴 **CRITICAL:** Minimum RUL is {np.min(valid_rul):.1f} hours. Component failure imminent - plan immediate maintenance.")
    
    critical_state_pct = (state_counts.get(2, 0) / pdm_results['n_samples']) * 100
    if critical_state_pct > 10:
        recommendations.append(f"🟠 **HIGH PRIORITY:** {critical_state_pct:.1f}% of samples in Critical health state. Investigate root cause.")
    
    if len(valid_rul) > 0:
        warning_samples = np.sum((valid_rul >= 24) & (valid_rul < 168))
        if warning_samples > 0:
            recommendations.append(f"🟡 **MEDIUM PRIORITY:** {warning_samples} samples show RUL between 1-7 days. Plan preventive maintenance within this window.")
    
    if avg_hi < failure_threshold and avg_fault_prob < 0.3:
        recommendations.append("✅ **NORMAL OPERATION:** Turbine appears healthy. Continue routine monitoring schedule.")
    else:
        recommendations.append("🟡 **MONITORING:** Increase monitoring frequency to detect early signs of degradation.")
    
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n\n"
    
    report += """### Preventive Measures

1. **Lubrication System:** Verify gearbox and bearing oil levels and quality
2. **Temperature Monitoring:** Check for abnormal temperature trends in transmission components
3. **Vibration Analysis:** Conduct vibration measurements on critical bearings
4. **Oil Analysis:** Perform oil contamination and wear particle analysis
5. **Generator Inspection:** Check generator windings and cooling system

### Long-term Strategy

1. **Predictive Maintenance Schedule:** Implement data-driven maintenance intervals based on actual health indicators
2. **Component Tracking:** Maintain detailed logs of component health metrics over time
3. **Failure Pattern Analysis:** Correlate failures with operating conditions to improve predictions
4. **Spare Parts Planning:** Ensure availability of critical components based on RUL predictions
5. **Training:** Ensure maintenance staff understand health indicator interpretation

---

## Technical Details

### ML Models Used

1. **Autoencoder:** Custom neural network (27→64→32→8→32→64→27) for Health Indicator extraction
2. **Gaussian Mixture Model (GMM):** 3-component clustering for health state classification
3. **Gradient Boosting Classifier:** Binary fault prediction (200 trees, depth 5)
4. **Random Forest Classifier:** Multi-class fault prediction (300 trees, depth 10)
5. **LSTM Networks:** State-specific RUL prediction (sequence length: 24 hours)

### Input Features (27 total)

**Transmission System (9 features):**
- Gearbox temperature, oil temperature, bearing temperatures (3 bearings)
- Gearbox oil pressure, main bearing oil pressure
- Temperature and oil variability indicators

**Generator System (3 features):**
- Generator winding temperature, generator speed, speed variability

**Environmental (2 features):**
- Wind speed, nacelle temperature

**Power System (2 features):**
- Active power output, grid current

**Engineered Features (9 features):**
- Thermal stress index, bearing stress index, power efficiency
- Temperature trends, oil pressure ratio, bearing temperature spread
- Generator thermal load, oil temperature trend, variability trend

### Model Performance

- **Accuracy:** 75.2%
- **F1 Score:** 0.746 (weighted average)
- **Training Data:** 53,810 samples from turbines 80, 81, 82
- **Test Data:** 35,411 samples from turbines 83, 84

---

## Conclusion

"""
    
    if avg_hi >= failure_threshold or avg_fault_prob > 0.5:
        report += f"""The analysis indicates **elevated risk** requiring immediate attention. With an average health indicator of {avg_hi:.4f} (threshold: {failure_threshold:.4f}) and fault probability of {avg_fault_prob:.2%}, proactive maintenance is strongly recommended to prevent unplanned downtime.

Estimated cost of unplanned failure: €500,000 - €2,000,000  
Estimated cost of planned maintenance: €50,000 - €150,000

**Recommendation:** Schedule maintenance within the next 24-72 hours to mitigate failure risk.
"""
    else:
        report += f"""The analysis indicates **normal operation** with an average health indicator of {avg_hi:.4f} (below threshold: {failure_threshold:.4f}) and low fault probability of {avg_fault_prob:.2%}. Continue routine monitoring and preventive maintenance schedule.

**Recommendation:** Maintain current monitoring frequency and schedule next inspection per standard maintenance plan.
"""
    
    report += """\n---

## Appendix: Methodology

### Data Collection
- **Frequency:** Hourly aggregated from 5-minute SCADA data
- **Features:** 18 raw SCADA features + 9 engineered features
- **Quality:** Missing values imputed, outliers handled via robust scaling

### ML Pipeline
1. **Feature Engineering:** Rolling statistics, thermal indices, efficiency metrics
2. **Health Indicator:** Autoencoder reconstruction error normalized to [0, 1] range
3. **State Classification:** GMM with 3 components for Healthy/Degrading/Critical states
4. **Fault Prediction:** Ensemble methods (Gradient Boosting + Random Forest)
5. **RUL Forecasting:** State-specific LSTM models with 24-hour lookback

### Model Training
- **Framework:** scikit-learn 1.8.0 + custom numpy implementations
- **Validation:** 5-fold cross-validation on training set
- **Hyperparameters:** Optimized via grid search
- **Retraining Schedule:** Quarterly or when new failure data becomes available

---

*This report was generated automatically by the Wind Turbine Predictive Maintenance System developed by mandar.tabib, SINTEF Digital*  
*For questions or concerns, contact: mandar.tabib@sintef.no*

"""
    
    return report


def display_fault_diagnosis(diagnosis):
    """
    Display fault diagnosis results from SHAP analysis.
    """
    import pandas as pd
    
    if 'error' in diagnosis:
        st.error(f"❌ {diagnosis['error']}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Critical Samples', diagnosis['critical_count'], 
                  delta='High Priority' if diagnosis['critical_count'] > 0 else None,
                  delta_color='inverse')
    
    with col2:
        st.metric('Degrading Samples', diagnosis['degrading_count'],
                  delta='Monitor Closely' if diagnosis['degrading_count'] > 0 else None,
                  delta_color='normal')
    
    # Display Critical turbines diagnosis
    if diagnosis['critical_diagnosis']:
        st.markdown('#### 🔴 Critical Turbines - Root Cause Analysis')
        st.markdown(f"Analysis based on {diagnosis['critical_count']} critical samples (GMM State = 2)")
        
        for diag in diagnosis['critical_diagnosis']:
            urgency_icon = '🔴' if diag['urgency'] == 'CRITICAL' else '🟠'
            threshold_msg = ''
            if diag.get('threshold_exceeded', False):
                threshold_msg = f" ⚠️ **THRESHOLD EXCEEDED** ({diag['threshold_type']})"
            
            with st.expander(
                f"{urgency_icon} Rank {diag['rank']}: {diag['component']} - {diag['urgency']} Urgency{' ⚠️' if diag.get('threshold_exceeded') else ''}",
                expanded=(diag['rank'] <= 2)
            ):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown(f"**Feature:** `{diag['feature_name']}`")
                    st.markdown(f"**SHAP Importance:** {diag['shap_importance']:.4f}")
                    st.markdown(f"**Average Value:** {diag['avg_value']:.2f}{threshold_msg}")
                
                with col_b:
                    st.markdown(f"**Urgency:** {diag['urgency']}")
                    st.markdown(f"**Est. Cost:** {diag['cost_estimate']}")
                
                st.markdown(f"**Root Cause:**")
                st.info(diag['root_cause'])
                
                st.markdown(f"**Recommended Maintenance Actions:**")
                st.success(diag['maintenance_action'])
        
        # Summary of critical actions
        st.markdown('---')
        st.markdown('##### 🔧 Immediate Action Summary for Critical Turbines')
        urgent_items = [d for d in diagnosis['critical_diagnosis'] if d['urgency'] in ['CRITICAL', 'HIGH']]
        if urgent_items:
            for i, item in enumerate(urgent_items[:3], 1):
                st.markdown(f"{i}. **{item['component']}**: {item['root_cause'].split(' - ')[1] if ' - ' in item['root_cause'] else item['root_cause']}")
    elif diagnosis['critical_count'] > 0:
        st.info('No specific fault diagnosis available for critical samples. Continue monitoring.')
    
    # Display Degrading turbines diagnosis
    if diagnosis['degrading_diagnosis']:
        st.markdown('#### 🟡 Degrading Turbines - Early Warning Analysis')
        st.markdown(f"Analysis based on {diagnosis['degrading_count']} degrading samples (GMM State = 1)")
        
        for diag in diagnosis['degrading_diagnosis']:
            urgency_icon = '🟠' if diag['urgency'] == 'HIGH' else '🟡'
            threshold_msg = ''
            if diag.get('threshold_exceeded', False):
                threshold_msg = f" ⚠️ **THRESHOLD EXCEEDED** ({diag['threshold_type']})"
            
            with st.expander(
                f"{urgency_icon} Rank {diag['rank']}: {diag['component']} - {diag['urgency']} Urgency{' ⚠️' if diag.get('threshold_exceeded') else ''}",
                expanded=(diag['rank'] == 1)
            ):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown(f"**Feature:** `{diag['feature_name']}`")
                    st.markdown(f"**SHAP Importance:** {diag['shap_importance']:.4f}")
                    st.markdown(f"**Average Value:** {diag['avg_value']:.2f}{threshold_msg}")
                
                with col_b:
                    st.markdown(f"**Urgency:** {diag['urgency']}")
                    st.markdown(f"**Est. Cost:** {diag['cost_estimate']}")
                
                st.markdown(f"**Root Cause:**")
                st.info(diag['root_cause'])
                
                st.markdown(f"**Preventive Actions:**")
                st.success(diag['maintenance_action'])
        
        # Summary of preventive actions
        st.markdown('---')
        st.markdown('##### 🛠️ Preventive Maintenance Plan for Degrading Turbines')
        preventive_items = [d for d in diagnosis['degrading_diagnosis'] if d['urgency'] in ['HIGH', 'MEDIUM']]
        if preventive_items:
            for i, item in enumerate(preventive_items[:3], 1):
                st.markdown(f"{i}. **{item['component']}**: Schedule inspection in next maintenance window")
    elif diagnosis['degrading_count'] > 0:
        st.info('No specific fault diagnosis available for degrading samples. Maintain regular monitoring schedule.')
    
    # No faults found
    if not diagnosis['critical_diagnosis'] and not diagnosis['degrading_diagnosis']:
        if diagnosis['critical_count'] == 0 and diagnosis['degrading_count'] == 0:
            st.success('✅ All turbines are in healthy state. Continue routine monitoring.')


def load_research_reports_context():
    """
    Load research report summaries from docs/research/*.tex files and build
    a supplementary reference context string with citations for the LLM chatbots.
    Returns a string that can be appended to existing context/system messages.
    """
    import os
    import re

    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'research')

    # Mapping of tex files to their citation references
    report_refs = {
        'technical_report_multiagent.tex': {
            'citation': '[1] M. Tabib. "A Multi-Agent AI Framework Integrating Large Language Models and Machine Learning for Wind Turbine Predictive Maintenance and Wake Steering Optimisation". Technical Report, SINTEF Digital, 2026.',
            'key': 'Tabib2026',
            'topics': 'multi-agent framework, LLM integration, predictive maintenance, wake steering optimisation'
        },
        'gpr_power_prediction.tex': {
            'citation': '[2] M. Tabib. "Gaussian Process Regression for Wind Turbine Rotor Power Prediction under Yaw Misalignment". Technical Report, SINTEF Digital, 2025a.',
            'key': 'Tabib2025a',
            'topics': 'Gaussian process regression, power prediction, yaw misalignment, rotor power'
        },
        'RESEARCH_PAPER_PM.tex': {
            'citation': '[3] M. Tabib. "A Semi-Supervised Machine Learning Framework for Predictive Maintenance of Wind Turbine Gearbox Systems Using Real SCADA Data". Technical Report, SINTEF Digital, 2025b.',
            'key': 'Tabib2025b',
            'topics': 'semi-supervised learning, predictive maintenance, SCADA data, gearbox, autoencoder, GMM, SHAP, RUL'
        },
        'deepwind2026_tt_opinf.tex': {
            'citation': '[4] M. Tabib. "Tensor Train Decomposition with Operator Inference for Parametric Wind Turbine Wake Flow Prediction". Technical Report, SINTEF Digital, 2025c.',
            'key': 'Tabib2025c',
            'topics': 'tensor train decomposition, operator inference, wake flow prediction, reduced-order model'
        }
    }

    references_text = "\n\nSUPPLEMENTARY RESEARCH REFERENCES:\n"
    references_text += "The following SINTEF technical reports provide additional background. "
    references_text += "When your answer draws on methods, results, or concepts from these reports, "
    references_text += "cite them using (Tabib, 2026), (Tabib, 2025a), (Tabib, 2025b), or (Tabib, 2025c). "
    references_text += "At the end of your answer, list the full citations under a 'References' heading.\n\n"

    for tex_file, ref_info in report_refs.items():
        filepath = os.path.join(reports_dir, tex_file)
        references_text += f"{ref_info['citation']}\n"

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract abstract
                abstract_match = re.search(
                    r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
                    content, re.DOTALL
                )
                if abstract_match:
                    abstract = abstract_match.group(1).strip()
                    # Clean LaTeX commands
                    abstract = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', abstract)
                    abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)
                    abstract = re.sub(r'[{}~]', ' ', abstract)
                    abstract = re.sub(r'\s+', ' ', abstract).strip()
                    if len(abstract) > 600:
                        abstract = abstract[:600] + '...'
                    references_text += f"  Summary: {abstract}\n"

                # Extract section titles
                sections = re.findall(r'\\section\{([^}]+)\}', content)
                if sections:
                    references_text += f"  Sections: {', '.join(sections[:10])}\n"
            except Exception:
                pass
        references_text += "\n"

    return references_text


def pdm_chatbot_section(pdm_results, models_dict):
    """
    Interactive chatbot for PdM analysis queries using LLM.
    """
    import numpy as np
    
    st.markdown('### 💬 Analysis Q&A Chatbot')
    st.markdown('Ask questions about the analysis methods, results, or get recommendations.')
    
    # LLM selection in sidebar
    col1, col2 = st.columns([4, 1])
    
    with col2:
        with st.expander('⚙️ LLM Settings'):
            llm_provider = st.selectbox(
                'Provider',
                ['NTNU', 'OpenAI', 'Ollama', 'Google', 'Anthropic'],
                key='pdm_llm_provider',
                help='Select LLM provider for chatbot'
            )
            
            if llm_provider == "NTNU":
                models = [
                    'moonshotai/Kimi-K2.5',
                    'Qwen/QwQ-32B-Preview',
                    'deepseek-ai/DeepSeek-R1',
                    'meta-llama/Llama-3.3-70B-Instruct-Turbo'
                ]
            elif llm_provider == "OpenAI":
                models = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo']
            elif llm_provider == "Ollama":
                models = ['llama2', 'mistral', 'codellama']
            elif llm_provider == "Google":
                models = ['gemini-pro', 'gemini-ultra']
            else:
                models = ['claude-3-opus', 'claude-3-sonnet', 'claude-2']
            
            selected_model = st.selectbox(
                'Model',
                models,
                key='pdm_selected_model',
                help=f'Select {llm_provider} model'
            )
    
    # Suggested questions
    with col1:
        st.markdown('**Suggested Questions:**')
        
        suggested_questions = [
            'What do the health indicator values mean?',
            'Why is my turbine classified as "Degrading"?',
            'How accurate is the RUL prediction?',
            'What maintenance actions should I take first?',
            'Explain the difference between binary and multi-class predictions',
            'What causes high health indicator values?',
            'How often should I retrain the ML models?',
            'What are the most important features for fault prediction?'
        ]
        
        # Display as buttons
        cols = st.columns(4)
        for i, question in enumerate(suggested_questions[:8]):
            with cols[i % 4]:
                if st.button(question[:40] + '...', key=f'pdm_q_{i}', use_container_width=True):
                    # Set the question and trigger processing flag
                    st.session_state.pdm_selected_question = question
                    st.session_state.pdm_trigger_ask = True
    
    # Chat input
    if 'pdm_chat_history' not in st.session_state:
        st.session_state.pdm_chat_history = []
    
    # Handle suggested question trigger
    if st.session_state.get('pdm_trigger_ask', False):
        user_question = st.session_state.get('pdm_selected_question', '')
        st.session_state.pdm_trigger_ask = False
        process_question = True
    else:
        user_question = st.text_input(
            'Your question:',
            value=st.session_state.get('pdm_chat_input', ''),
            placeholder='Ask anything about the analysis...',
            key='pdm_chat_input_field'
        )
        process_question = st.button('🚀 Ask', key='pdm_ask_btn')
    
    if process_question and user_question:
        with st.spinner(f'Consulting {llm_provider} {selected_model}...'):
            # Prepare context for LLM
            avg_hi = np.nanmean(pdm_results['health_indicator'])
            failure_threshold = pdm_results['failure_threshold']
            avg_fault_prob = np.nanmean(pdm_results['binary_probabilities'][:, 1])
            valid_rul = pdm_results['rul_predictions'][~np.isnan(pdm_results['rul_predictions'])]
            
            state_counts = {}
            for state in [0, 1, 2]:
                state_counts[state] = np.sum(pdm_results['health_states'] == state)
            
            min_rul_str = f"{np.min(valid_rul):.1f} hours" if len(valid_rul) > 0 else 'N/A'
            
            context = f"""Predictive Maintenance Analysis Context:
- Average Health Indicator: {avg_hi:.4f} (threshold: {failure_threshold:.4f})
- Average Fault Probability: {avg_fault_prob:.2%}
- Health State Distribution: Healthy={state_counts[0]}, Degrading={state_counts[1]}, Critical={state_counts[2]}
- Valid RUL predictions: {len(valid_rul) if len(valid_rul) > 0 else 0}
- Min RUL: {min_rul_str}
- Turbine Type: Fuhrlander FL2500 (2.5MW)
- Analysis Samples: {pdm_results['n_samples']}

ML Models:
1. Autoencoder (27→8→27) for Health Indicator extraction
2. Gaussian Mixture Model (3 components) for health state classification
3. Gradient Boosting Classifier (200 trees) for binary fault prediction
4. Random Forest Classifier (300 trees) for multi-class prediction
5. LSTM networks for RUL prediction (24-hour sequences)
"""
            
            system_message = f"""You are an expert wind turbine predictive maintenance specialist with deep knowledge of:
- Machine learning models for anomaly detection and fault prediction
- Wind turbine SCADA data analysis
- Fuhrlander FL2500 turbine specifications and failure modes
- Maintenance strategies and cost-benefit analysis

Provide clear, actionable answers based on the analysis context. Be specific about:
- Technical explanations of ML model outputs
- Interpretation of health indicators and probabilities
- Maintenance recommendations with rationale
- Industry best practices

Use technical terminology appropriately but explain complex concepts clearly."""

            # Load supplementary research report references
            reports_context = load_research_reports_context()

            prompt = f"{context}\n{reports_context}\n\nUser Question: {user_question}\n\nProvide a detailed, technical answer:"
            
            try:
                # Use unified LLM function with global configuration
                response = query_unified_llm(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,  # Lower temperature for consistent responses
                    max_tokens=1500,  # Reduced for faster responses
                    timeout=60.0  # Increased timeout for complete responses
                )
                
                # Add to chat history
                st.session_state.pdm_chat_history.append({
                    'question': user_question,
                    'answer': response,
                    'provider': llm_provider,
                    'model': selected_model
                })
                
                st.session_state.pdm_chat_input = ''
                st.session_state.pdm_selected_question = ''
                st.rerun()  # Rerun to show the new response
                
            except Exception as e:
                st.error(f"Error querying LLM: {str(e)}")
                st.info("Check your API configuration and network connection.")
                st.session_state.pdm_selected_question = ''
    
    # Display chat history
    if st.session_state.pdm_chat_history:
        st.markdown('---')
        st.markdown('**Chat History:**')
        
        for i, chat in enumerate(reversed(st.session_state.pdm_chat_history[-5:])):
            with st.expander(f"Q: {chat['question'][:80]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer** ({chat['provider']} - {chat['model']}):")
                st.markdown(chat['answer'])
        
        if st.button('🗑️ Clear Chat History'):
            st.session_state.pdm_chat_history = []
            st.rerun()


# =============================================================================
# Weather Agent Functions
# =============================================================================
def rotate_line(line, angle, ax):
    """
    Rotates a Line2D object by a specified angle in degrees around its center.

    Parameters:
    - line: The Line2D object to rotate.
    - angle: The rotation angle in degrees (float).
    - ax: The Axes object where the line is plotted.
    """
    # Get the line's data points (x, y)
    xdata, ydata = line.get_data()
    
    # Create a rotation transformation around the origin
    transform = transforms.Affine2D().rotate_deg(angle) + ax.transData
    
    # Set the new transform for the line
    line.set_transform(transform)
    
def fetch_weather(city: str, country: str):
    """Fetch weather data for a location."""
    import requests
    
    location = f"{city}, {country}"
    
    # Geocoding
    try:
        geocoding_api = "https://geocoding-api.open-meteo.com/v1/search"
        response = requests.get(
            geocoding_api,
            params={"name": f"{city} {country}", "count": 1},
            timeout=10
        )
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            lat, lon = result["latitude"], result["longitude"]
        else:
            raise Exception("Location not found")
    except Exception as e:
        st.warning(f"Could not find exact location. Using default coordinates.")
        lat, lon = 40.0150, -105.2705  # Default to Boulder, Colorado
    
    # Weather data
    try:
        api_base = "https://api.open-meteo.com/v1/forecast"
        response = requests.get(
            api_base,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,wind_direction_10m,weather_code",
                "wind_speed_unit": "ms"
            },
            timeout=10
        )
        data = response.json()
        current = data.get("current", {})
        
        return {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "wind_speed_ms": current.get("wind_speed_10m", 8.5),
            "wind_direction_deg": current.get("wind_direction_10m", 275),
            "temperature_c": current.get("temperature_2m", 15),
            "data_source": "Open-Meteo API (Live)"
        }
    except Exception as e:
        # Simulated data
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        return {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "wind_speed_ms": np.random.uniform(6, 12),
            "wind_direction_deg": np.random.uniform(265, 290),
            "temperature_c": np.random.uniform(10, 25),
            "data_source": "Simulated Data"
        }


def fetch_weather_for_farm(farm_name: str, farm_info: dict):
    """Fetch weather data for a specific Norwegian wind farm."""
    import requests
    
    lat = farm_info["latitude"]
    lon = farm_info["longitude"]
    location = f"{farm_name}, {farm_info['location']}"
    
    # Weather data from Open-Meteo
    try:
        api_base = "https://api.open-meteo.com/v1/forecast"
        response = requests.get(
            api_base,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,wind_direction_10m,weather_code",
                "wind_speed_unit": "ms"
            },
            timeout=10
        )
        data = response.json()
        current = data.get("current", {})
        
        return {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "wind_speed_ms": current.get("wind_speed_10m", 8.5),
            "wind_direction_deg": current.get("wind_direction_10m", 275),
            "temperature_c": current.get("temperature_2m", 5),
            "data_source": "Open-Meteo API (Live)",
            "farm_name": farm_name
        }
    except Exception as e:
        # Simulated data based on typical Norwegian coastal conditions
        np.random.seed(int(datetime.now().timestamp()) % 1000 + hash(farm_name) % 100)
        return {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "wind_speed_ms": np.random.uniform(7, 14),  # Norwegian wind farms typically have good wind
            "wind_direction_deg": np.random.uniform(250, 320),  # Predominantly westerly winds
            "temperature_c": np.random.uniform(-5, 15),  # Norwegian temperatures
            "data_source": "Simulated Data",
            "farm_name": farm_name
        }


# =============================================================================
# Agent 1A: Met.no (yr.no) Weather Fetch
# =============================================================================
def fetch_weather_yr_no(lat, lon):
    """
    Fetch wind speed and direction from Met.no Locationforecast API (yr.no).
    Returns dict with wind_speed_ms and wind_direction_deg.
    """
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    headers = {
        "User-Agent": "mandar.tabib@sintef.no"  # REQUIRED: Use your email or project name
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        timeseries = data["properties"]["timeseries"]
        if not timeseries:
            raise Exception("No timeseries data found")
        latest = timeseries[0]["data"]["instant"]["details"]
        wind_speed = latest.get("wind_speed", None)
        wind_direction = latest.get("wind_from_direction", None)
        return {
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_direction,
            "data_source": "yr.no (Met.no Locationforecast API)"
        }
    except Exception as e:
        print(f"Error fetching weather from yr.no: {e}")
        return {
            "wind_speed_ms": None,
            "wind_direction_deg": None,
            "data_source": "yr.no (Met.no Locationforecast API) - Error"
        }


# =============================================================================
# Turbine Location Search Functions
# =============================================================================
def generate_turbine_grid_pattern(farm_info, num_turbines):
    """
    Generate approximate turbine locations in a grid pattern around farm center.
    This is a fallback when exact locations are not available.
    """
    center_lat = farm_info['latitude']
    center_lon = farm_info['longitude']
    
    # Estimate optimal spacing (typically 5-9 rotor diameters)
    # NREL 5MW has 126m rotor diameter, so ~630m spacing
    spacing_km = 0.63  # 630m converted to km
    
    # Calculate grid dimensions
    grid_cols = int(math.ceil(math.sqrt(num_turbines)))
    grid_rows = int(math.ceil(num_turbines / grid_cols))
    
    # Convert km to degrees (rough approximation)
    lat_deg_per_km = 1 / 111.0  # ~1 degree latitude = 111 km
    lon_deg_per_km = 1 / (111.0 * math.cos(math.radians(center_lat)))  # longitude varies with latitude
    
    turbine_locations = []
    turbine_id = 1
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if turbine_id > num_turbines:
                break
                
            # Calculate offset from center
            lat_offset = (row - grid_rows/2) * spacing_km * lat_deg_per_km
            lon_offset = (col - grid_cols/2) * spacing_km * lon_deg_per_km
            
            turbine_lat = center_lat + lat_offset
            turbine_lon = center_lon + lon_offset
            
            turbine_locations.append({
                'turbine_id': turbine_id,
                'latitude': turbine_lat,
                'longitude': turbine_lon,
                'type': 'estimated_grid'
            })
            turbine_id += 1
            
        if turbine_id > num_turbines:
            break
    
    return turbine_locations


def search_turbine_locations(farm_name, farm_info):
    """
    Search for turbine locations for a specific wind farm.
    
    Attempts multiple strategies:
    1. Known turbine databases (if accessible)
    2. Web search for publicly available data
    3. Generate estimated grid pattern as fallback
    
    Returns dict with turbine locations and search status.
    """
    num_turbines = farm_info.get('turbines', 0)
    
    search_results = {
        'farm_name': farm_name,
        'search_status': 'searching',
        'turbine_locations': [],
        'data_source': 'unknown',
        'search_methods_tried': [],
        'total_turbines_found': 0
    }
    
    # Method 1: Check for known coordinates (hardcoded database)
    search_results['search_methods_tried'].append('Known Database')
    known_locations = get_known_turbine_locations(farm_name)
    
    if known_locations:
        search_results['turbine_locations'] = known_locations
        search_results['data_source'] = 'Known Database'
        search_results['search_status'] = 'found_exact'
        search_results['total_turbines_found'] = len(known_locations)
        return search_results
    
    # Method 2: Try web search for turbine coordinates
    search_results['search_methods_tried'].append('Web Search')
    web_locations = search_web_for_turbine_locations(farm_name)
    
    if web_locations:
        search_results['turbine_locations'] = web_locations
        search_results['data_source'] = 'Web Search'
        search_results['search_status'] = 'found_web'
        search_results['total_turbines_found'] = len(web_locations)
        return search_results
    
    # Method 3: Generate estimated grid pattern
    search_results['search_methods_tried'].append('Grid Pattern Generation')
    if num_turbines > 0:
        grid_locations = generate_turbine_grid_pattern(farm_info, num_turbines)
        search_results['turbine_locations'] = grid_locations
        search_results['data_source'] = 'Estimated Grid Pattern'
        search_results['search_status'] = 'estimated'
        search_results['total_turbines_found'] = len(grid_locations)
    else:
        search_results['search_status'] = 'no_data'
        search_results['data_source'] = 'None'
    
    return search_results


def get_known_turbine_locations(farm_name):
    """
    Database of known turbine locations for Norwegian wind farms.
    This would ideally be populated with real turbine coordinate data.
    """
    # Known turbine locations (some real, some estimated based on available data)
    known_turbines = {
        "Smøla Wind Farm": [
            {'turbine_id': 1, 'latitude': 63.3980, 'longitude': 8.0100, 'type': 'known'},
            {'turbine_id': 2, 'latitude': 63.3990, 'longitude': 8.0150, 'type': 'known'},
            {'turbine_id': 3, 'latitude': 63.4000, 'longitude': 8.0200, 'type': 'known'},
            {'turbine_id': 4, 'latitude': 63.4010, 'longitude': 8.0120, 'type': 'known'},
            {'turbine_id': 5, 'latitude': 63.4020, 'longitude': 8.0180, 'type': 'known'},
            {'turbine_id': 6, 'latitude': 63.3970, 'longitude': 8.0080, 'type': 'known'},
            {'turbine_id': 7, 'latitude': 63.4030, 'longitude': 8.0220, 'type': 'known'},
            {'turbine_id': 8, 'latitude': 63.4005, 'longitude': 8.0090, 'type': 'known'},
        ],
        
        # Bessaker Wind Farm - real coordinates from provided data
        "Bessaker Wind Farm": [
            {'turbine_id': 1, 'latitude': 64.230229, 'longitude': 10.380853, 'type': 'known'},
            {'turbine_id': 2, 'latitude': 64.230062, 'longitude': 10.376347, 'type': 'known'},
            {'turbine_id': 3, 'latitude': 64.230828, 'longitude': 10.371845, 'type': 'known'},
            {'turbine_id': 4, 'latitude': 64.232406, 'longitude': 10.380896, 'type': 'known'},
            {'turbine_id': 5, 'latitude': 64.235126, 'longitude': 10.371195, 'type': 'known'},
            {'turbine_id': 6, 'latitude': 64.23371, 'longitude': 10.364704, 'type': 'known'},
            {'turbine_id': 7, 'latitude': 64.233147, 'longitude': 10.359188, 'type': 'known'},
            {'turbine_id': 8, 'latitude': 64.22204, 'longitude': 10.37226, 'type': 'known'},
            {'turbine_id': 9, 'latitude': 64.222937, 'longitude': 10.366744, 'type': 'known'},
            {'turbine_id': 10, 'latitude': 64.223687, 'longitude': 10.361344, 'type': 'known'},
            {'turbine_id': 11, 'latitude': 64.227768, 'longitude': 10.356261, 'type': 'known'},
            {'turbine_id': 12, 'latitude': 64.228715, 'longitude': 10.36017, 'type': 'known'},
            {'turbine_id': 13, 'latitude': 64.221613, 'longitude': 10.37919, 'type': 'known'},
            {'turbine_id': 14, 'latitude': 64.218486, 'longitude': 10.37555, 'type': 'known'},
            {'turbine_id': 15, 'latitude': 64.218511, 'longitude': 10.382095, 'type': 'known'},
            {'turbine_id': 16, 'latitude': 64.215674, 'longitude': 10.382909, 'type': 'known'},
            {'turbine_id': 17, 'latitude': 64.21301, 'longitude': 10.38676, 'type': 'known'},
            {'turbine_id': 18, 'latitude': 64.21262, 'longitude': 10.382618, 'type': 'known'},
            {'turbine_id': 19, 'latitude': 64.213965, 'longitude': 10.379222, 'type': 'known'},
            {'turbine_id': 20, 'latitude': 64.215294, 'longitude': 10.37591, 'type': 'known'},
            {'turbine_id': 21, 'latitude': 64.217428, 'longitude': 10.365466, 'type': 'known'},
            {'turbine_id': 22, 'latitude': 64.219413, 'longitude': 10.362274, 'type': 'known'},
            {'turbine_id': 23, 'latitude': 64.220551, 'longitude': 10.357687, 'type': 'known'},
            {'turbine_id': 24, 'latitude': 64.223222, 'longitude': 10.351111, 'type': 'known'},
            {'turbine_id': 25, 'latitude': 64.223377, 'longitude': 10.344205, 'type': 'known'},
        ],
        
        # Add more known farms here as data becomes available
        # Note: These are example/estimated coordinates for demonstration
        # Real implementations should use actual turbine coordinate databases
    }
    
    return known_turbines.get(farm_name, [])


def search_web_for_turbine_locations(farm_name):
    """
    Attempt to find turbine locations from web sources.
    This is a placeholder for more sophisticated web scraping.
    """
    # For now, return empty list as web search is complex
    # In a real implementation, this could:
    # 1. Search OpenStreetMap for wind turbine tags
    # 2. Query government databases
    # 3. Use specialized wind farm databases if available
    
    return []  # No web results found


def create_turbine_map(farm_info, turbine_locations):
    """
    Create a map visualization of turbine locations using streamlit.
    """
    if not turbine_locations:
        return None
    
    # Prepare data for map
    map_data = []
    for turbine in turbine_locations:
        map_data.append({
            'lat': turbine['latitude'],
            'lon': turbine['longitude'],
            'turbine_id': turbine['turbine_id'],
            'type': turbine.get('type', 'unknown')
        })
    
    # Convert to DataFrame for streamlit
    df = pd.DataFrame(map_data)
    
    return df


# =============================================================================
# Expert Agent Functions
# =============================================================================

def validate_turbine_pair_direction(turbine_locations, upstream_id, downstream_id, wind_direction):
    """
    Validate that downstream turbine is actually downstream based on wind direction.
    
    Args:
        turbine_locations: List of turbine location dicts with lat/lon/id
        upstream_id: ID of upstream turbine
        downstream_id: ID of downstream turbine
        wind_direction: Wind direction in degrees (meteorological: wind FROM this direction)
    
    Returns:
        tuple: (is_valid: bool, angle_diff: float, bearing: float)
    """
    try:
        # Find turbine locations
        upstream_loc = None
        downstream_loc = None
        
        for turbine in turbine_locations:
            if turbine['turbine_id'] == upstream_id:
                upstream_loc = turbine
            elif turbine['turbine_id'] == downstream_id:
                downstream_loc = turbine
        
        if upstream_loc is None or downstream_loc is None:
            return False, 999, 0
        
        # Calculate bearing from upstream to downstream
        dx = downstream_loc['longitude'] - upstream_loc['longitude']
        dy = downstream_loc['latitude'] - upstream_loc['latitude']
        
        # Bearing in degrees (0° = North, 90° = East)
        bearing_deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        
        # Wind blows FROM wind_direction TO (wind_direction + 180) % 360
        # So downstream direction is (wind_direction + 180) % 360
        downstream_direction = (wind_direction + 180) % 360
        
        # Calculate angle difference (should be small if turbine is downstream)
        angle_diff = abs((bearing_deg - downstream_direction + 180) % 360 - 180)
        
        # Valid if downstream turbine is within ±45° of wind direction
        # (±45° allows for some wake expansion and meandering)
        is_valid = angle_diff < 45.0
        
        return is_valid, angle_diff, bearing_deg
        
    except Exception as e:
        # If validation fails for any reason, reject the pair
        return False, 999, 0


def convert_latlon_to_cartesian(turbine_locations):
    """
    Convert turbine lat/lon coordinates to relative Cartesian coordinates (meters).
    Uses the first turbine as origin (0, 0).
    
    Parameters:
    -----------
    turbine_locations : list
        List of dicts with 'turbine_id', 'latitude', 'longitude'
    
    Returns:
    --------
    list
        List of dicts with added 'x_m' and 'y_m' fields (position in meters)
    """
    if not turbine_locations:
        return []
    
    # Use first turbine as reference/origin
    ref_lat = turbine_locations[0]['latitude']
    ref_lon = turbine_locations[0]['longitude']
    
    result = []
    for turbine in turbine_locations:
        lat = turbine['latitude']
        lon = turbine['longitude']
        
        # Calculate approximate x, y in meters from reference point
        # x = east-west distance, y = north-south distance
        
        # North-south distance (y-axis)
        if GEOPY_AVAILABLE:
            from geopy.distance import geodesic
            y_m = geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
            if lat < ref_lat:
                y_m = -y_m  # South is negative
            
            # East-west distance (x-axis)
            x_m = geodesic((lat, ref_lon), (lat, lon)).meters
            if lon < ref_lon:
                x_m = -x_m  # West is negative
        else:
            # Fallback calculation
            y_m = calculate_distance_fallback(ref_lat, ref_lon, lat, ref_lon)
            if lat < ref_lat:
                y_m = -y_m
            x_m = calculate_distance_fallback(lat, ref_lon, lat, lon)
            if lon < ref_lon:
                x_m = -x_m
        
        turbine_copy = turbine.copy()
        turbine_copy['x_m'] = round(x_m, 1)
        turbine_copy['y_m'] = round(y_m, 1)
        result.append(turbine_copy)
    
    return result


def get_turbine_pair_recommendations(turbine_locations, wind_speed, wind_dir, provider='NTNU', model='moonshotai/Kimi-K2.5'):
    """
    Agent 2C: LLM-based turbine pair selection for wake optimization.
    
    Analyzes turbine locations and wind conditions to identify critical turbine pairs
    that are most affected by wake effects based on proximity and wind direction.
    
    Args:
        turbine_locations: List of turbine location dicts with lat/lon/id
        wind_speed: Wind speed in m/s
        wind_dir: Wind direction in degrees
        provider: LLM provider
        model: LLM model name
    
    Returns:
        dict: Analysis results with turbine pairs and recommendations
    """
    if not turbine_locations or len(turbine_locations) < 2:
        return {
            'agent': 'Agent 2C - Turbine Pair Selector',
            'status': 'error',
            'message': 'Insufficient turbine data for pair analysis',
            'turbine_pairs': [],
            'analysis': 'Need at least 2 turbines for wake analysis'
        }
    
    try:
        # Convert lat/lon to Cartesian coordinates for LLM
        turbine_locations_xy = convert_latlon_to_cartesian(turbine_locations)
        
        # Prepare turbine data with BOTH geographic and Cartesian coordinates
        turbine_summary = []
        for turbine in turbine_locations_xy:
            # Format: T1(x=350m, y=120m)
            turbine_summary.append(
                f"T{turbine['turbine_id']}(x={turbine['x_m']:.0f}m, y={turbine['y_m']:.0f}m)"
            )
        
        # Create CLEAR prompt with Cartesian coordinates
        prompt = f"""Wind farm wake analysis with {len(turbine_locations)} turbines.

WIND CONDITIONS:
- Speed: {wind_speed:.1f} m/s
- Direction: {wind_dir:.0f}° (wind blowing FROM {wind_dir:.0f}° TOWARD {(wind_dir + 180) % 360:.0f}°)
- Meteorological convention: 0°=North, 90°=East, 180°=South, 270°=West

TURBINE POSITIONS (Cartesian coordinates in meters, origin at T1):
{chr(10).join(turbine_summary)}

TASK:
Identify 3-5 critical upstream→downstream turbine pairs most affected by wake effects.

CRITERIA:
1. Downstream turbine must be aligned with wind direction (±20° tolerance)
2. Distance: 600-1200m apart (5-10 rotor diameters for NREL 5MW with D=126m)
3. Prioritize pairs with strongest wake interactions

RETURN FORMAT - JSON only:
{{
  "critical_pairs": [
    {{
      "upstream_turbine": 1,
      "downstream_turbine": 5,
      "distance_km": 0.85,
      "wake_strength": "high",
      "priority": 1
    }}
  ],
  "analysis_summary": "Brief explanation",
  "optimization_strategy": "Optimization approach"
}}

Respond with JSON:"""
        
        # Log prompt for debugging
        import sys
        print(f"\n[Agent 2C Debug] Turbines converted to Cartesian coordinates", file=sys.stderr)
        if len(turbine_locations_xy) > 0:
            print(f"[Agent 2C Debug] Example: T1 at origin (0m, 0m)", file=sys.stderr)
            if len(turbine_locations_xy) > 1:
                print(f"[Agent 2C Debug] Example: T{turbine_locations_xy[1]['turbine_id']} at ({turbine_locations_xy[1]['x_m']:.0f}m, {turbine_locations_xy[1]['y_m']:.0f}m)", file=sys.stderr)
        print(f"[Agent 2C Debug] Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)", file=sys.stderr)
        print(f"[Agent 2C Debug] Analyzing {len(turbine_locations)} turbines", file=sys.stderr)
        print(f"[Agent 2C Debug] Wind conditions: {wind_speed}m/s from {wind_dir}°", file=sys.stderr)
        
        # Get LLM configuration (user-provided key → env → Streamlit secrets)
        api_base, api_key = GlobalLLMConfig.get_api_config()

        config = {
            'provider': provider,
            'model': model,
            'api_base': api_base,
            'api_key': api_key,
            'temperature': 0.1,
            'max_tokens': 2000,
            'timeout': 30.0
        }
        
        # Query the LLM using the unified interface
        # Agent 2C requires MORE tokens and time than Agent 2B due to complex JSON output
        print(f"[Agent 2C Debug] Calling LLM (Attempt 1: WITH system message) provider={provider}, model={GlobalLLMConfig.get_model()}", file=sys.stderr)
        response = query_unified_llm(
            prompt=prompt,
            system_message="You are a wind farm wake analysis expert. Respond ONLY with valid JSON.",
            temperature=0.1,
            max_tokens=4000,
            timeout=60.0
        )
        print(f"[Agent 2C Debug] LLM response length: {len(response)} chars", file=sys.stderr)
        print(f"[Agent 2C Debug] Response preview: {response[:200]}...", file=sys.stderr)
        
        # If empty response, try WITHOUT system message (some models don't handle it well)
        if response.startswith('ERROR_EMPTY_RESPONSE'):
            print(f"[Agent 2C Debug] First attempt failed with empty response. Trying WITHOUT system message...", file=sys.stderr)
            time.sleep(2)
            
            response = query_unified_llm(
                prompt=f"Analyze wind farm wake interactions and respond with JSON.\n\n{prompt}",
                system_message=None,  # Try without system message
                temperature=0.2,
                max_tokens=4000,
                timeout=60.0
            )
            print(f"[Agent 2C Debug] Attempt 2 response length: {len(response)} chars", file=sys.stderr)
        
        # If still empty, try asking for plain text instead of JSON
        if response.startswith('ERROR_EMPTY_RESPONSE'):
            print(f"[Agent 2C Debug] Second attempt also failed. Trying plain text request...", file=sys.stderr)
            time.sleep(2)
            
            simple_prompt = f"""List the critical turbine pairs for wake optimization.

Wind: {wind_speed:.1f}m/s from {wind_dir:.0f}°
Turbines: {', '.join([f'T{t["turbine_id"]}' for t in turbine_locations[:10]])}{' ...' if len(turbine_locations) > 10 else ''}

Identify 3-5 upstream→downstream pairs aligned with wind direction.
Format: T1→T2, T3→T4, etc."""
            
            response = query_unified_llm(
                prompt=simple_prompt,
                system_message=None,
                temperature=0.3,
                max_tokens=2000,
                timeout=45.0
            )
            print(f"[Agent 2C Debug] Attempt 3 (simple) response length: {len(response)} chars", file=sys.stderr)
        
        # Check for error indicators in the response
        if response.startswith('ERROR_'):
            error_type = response.split(':')[0].replace('ERROR_', '')
            error_message = response.split(':', 1)[1] if ':' in response else response
            
            print(f"[Agent 2C Debug] ERROR detected: {error_type}", file=sys.stderr)
            print(f"[Agent 2C Debug] Error message: {error_message}", file=sys.stderr)
            
            return {
                'agent': 'Agent 2C - Turbine Pair Selector',
                'status': 'error',
                'provider': provider,
                'model': model,
                'turbine_pairs': [],
                'message': f'LLM Error ({error_type}): {error_message}',
                'analysis_summary': f'Failed due to {error_type} after trying 3 different approaches. ' + (
                    'Rate limit exceeded - wait and try again.' if error_type == 'RATE_LIMIT' else
                    'API call failed - check network/configuration.' if error_type == 'API_CALL' else
                    'Model returned empty response despite: (1) JSON request, (2) No system message, (3) Simple text request. This model may not be compatible with this task. **Use Agent 2D** or try a different model (e.g., GPT-4, Claude).'
                ),
                'total_turbines': len(turbine_locations),
                'wind_conditions': f"{wind_speed:.1f} m/s from {wind_dir:.0f}°",
                'raw_response': response,
                'error_code': '429' if error_type == 'RATE_LIMIT' else None,
                'prompt_length': len(prompt),
                'prompt_preview': prompt[:500],
                'recovery_attempts': 'Tried: (1) JSON with system msg, (2) JSON without system msg, (3) Plain text request - all failed'
            }
        
        # Parse JSON response
        import json
        import re
        try:
            # Try direct JSON parsing first
            try:
                analysis_data = json.loads(response)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown code blocks or text
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group(1))
                else:
                    # Try to find JSON object in the response (more flexible)
                    json_match = re.search(r'(\{[^{}]*"critical_pairs".*?\})', response, re.DOTALL)
                    if json_match:
                        analysis_data = json.loads(json_match.group(1))
                    else:
                        # No JSON found, raise error to fall back to text parsing
                        raise json.JSONDecodeError("No JSON found in response", response, 0)
            
            llm_pairs = analysis_data.get('critical_pairs', [])
            
            # DEBUG: Log what the LLM returned
            import sys
            print(f"\n[DEBUG Agent 2C] LLM returned {len(llm_pairs)} pairs", file=sys.stderr)
            if llm_pairs:
                print(f"[DEBUG Agent 2C] First pair content: {llm_pairs[0]}", file=sys.stderr)
                print(f"[DEBUG Agent 2C] First pair keys: {list(llm_pairs[0].keys())}", file=sys.stderr)
                print(f"[DEBUG Agent 2C] 'distance_km' in first pair? {'distance_km' in llm_pairs[0]}", file=sys.stderr)
            
            # Validate each pair to ensure downstream turbine is actually downstream
            validated_pairs = []
            rejected_pairs = []
            
            for pair in llm_pairs:
                upstream_id = pair.get('upstream_turbine')
                downstream_id = pair.get('downstream_turbine')
                
                if upstream_id is None or downstream_id is None:
                    rejected_pairs.append({**pair, 'reason': 'Missing turbine IDs'})
                    continue
                
                is_valid, angle_diff, bearing = validate_turbine_pair_direction(
                    turbine_locations, upstream_id, downstream_id, wind_dir
                )
                
                if is_valid:
                    # Add validation info to the pair
                    validated_pair = pair.copy()
                    validated_pair['validated'] = True
                    validated_pair['bearing_deg'] = round(bearing, 1)
                    validated_pair['angle_deviation_deg'] = round(angle_diff, 1)
                    
                    # Calculate distance in rotor diameters for optimizer (NREL 5MW: D=126m)
                    if 'distance_km' in validated_pair:
                        distance_m = validated_pair['distance_km'] * 1000.0
                        validated_pair['distance_rotor_diameters'] = distance_m / 126.0
                        print(f"[DEBUG Agent 2C] Pair {upstream_id}->{downstream_id}: distance_km={validated_pair['distance_km']:.3f}, distance_rotor_diameters={validated_pair['distance_rotor_diameters']:.2f}", file=sys.stderr)
                    else:
                        print(f"[DEBUG Agent 2C] Pair {upstream_id}->{downstream_id}: 'distance_km' NOT FOUND in validated_pair! Keys: {list(validated_pair.keys())}", file=sys.stderr)
                    
                    validated_pairs.append(validated_pair)
                else:
                    rejected_pairs.append({
                        **pair, 
                        'reason': f'Not aligned with wind (deviation: {angle_diff:.1f}°)',
                        'bearing_deg': round(bearing, 1)
                    })
            
            # Build analysis summary with validation info
            validation_summary = f"\nValidated {len(validated_pairs)}/{len(llm_pairs)} LLM-identified pairs."
            if rejected_pairs:
                validation_summary += f" Rejected {len(rejected_pairs)} pair(s) not aligned with wind direction."
            
            return {
                'agent': 'Agent 2C - Turbine Pair Selector',
                'status': 'success',
                'provider': provider,
                'model': model,
                'turbine_pairs': validated_pairs,
                'rejected_pairs': rejected_pairs,
                'llm_identified_pairs': len(llm_pairs),
                'validated_pairs_count': len(validated_pairs),
                'analysis_summary': analysis_data.get('analysis_summary', '') + validation_summary,
                'optimization_strategy': analysis_data.get('optimization_strategy', ''),
                'total_turbines': len(turbine_locations),
                'wind_conditions': f"{wind_speed:.1f} m/s from {wind_dir:.0f}°",
                'raw_response': response
            }
        except json.JSONDecodeError as json_err:
            # Fallback: The LLM returned text but not valid JSON
            # This is a partial success - we got a response but can't parse it
            return {
                'agent': 'Agent 2C - Turbine Pair Selector',
                'status': 'partial',
                'provider': provider,
                'model': model,
                'turbine_pairs': [],
                'message': f'LLM returned text response instead of JSON format',
                'analysis_summary': f"LLM provided text analysis but not in required JSON format. Response preview: {response[:300]}{'...' if len(response) > 300 else ''}",
                'optimization_strategy': 'The model may not support JSON output. Try: 1) Different model, 2) Use Agent 2D',
                'total_turbines': len(turbine_locations),
                'wind_conditions': f"{wind_speed:.1f} m/s from {wind_dir:.0f}°",
                'raw_response': response,
                'parse_error': f'JSON parsing failed: {str(json_err)}'
            }
    
    except Exception as e:
        error_msg = str(e)
        error_code = None
        
        # Extract error code if present
        if '401' in error_msg:
            error_code = '401'
        elif '403' in error_msg:
            error_code = '403'
        elif '429' in error_msg:
            error_code = '429'
        elif '500' in error_msg or '502' in error_msg or '503' in error_msg:
            error_code = '5xx'
        
        return {
            'agent': 'Agent 2C - Turbine Pair Selector',
            'status': 'error',
            'error_code': error_code,
            'message': f'Error in turbine pair analysis: {error_msg}',
            'turbine_pairs': [],
            'analysis': 'Analysis failed due to technical error',
            'provider': provider,
            'model': model
        }


def get_expert_recommendation(wind_speed: float, wind_direction: float):
    """Get expert recommendation for yaw angle."""
    
    turbine_specs = {
        "name": "NREL 5MW Reference Wind Turbine",
        "rated_power_mw": 5.0,
        "rotor_diameter_m": 126,
        "hub_height_m": 90,
        "cut_in_wind_speed_ms": 3.0,
        "rated_wind_speed_ms": 11.4,
        "cut_out_wind_speed_ms": 25.0
    }
    
    # ROM model constraints (CFD training data range)
    rom_yaw_min, rom_yaw_max = 272, 285
    
    # Actual real-world recommendation
    actual_yaw = wind_direction  # In reality, turbine aligns with wind
    actual_pitch = 0.0  # Normal operating pitch (fine pitch)
    
    reasoning = []
    actual_recommendation = ""
    rom_explanation = ""
    
    # Determine operating region and actual recommendation
    if wind_speed < turbine_specs["cut_in_wind_speed_ms"]:
        operating_region = "Below Cut-in"
        actual_pitch = 90.0  # Feathered position
        actual_recommendation = (
            f"**Actual Real-World Recommendation:** At wind speed of {wind_speed:.1f} m/s (below cut-in of "
            f"{turbine_specs['cut_in_wind_speed_ms']} m/s), the turbine should enter standby/idle mode. "
            f"The recommended yaw angle is {wind_direction:.1f}° (aligned with wind direction) to be ready "
            f"when wind increases. Blade pitch should be feathered to ~90° to minimize rotor rotation and "
            f"reduce mechanical wear on bearings and gearbox. The generator remains disconnected from the grid."
        )
        reasoning.append(f"Wind speed ({wind_speed:.1f} m/s) is below cut-in speed ({turbine_specs['cut_in_wind_speed_ms']} m/s)")
        reasoning.append("Turbine enters standby mode - no power generation")
        reasoning.append(f"Actual recommendation: Yaw = {wind_direction:.1f}°, Pitch = 90° (feathered)")
        
    elif wind_speed > turbine_specs["cut_out_wind_speed_ms"]:
        operating_region = "Above Cut-out"
        actual_pitch = 90.0  # Feathered for protection
        actual_recommendation = (
            f"**Actual Real-World Recommendation:** At wind speed of {wind_speed:.1f} m/s (above cut-out of "
            f"{turbine_specs['cut_out_wind_speed_ms']} m/s), the turbine should shut down for safety. "
            f"Blades should be feathered to 90° to reduce loads. The nacelle may yaw out of the wind "
            f"or remain aligned depending on the control strategy."
        )
        reasoning.append(f"Wind speed ({wind_speed:.1f} m/s) exceeds cut-out speed")
        reasoning.append("Emergency shutdown - blades feathered for protection")
        
    elif wind_speed <= turbine_specs["rated_wind_speed_ms"]:
        operating_region = "Partial Load (Region 2)"
        actual_pitch = 0.0  # Fine pitch for max power capture
        actual_recommendation = (
            f"**Actual Real-World Recommendation:** At wind speed of {wind_speed:.1f} m/s (partial load region), "
            f"the turbine should align precisely with the wind direction at {wind_direction:.1f}°. "
            f"Blade pitch should be at fine pitch (~0°) to maximize aerodynamic torque and energy capture. "
            f"In this region, even small yaw misalignments significantly reduce power output (power ∝ cos³(misalignment))."
        )
        reasoning.append("Operating in partial load region - maximize energy capture")
        reasoning.append(f"Actual recommendation: Yaw = {wind_direction:.1f}°, Pitch = 0° (fine pitch)")
        
    else:
        operating_region = "Full Load (Region 3)"
        actual_pitch = 10.0  # Pitched to limit power
        actual_recommendation = (
            f"**Actual Real-World Recommendation:** At wind speed of {wind_speed:.1f} m/s (above rated), "
            f"the turbine operates at full load. Yaw should align with wind at {wind_direction:.1f}°. "
            f"Blade pitch increases (typically 10-25°) to shed excess power and maintain rated output of "
            f"{turbine_specs['rated_power_mw']} MW. Wake steering may apply a small intentional yaw offset "
            f"(+3-5°) to benefit downstream turbines in a wind farm."
        )
        reasoning.append("Operating in full load region - pitch to limit power")
        reasoning.append(f"Actual recommendation: Yaw = {wind_direction:.1f}°, Pitch = ~10-25°")
    
    # Select yaw for ROM testing (constrained to training data range)
    rom_yaw = np.clip(wind_direction, rom_yaw_min, rom_yaw_max)
    
    # Calculate efficiency deviation due to ROM constraint
    yaw_misalignment = abs(rom_yaw - wind_direction)
    rom_efficiency = np.cos(np.radians(yaw_misalignment)) ** 3
    ideal_efficiency = 1.0  # 100% when perfectly aligned
    efficiency_deviation = (ideal_efficiency - rom_efficiency) * 100
    
    # ROM explanation
    rom_explanation = (
        f"**For ROM Testing:** The Reduced Order Model was trained on high-fidelity CFD simulations "
        f"with yaw angles between {rom_yaw_min}° and {rom_yaw_max}°. This range was selected because: "
        f"(1) It represents typical westerly wind conditions at Norwegian wind farms, "
        f"(2) CFD simulations are computationally expensive (~1000+ CPU-hours each), so only a representative "
        f"parameter range was simulated, and (3) This range captures the key wake dynamics for yaw-based control studies. "
        f"The ROM test yaw is set to {rom_yaw:.1f}°, which deviates {yaw_misalignment:.1f}° from the actual wind direction. "
        f"This results in an efficiency of {rom_efficiency*100:.1f}% compared to perfect alignment, "
        f"representing a {efficiency_deviation:.1f}% power loss due to this constraint."
    )
    
    return {
        "suggested_yaw": rom_yaw,  # For ROM testing
        "actual_yaw": actual_yaw,  # Real-world recommendation
        "actual_pitch": actual_pitch,
        "yaw_misalignment": yaw_misalignment,
        "expected_efficiency": rom_efficiency,
        "efficiency_deviation": efficiency_deviation,
        "operating_region": operating_region,
        "reasoning": reasoning,
        "actual_recommendation": actual_recommendation,
        "rom_explanation": rom_explanation,
        "rom_yaw_range": (rom_yaw_min, rom_yaw_max),
        "turbine_specs": turbine_specs
    }


# =============================================================================
# Helper Functions for Agent 2D
# =============================================================================


def calculate_distance_fallback(lat1, lon1, lat2, lon2):
    """
    Calculate approximate distance between two lat/lon points using Haversine formula.
    Fallback when geopy is not available.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of first point
    lat2, lon2 : float
        Latitude and longitude of second point
    
    Returns:
    --------
    float
        Distance in meters
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    
    return distance


# =============================================================================
# Agent 2D: Wake-Based Turbine Pair Identification (Backup to Agent 2C)
# =============================================================================
def get_wake_influenced_turbine_pairs(turbine_locations, wind_speed, wind_direction, 
                                       rotor_diameter=126.0, hub_height=90.0, 
                                       wake_expansion_factor=0.1, min_influence_threshold=0.05):
    """
    Agent 2D: Identify turbine pairs influenced by wake deficits using physical wake models.
    
    This is a backup to Agent 2C (LLM-based) when LLM services are unavailable.
    Uses engineering wake models to identify critical turbine interactions.
    
    Parameters:
    -----------
    turbine_locations : list or pd.DataFrame
        Turbine locations with columns ['turbine_id', 'lat', 'lon'] and optional 'hub_height'
    wind_speed : float
        Wind speed in m/s
    wind_direction : float  
        Wind direction in degrees (meteorological convention: 0° = North, 90° = East)
    rotor_diameter : float
        Rotor diameter in meters (default: 126.0 for NREL 5MW)
    hub_height : float
        Hub height in meters (default: 90.0 for NREL 5MW)
    wake_expansion_factor : float
        Wake expansion coefficient (default: 0.1)
    min_influence_threshold : float
        Minimum wake influence to consider (default: 0.05 = 5%)
    
    Returns:
    --------
    dict
        Results containing turbine pairs and chains influenced by wake
    """
    import numpy as np
    import pandas as pd
    
    # Convert to DataFrame if needed
    if isinstance(turbine_locations, list):
        df = pd.DataFrame(turbine_locations)
    else:
        df = turbine_locations.copy()
    
    if df.empty:
        return {
            'agent_type': 'Agent 2D (Wake-Based)',
            'turbine_pairs': [],
            'turbine_chains': [],
            'wake_analysis': 'No turbine locations provided',
            'total_pairs_found': 0,
            'method': 'Physical Wake Model',
            'parameters': {
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'rotor_diameter': rotor_diameter,
                'hub_height': hub_height
            },
            'status': 'no_pairs'
        }
    
    # Normalize column names to handle different formats
    # Accept 'latitude'/'longitude' or 'lat'/'lon'
    if 'latitude' in df.columns and 'lat' not in df.columns:
        df['lat'] = df['latitude']
    if 'longitude' in df.columns and 'lon' not in df.columns:
        df['lon'] = df['longitude']
    
    # Verify required columns exist
    if 'lat' not in df.columns or 'lon' not in df.columns:
        # Try one more time with case-insensitive search
        lat_col = None
        lon_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower and lat_col is None:
                lat_col = col
            if 'lon' in col_lower and lon_col is None:
                lon_col = col
        
        if lat_col and lon_col:
            df['lat'] = df[lat_col]
            df['lon'] = df[lon_col]
        else:
            return {
                'agent_type': 'Agent 2D (Wake-Based)',
                'turbine_pairs': [],
                'turbine_chains': [],
                'wake_analysis': f'Invalid turbine location data. Missing latitude/longitude columns. Found: {list(df.columns)}',
                'total_pairs_found': 0,
                'method': 'Physical Wake Model',
                'error': f'Missing lat/lon columns. Available columns: {list(df.columns)}',
                'parameters': {
                    'wind_speed': wind_speed,
                    'wind_direction': wind_direction,
                    'rotor_diameter': rotor_diameter,
                    'hub_height': hub_height
                },
                'status': 'error'
            }
    
    # Ensure turbine_id exists
    if 'turbine_id' not in df.columns:
        df['turbine_id'] = range(1, len(df) + 1)
    
    # Ensure required columns
    if 'hub_height' not in df.columns:
        df['hub_height'] = hub_height
    
    n_turbines = len(df)
    wake_pairs = []
    wake_influences = []
    
    # Convert wind direction from meteorological to mathematical convention
    # Meteorological: 0° = North, 90° = East
    # Mathematical: 0° = East, 90° = North
    math_wind_dir = (90 - wind_direction) % 360
    wind_dir_rad = np.radians(math_wind_dir)
    
    # Calculate distances and relative positions for all turbine pairs
    for i in range(n_turbines):
        turb_i = df.iloc[i]
        
        for j in range(n_turbines):
            if i == j:
                continue
                
            turb_j = df.iloc[j]
            
            # Calculate distance between turbines
            if GEOPY_AVAILABLE:
                dist_m = geodesic(
                    (turb_i['lat'], turb_i['lon']),
                    (turb_j['lat'], turb_j['lon'])
                ).meters
            else:
                # Use fallback distance calculation
                dist_m = calculate_distance_fallback(
                    turb_i['lat'], turb_i['lon'],
                    turb_j['lat'], turb_j['lon']
                )
            
            # Calculate relative position vector (i -> j)
            lat_diff = turb_j['lat'] - turb_i['lat']
            lon_diff = turb_j['lon'] - turb_i['lon']
            
            # Convert to meters (approximate for small distances)
            # More accurate would be using proper projection, but this is sufficient for relative positioning
            lat_dist = lat_diff * 111320  # meters per degree latitude
            lon_dist = lon_diff * 111320 * np.cos(np.radians(turb_i['lat']))  # adjust for longitude convergence
            
            # Vector from turb_i to turb_j
            dx = lon_dist
            dy = lat_dist
            
            # Check if turb_j is downstream of turb_i
            # Project position vector onto wind direction
            downstream_dist = dx * np.cos(wind_dir_rad) + dy * np.sin(wind_dir_rad)
            lateral_dist = abs(-dx * np.sin(wind_dir_rad) + dy * np.cos(wind_dir_rad))
            
            # Only consider if turb_j is significantly downstream
            if downstream_dist > rotor_diameter:  # Must be at least 1D downstream
                
                # Calculate wake width at downstream distance using linear expansion
                wake_radius_at_j = rotor_diameter / 2 + wake_expansion_factor * downstream_dist
                
                # Check if downstream turbine is within wake influence zone
                if lateral_dist <= wake_radius_at_j:
                    
                    # Calculate wake deficit using simplified Jensen model
                    # Ct = 0.8 (typical thrust coefficient)
                    ct = 0.8
                    wake_deficit = (1 - np.sqrt(1 - ct)) / (1 + 2 * wake_expansion_factor * downstream_dist / rotor_diameter) ** 2
                    
                    # Consider influence significant if above threshold
                    if wake_deficit >= min_influence_threshold:
                        
                        wake_pairs.append({
                            'upstream_turbine': int(turb_i['turbine_id']),
                            'downstream_turbine': int(turb_j['turbine_id']),
                            'distance_m': dist_m,
                            'distance_D': dist_m / rotor_diameter,
                            'downstream_distance_m': downstream_dist,
                            'lateral_distance_m': lateral_dist,
                            'wake_deficit': wake_deficit,
                            'wake_radius_m': wake_radius_at_j
                        })
                        
                        wake_influences.append(wake_deficit)
    
    # Sort pairs by wake influence (strongest first)
    wake_pairs.sort(key=lambda x: x['wake_deficit'], reverse=True)
    
    # Create turbine chains by merging pairs with common turbines
    turbine_chains = create_turbine_chains(wake_pairs)
    
    # Format results for compatibility with Agent 2C output
    formatted_pairs = []
    import sys
    print(f"\n[DEBUG Agent 2D] Formatting {len(wake_pairs[:10])} pairs with distance_rotor_diameters", file=sys.stderr)
    
    for pair in wake_pairs[:10]:  # Limit to top 10 most critical pairs
        formatted_pair = {
            'upstream_turbine': pair['upstream_turbine'],  # Consistent with Agent 2C
            'downstream_turbine': pair['downstream_turbine'],  # Consistent with Agent 2C
            'distance_km': pair['distance_m'] / 1000.0,
            'wake_strength': (
                'high' if pair['wake_deficit'] > 0.15 else
                'medium' if pair['wake_deficit'] > 0.08 else 'low'
            ),
            'priority': len(formatted_pairs) + 1,
            'distance_rotor_diameters': pair['distance_D'],
            'downstream_distance_D': pair['downstream_distance_m'] / rotor_diameter,
            'lateral_offset_D': pair['lateral_distance_m'] / rotor_diameter,
            'wake_deficit': pair['wake_deficit']
        }
        formatted_pairs.append(formatted_pair)
        
        if len(formatted_pairs) <= 2:  # Log first 2 pairs
            print(f"[DEBUG Agent 2D] Pair {formatted_pair['upstream_turbine']}->{formatted_pair['downstream_turbine']}: distance_rotor_diameters={formatted_pair['distance_rotor_diameters']:.2f}", file=sys.stderr)
    
    # Calculate summary statistics
    total_pairs = len(wake_pairs)
    avg_influence = np.mean(wake_influences) if wake_influences else 0
    max_influence = max(wake_influences) if wake_influences else 0
    
    wake_analysis = (
        f"Agent 2D identified {total_pairs} wake-influenced turbine pairs using physical models. "
        f"Wind from {wind_direction:.0f}° at {wind_speed:.1f} m/s creates wake deficits up to {max_influence:.1%}. "
        f"Average wake influence: {avg_influence:.1%}. "
        f"Analysis based on Jensen wake model with {wake_expansion_factor:.2f} expansion factor."
    )
    
    # Determine status
    if total_pairs == 0:
        status = 'no_pairs'
    elif total_pairs > 0:
        status = 'success'
    else:
        status = 'partial'
    
    return {
        'status': status,
        'agent_type': 'Agent 2D (Wake-Based)',
        'turbine_pairs': formatted_pairs,
        'turbine_chains': turbine_chains,
        'raw_wake_pairs': wake_pairs,  # Detailed wake calculations
        'wake_analysis': wake_analysis,
        'total_pairs_found': total_pairs,
        'max_wake_deficit': max_influence,
        'avg_wake_deficit': avg_influence,
        'method': 'Jensen Wake Model + Physical Constraints',
        'parameters': {
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'rotor_diameter': rotor_diameter,
            'hub_height': hub_height,
            'wake_expansion_factor': wake_expansion_factor,
            'min_influence_threshold': min_influence_threshold
        }
    }


def create_turbine_chains(wake_pairs):
    """
    Create turbine chains by merging pairs with common turbines.
    
    Parameters:
    -----------
    wake_pairs : list
        List of wake pair dictionaries with 'upstream_turbine' and 'downstream_turbine'
    
    Returns:
    --------
    list
        List of turbine chains (lists of turbine IDs in wake order)
    """
    if not wake_pairs:
        return []
    
    # Create a graph of wake connections
    wake_graph = {}
    for pair in wake_pairs:
        upstream = pair['upstream_turbine']
        downstream = pair['downstream_turbine']
        
        if upstream not in wake_graph:
            wake_graph[upstream] = []
        wake_graph[upstream].append(downstream)
    
    # Find chains by following downstream connections
    chains = []
    processed = set()
    
    for pair in wake_pairs:
        upstream = pair['upstream_turbine']
        
        if upstream not in processed:
            # Start a new chain from this upstream turbine
            chain = [upstream]
            current = upstream
            
            # Follow the chain downstream
            while current in wake_graph:
                # Find the next turbine in chain (prefer strongest connection)
                downstream_options = wake_graph[current]
                if downstream_options:
                    # Take the first downstream turbine (pairs are sorted by strength)
                    next_turbine = downstream_options[0]
                    if next_turbine not in chain:  # Avoid cycles
                        chain.append(next_turbine)
                        current = next_turbine
                    else:
                        break
                else:
                    break
            
            # Only add chains with multiple turbines
            if len(chain) > 1:
                chains.append(chain)
                processed.update(chain)
    
    # Sort chains by length (longest first) and then by wake strength
    chains.sort(key=lambda x: len(x), reverse=True)
    
    return chains


# =============================================================================
# Agent 2B: LLM-based Turbine Expert Functions
# =============================================================================
def load_llm_config(config_path: str = None):
    """
    Load LLM configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config.yaml file. If None, uses default path.
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    import yaml
    
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        llm_config = config.get('llm', {})
        # Resolve environment variable placeholders like ${NTNU_API_KEY}
        for key, value in llm_config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                llm_config[key] = os.getenv(env_var, value)
        return llm_config
    except FileNotFoundError:
        st.warning(f"⚠️ Config file not found at {config_path}. Using default values.")
        return {
            'api_key': '<INSERT_YOUR_API_KEY_HERE>',
            'api_base': 'http://localhost:8000/v1',
            'model': 'moonshotai/Kimi-K2.5',
            'temperature': 0.1,
            'max_tokens': 200000,
            'timeout': 1000.0
        }
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}


def query_unified_llm(prompt: str, system_message: str = None, temperature: float = None, max_tokens: int = 1000, timeout: float = None):
    """
    Unified LLM query function using global configuration and modern LLMFactory interface.
    
    Parameters:
    -----------
    prompt : str
        The user prompt/query for the LLM
    system_message : str, optional
        System message to set the context for the LLM
    temperature : float, optional 
        Sampling temperature (uses global config if None)
    max_tokens : int, optional
        Maximum number of tokens to generate (default: 1000)
    timeout : float, optional
        Timeout for the API request (default: None)
    
    Returns:
    --------
    str
        The response from the LLM
    """
    try:
        # Use global temperature if not specified
        if temperature is None:
            temperature = GlobalLLMConfig.get_temperature()
        
        # Try modern LLMFactory first
        try:
            llm = GlobalLLMConfig.create_llm_instance()
            if llm:
                import asyncio
                
                async def get_response():
                    return await llm.complete(
                        prompt=prompt,
                        system=system_message,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                
                # Run async function in sync context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If already in event loop, create a task
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, get_response())
                            return future.result(timeout=timeout)
                    else:
                        return asyncio.run(get_response())
                except:
                    # Fallback for sync execution
                    return asyncio.run(get_response())
        except Exception as e:
            st.warning(f"LLMFactory failed, using fallback: {e}")
        
        # Fallback to legacy query_local_llm
        api_base, api_key = GlobalLLMConfig.get_api_config()
        model_name = GlobalLLMConfig.get_model()
        
        return query_local_llm(
            api_key=api_key,
            api_base=api_base, 
            model_name=model_name,
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    except Exception as e:
        error_msg = f"LLM query failed: {str(e)}"
        st.error(error_msg)
        return f"Error: {error_msg}"


def query_local_llm(api_key: str, api_base: str, model_name: str, prompt: str, system_message: str = None, temperature: float = 0.3, max_tokens: int = 1000, timeout: float = None, max_retries: int = 3):
    """
    Query the local LLM using OpenAI-compatible API (openai>=1.0.0 interface) with retry logic.
    
    Parameters:
    -----------
    api_key : str
        API key to authenticate with the local LLM server
    api_base : str
        Base URL of the local LLM server (e.g., "http://localhost:8000/v1")
    model_name : str
        Name of the LLM model (e.g., "moonshotai/Kimi-K2.5")
    prompt : str
        The user prompt/query for the LLM
    system_message : str, optional
        System message to set the context for the LLM
    temperature : float, optional
        Sampling temperature for the LLM (default: 0.3 for more deterministic responses)
    max_tokens : int, optional
        Maximum number of tokens to generate (default: 1000)
    timeout : float, optional
        Timeout for the API request (default: None)
    max_retries : int, optional
        Maximum number of retry attempts (default: 3)

    Returns:
    --------
    str
        The response from the LLM
    """
    # Use the new OpenAI API client (>=1.0.0)
    import openai
    import time
    
    client = openai.OpenAI(api_key=api_key, base_url=api_base)

    for attempt in range(max_retries):
        try:
            messages = []
            if system_message:
                # Don't enhance system message - use it as provided
                # Agent 2C passes specific instructions that shouldn't be modified
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})

            create_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if timeout is not None:
                create_kwargs["timeout"] = timeout

            response = client.chat.completions.create(**create_kwargs)
            answer = response.choices[0].message.content
            
            # Validate response - check for empty, "none", or too short responses
            if answer and answer.strip() and answer.strip().lower() not in ['none', 'n/a', 'null']:
                if len(answer.strip()) > 20:  # Ensure substantive answer
                    return answer.strip()
            
            # If response is invalid and we have retries left, try again
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief delay before retry
                continue
            else:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # Longer delay after error before retry
                continue
            else:
                return f"Error querying the LLM after {max_retries} attempts: {str(e)}"
    
    return "Unable to get a response from the LLM. Please try again."


def get_llm_expert_recommendation(wind_speed: float, wind_direction: float, config: dict = None):
    """
    Use the local LLM to get expert recommendations for yaw and pitch.
    
    Parameters:
    -----------
    wind_speed : float
        Wind speed in m/s
    wind_direction : float
        Wind direction in degrees
    config : dict, optional
        LLM configuration dictionary. If None, uses GUI selections from session state
    
    Returns:
    --------
    dict
        LLM response with recommendations
    """
    # Use GUI selections if no config provided
    if config is None:
        # Get provider and model from GUI selections with fallbacks
        provider = getattr(st.session_state, 'llm_provider', 'NTNU')
        model = getattr(st.session_state, 'selected_model', 'moonshotai/Kimi-K2.5')
        
        # Get API configuration (user-provided key → env → Streamlit secrets)
        api_base, api_key = GlobalLLMConfig.get_api_config()

        config = {
            'provider': provider,
            'model': model,
            'api_base': api_base,
            'api_key': api_key,
            'temperature': 0.1,
            'max_tokens': 200000,
            'timeout': 1000.0
        }
    else:
        # Fallback to config.yaml values, but extract provider info if available
        provider = config.get('provider', 'NTNU')
        model = config.get('model', 'moonshotai/Kimi-K2.5')
    
    system_message = (
        f"You are an expert in NREL 5 MW wind turbine operations and control using {provider} AI. "
        "You have deep knowledge of turbine specifications, operating regions, "
        "yaw control, pitch control, and power optimization strategies. "
        "Provide detailed, technical recommendations based on the given wind conditions."
    )
    
    prompt = (
        f"Given the following wind conditions:\n"
        f"- Wind Speed: {wind_speed:.2f} m/s\n"
        f"- Wind Direction: {wind_direction:.1f}°\n\n"
        f"For an NREL 5 MW Reference Wind Turbine with the following specifications:\n"
        f"- Rated Power: 5.0 MW\n"
        f"- Rotor Diameter: 126 m\n"
        f"- Hub Height: 90 m\n"
        f"- Cut-in Wind Speed: 3.0 m/s\n"
        f"- Rated Wind Speed: 11.4 m/s\n"
        f"- Cut-out Wind Speed: 25.0 m/s\n\n"
        f"Please suggest:\n"
        f"1. The optimal yaw angle (nacelle direction)\n"
        f"2. The optimal pitch angle\n"
        f"3. The expected operating region (cut-in, partial load, rated, cut-out)\n"
        f"4. Detailed explanation of why these settings are chosen\n\n"
        f"Format your response clearly with labeled sections."
    )
    
    try:
        # Use unified LLM function - no need for manual API configuration
        llm_response = query_unified_llm(
            prompt=prompt,
            system_message=system_message,
            temperature=config.get('temperature', 0.1),
            max_tokens=config.get('max_tokens', 200000),
            timeout=config.get('timeout', 1000.0)
        )
        
        # Add provider info to response for debugging
        provider = GlobalLLMConfig.get_provider()
        model = GlobalLLMConfig.get_model()
        if not llm_response.startswith("Error"):
            llm_response = f"**[Using {provider} - {model}]**\n\n{llm_response}"
            
    except Exception as e:
        llm_response = f"Error connecting to {provider} LLM service: {str(e)}\n\nPlease check your API configuration and network connection."
    
    return {
        "llm_response": llm_response,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "config": config
    }


# =============================================================================
# Yaw Misalignment <-> Nacelle Direction Conversion Functions
# =============================================================================
def yaw_misalignment_to_nacelle_direction(yaw_misalignment: float) -> float:
    """
    Convert yaw misalignment angle to nacelle direction for ML model input.
    
    The ML models are trained with:
    - 270° nacelle direction = 0° yaw misalignment (aligned with wind)
    - 285° nacelle direction = 15° yaw misalignment
    
    Parameters:
    -----------
    yaw_misalignment : float
        Yaw misalignment angle in degrees (0 to 15)
    
    Returns:
    --------
    float
        Nacelle direction in degrees (270 to 285)
    """
    return 270.0 + yaw_misalignment


def nacelle_direction_to_yaw_misalignment(nacelle_direction: float) -> float:
    """
    Convert nacelle direction to yaw misalignment angle.
    
    Parameters:
    -----------
    nacelle_direction : float
        Nacelle direction in degrees (270 to 285)
    
    Returns:
    --------
    float
        Yaw misalignment angle in degrees (0 to 15)
    """
    return nacelle_direction - 270.0


# =============================================================================
# Two-Turbine Wake Steering Optimizer (Gradient-Based with PyTorch AD)
# =============================================================================
# Differentiable ML Surrogate for True AD through ML Models
# =============================================================================
class DifferentiablePowerSurrogate(torch.nn.Module):
    """
    A differentiable surrogate model for power prediction.
    
    This model fits a polynomial/RBF approximation to the GP model's predictions,
    enabling true automatic differentiation through the ML model.
    """
    def __init__(self, power_agent, yaw_range=(270.0, 285.0), n_fit_points=16):
        super().__init__()
        
        # Sample the GP model at multiple yaw angles
        yaw_samples = np.linspace(yaw_range[0], yaw_range[1], n_fit_points)
        power_samples = []
        
        for yaw in yaw_samples:
            result = power_agent.predict(yaw_angle=yaw, n_time_points=50, return_samples=False)
            power_samples.append(extract_settled_power(result['power_mean_MW']))
        
        self.yaw_samples = torch.tensor(yaw_samples, dtype=torch.float64)
        self.power_samples = torch.tensor(power_samples, dtype=torch.float64)
        
        # Fit a polynomial (degree 3) to the GP predictions
        # P(γ) = a₀ + a₁γ + a₂γ² + a₃γ³
        # Normalize yaw to [0, 1] for numerical stability
        self.yaw_min = yaw_range[0]
        self.yaw_max = yaw_range[1]
        
        yaw_norm = (self.yaw_samples - self.yaw_min) / (self.yaw_max - self.yaw_min)
        
        # Solve for polynomial coefficients using least squares
        A = torch.stack([torch.ones_like(yaw_norm), yaw_norm, yaw_norm**2, yaw_norm**3], dim=1)
        self.coeffs = torch.linalg.lstsq(A, self.power_samples).solution
        
    def forward(self, nacelle_direction):
        """
        Compute power for a given nacelle direction (differentiable).
        
        Args:
            nacelle_direction: torch.Tensor, nacelle direction in degrees (270-285)
            
        Returns:
            power: torch.Tensor, predicted power in MW
        """
        # Normalize input
        yaw_norm = (nacelle_direction - self.yaw_min) / (self.yaw_max - self.yaw_min)
        
        # Polynomial evaluation (differentiable)
        power = (self.coeffs[0] + 
                 self.coeffs[1] * yaw_norm + 
                 self.coeffs[2] * yaw_norm**2 + 
                 self.coeffs[3] * yaw_norm**3)
        
        return power


class DifferentiableWakeSurrogate(torch.nn.Module):
    """
    A differentiable surrogate model for wake deficit prediction.
    
    This model fits an approximation to the TT-OpInf model's wake deficit,
    enabling true automatic differentiation through the ML model.
    """
    def __init__(self, wake_agent, yaw_range=(270.0, 285.0), n_fit_points=8, 
                 freestream_velocity=8.5):
        super().__init__()
        
        self.freestream = freestream_velocity
        
        # Sample the TT-OpInf model at multiple yaw angles
        yaw_samples = np.linspace(yaw_range[0], yaw_range[1], n_fit_points)
        deficit_samples = []
        
        for yaw in yaw_samples:
            predictions, _ = wake_agent.predict(
                yaw_angle=yaw, n_timesteps=30, export_vtk=False, verbose=False
            )
            velocity_mag = np.linalg.norm(predictions, axis=2)
            mean_velocity = np.mean(velocity_mag)
            deficit = 1.0 - (mean_velocity / freestream_velocity)
            deficit_samples.append(np.clip(deficit, 0, 0.5))
        
        self.yaw_samples = torch.tensor(yaw_samples, dtype=torch.float64)
        self.deficit_samples = torch.tensor(deficit_samples, dtype=torch.float64)
        
        # Fit a polynomial to the wake deficit
        self.yaw_min = yaw_range[0]
        self.yaw_max = yaw_range[1]
        
        yaw_norm = (self.yaw_samples - self.yaw_min) / (self.yaw_max - self.yaw_min)
        
        # Quadratic fit for wake deficit
        A = torch.stack([torch.ones_like(yaw_norm), yaw_norm, yaw_norm**2], dim=1)
        self.coeffs = torch.linalg.lstsq(A, self.deficit_samples).solution
        
    def forward(self, nacelle_direction):
        """
        Compute wake deficit for a given nacelle direction (differentiable).
        
        Args:
            nacelle_direction: torch.Tensor, nacelle direction in degrees (270-285)
            
        Returns:
            deficit: torch.Tensor, wake deficit (0 to 0.5)
        """
        yaw_norm = (nacelle_direction - self.yaw_min) / (self.yaw_max - self.yaw_min)
        
        deficit = (self.coeffs[0] + 
                   self.coeffs[1] * yaw_norm + 
                   self.coeffs[2] * yaw_norm**2)
        
        return torch.clamp(deficit, 0.0, 0.5)


class BayesianTransferOptimizer:
    """
    Bayesian Optimization with Transfer Learning for yaw angle optimization.
    
    This optimizer treats the pre-trained GPR model's posterior mean as a base function,
    then trains a residual GP to capture correction terms. This is a common approach
    in transfer learning for surrogate modeling.
    
    Uses Expected Improvement (EI) acquisition function with:
    - Matérn 5/2 kernel with ARD length-scales
    - Multi-start L-BFGS-B to optimize EI
    - Constraint: yaw angles between 0-15 degrees
    """
    
    def __init__(self, base_gp_agent, wake_agent, turbine_spacing_D=7.0,
                 n_initial_samples=5, xi_lengthscale_range=(0.01, 0.05),
                 n_timesteps=50, freestream_velocity=8.5, lateral_offset_D=0.0):
        """
        Initialize Bayesian Transfer Optimizer.

        Parameters:
        -----------
        base_gp_agent : RotorPowerAgent
            Pre-trained GP model for power prediction
        wake_agent : WakeFlowAgent
            Wake flow prediction agent
        turbine_spacing_D : float
            Streamwise turbine spacing in rotor diameters
        n_initial_samples : int
            Number of initial random samples
        xi_lengthscale_range : tuple
            Range for ARD length-scale parameter (min, max)
        n_timesteps : int
            Number of timesteps for predictions
        freestream_velocity : float
            Freestream wind velocity (m/s)
        lateral_offset_D : float
            Cross-wind lateral offset of downstream turbine in rotor diameters (default: 0.0)
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF

        self.base_gp_agent = base_gp_agent
        self.wake_agent = wake_agent
        self.turbine_spacing_D = turbine_spacing_D
        self.lateral_offset_D = lateral_offset_D
        self.n_initial_samples = n_initial_samples
        self.n_timesteps = n_timesteps
        self.freestream_velocity = freestream_velocity
        
        # Initialize residual GP with Matérn 5/2 kernel (twice differentiable)
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 Matern(length_scale=0.03, length_scale_bounds=xi_lengthscale_range, nu=2.5)
        
        self.residual_gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        # Storage for observations
        self.X_obs = []
        self.y_obs = []
        self.y_base_obs = []
        
    def _generate_initial_observations(self, yaw_range=(0, 15)):
        """
        Generate initial observations by sampling base GP with synthetic noise.
        
        This simulates actual observations and allows us to compute residuals.
        """
        # Spacing-dependent seed so different turbine pairs explore different initial yaw angles
        np.random.seed(int(hash((self.turbine_spacing_D, self.lateral_offset_D)) % 2**31))

        # Sample random yaw angles in the range
        yaw_samples = np.random.uniform(yaw_range[0], yaw_range[1], self.n_initial_samples)
        
        X_init = []
        y_init = []
        y_base_init = []
        
        for yaw_misalign in yaw_samples:
            # Get base GP prediction for farm power
            farm_power_base = self._evaluate_farm_power_base(yaw_misalign)
            
            # Use actual farm power without synthetic noise — synthetic noise
            # causes the residual GP to train on pure noise, masking real differences
            farm_power_observed = farm_power_base
            
            X_init.append([yaw_misalign])
            y_init.append(farm_power_observed)
            y_base_init.append(farm_power_base)
        
        self.X_obs = np.array(X_init)
        self.y_obs = np.array(y_init)
        self.y_base_obs = np.array(y_base_init)
        
        # Compute residuals
        y_residuals = self.y_obs - self.y_base_obs
        
        return self.X_obs, y_residuals
    
    def _evaluate_farm_power_base(self, yaw_misalignment):
        """
        Evaluate total farm power using base GP model (no residual correction).

        Parameters:
        -----------
        yaw_misalignment : float
            Upstream turbine yaw misalignment (degrees)

        Returns:
        --------
        float
            Total farm power (MW)
        """
        # Calculate downstream yaw based on wake deflection physics
        downstream_misalign = calculate_wake_deflected_downstream_yaw(
            yaw_misalignment, self.turbine_spacing_D
        )

        # Convert to nacelle directions
        upstream_nacelle = 270.0 + yaw_misalignment
        downstream_nacelle = 270.0 + downstream_misalign

        # Get upstream power from base GP
        upstream_result = self.base_gp_agent.predict(
            yaw_angle=upstream_nacelle, n_time_points=self.n_timesteps
        )
        upstream_power = extract_settled_power(upstream_result['power_mean_MW'])

        # Distance-dependent wake deficit using Bastankhah & Porté-Agel (2016) model
        # Accounts for actual streamwise spacing and cross-wind lateral offset
        effective_deficit = calculate_wake_deficit(
            turbine_spacing_D=self.turbine_spacing_D,
            upstream_yaw_deg=yaw_misalignment,
            thrust_coefficient=0.8,
            wake_expansion_coeff=0.022,
            lateral_offset_D=self.lateral_offset_D
        )

        # Get downstream power
        downstream_result = self.base_gp_agent.predict(
            yaw_angle=downstream_nacelle, n_time_points=self.n_timesteps
        )
        downstream_power_base = extract_settled_power(downstream_result['power_mean_MW'])
        downstream_power = downstream_power_base * (1 - effective_deficit)**3

        total_power = upstream_power + downstream_power
        return total_power
    
    def fit_residual_gp(self, X, y_residuals):
        """
        Fit the residual GP on observed residuals.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, 1)
            Yaw misalignment angles
        y_residuals : ndarray, shape (n_samples,)
            Residuals (observed - base GP predictions)
        """
        self.residual_gp.fit(X, y_residuals)
    
    def predict_combined(self, yaw_angle):
        """
        Predict farm power using base GP + residual GP correction.
        
        Parameters:
        -----------
        yaw_angle : float or ndarray
            Yaw misalignment angle(s)
        
        Returns:
        --------
        tuple
            (mean_power, std_power) - Mean and standard deviation of predicted power
        """
        if np.isscalar(yaw_angle):
            X_pred = np.array([[yaw_angle]])
        else:
            X_pred = np.array(yaw_angle).reshape(-1, 1)
        
        # Get base GP prediction
        y_base = np.array([self._evaluate_farm_power_base(x[0]) for x in X_pred])
        
        # Get residual GP prediction
        y_residual_mean, y_residual_std = self.residual_gp.predict(X_pred, return_std=True)
        
        # Combine: total = base + residual
        y_total_mean = y_base + y_residual_mean
        y_total_std = y_residual_std  # Uncertainty from residual GP
        
        if np.isscalar(yaw_angle):
            return y_total_mean[0], y_total_std[0]
        else:
            return y_total_mean, y_total_std
    
    def expected_improvement(self, yaw_angle, f_best, xi=0.01):
        """
        Compute Expected Improvement acquisition function.
        
        Parameters:
        -----------
        yaw_angle : float or ndarray
            Yaw misalignment angle(s)
        f_best : float
            Best observed function value so far
        xi : float
            Exploration-exploitation trade-off parameter
        
        Returns:
        --------
        float or ndarray
            Expected Improvement value(s)
        """
        from scipy.stats import norm
        
        mu, sigma = self.predict_combined(yaw_angle)
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        # Compute EI
        Z = (mu - f_best - xi) / sigma
        ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei
    
    def optimize_acquisition_multistart(self, f_best, n_starts=10, bounds=(0, 15), xi=0.01):
        """
        Optimize Expected Improvement using multi-start L-BFGS-B.
        
        Parameters:
        -----------
        f_best : float
            Best observed function value
        n_starts : int
            Number of random restarts
        bounds : tuple
            (min, max) bounds for yaw angle
        xi : float
            EI exploration parameter
        
        Returns:
        --------
        float
            Optimal yaw angle that maximizes EI
        """
        from scipy.optimize import minimize
        
        best_x = None
        best_ei = -np.inf
        
        # Random starting points
        np.random.seed(None)  # Use current time for randomness
        x0_samples = np.random.uniform(bounds[0], bounds[1], n_starts)
        
        for x0 in x0_samples:
            # Minimize negative EI (maximize EI)
            result = minimize(
                fun=lambda x: -self.expected_improvement(x, f_best, xi),
                x0=[x0],
                method='L-BFGS-B',
                bounds=[bounds]
            )
            
            if result.success:
                ei_value = -result.fun
                if ei_value > best_ei:
                    best_ei = ei_value
                    best_x = result.x[0]
        
        # Fallback if no optimization succeeded
        if best_x is None:
            best_x = x0_samples[0]
        
        return np.clip(best_x, bounds[0], bounds[1])
    
    def optimize(self, n_iterations=15, verbose=False, xi_range=(0.01, 0.05)):
        """
        Run Bayesian Optimization loop.
        
        Parameters:
        -----------
        n_iterations : int
            Number of BO iterations
        verbose : bool
            Print progress
        xi_range : tuple
            Range for exploration parameter xi
        
        Returns:
        --------
        dict
            Optimization results
        """
        # Generate initial observations
        X_init, y_residuals = self._generate_initial_observations()
        
        # Fit residual GP on initial data
        self.fit_residual_gp(X_init, y_residuals)
        
        # Track best result
        best_power = np.max(self.y_obs)
        best_idx = np.argmax(self.y_obs)
        best_yaw = self.X_obs[best_idx, 0]
        
        if verbose:
            print(f"  Initial best: yaw={best_yaw:.2f}°, power={best_power:.4f} MW")
        
        results_history = []
        
        # Bayesian Optimization loop
        for iteration in range(n_iterations):
            # Adaptive xi: higher exploration early, more exploitation later
            xi = xi_range[0] + (xi_range[1] - xi_range[0]) * (1 - iteration / n_iterations)
            
            # Optimize acquisition function to find next point
            next_yaw = self.optimize_acquisition_multistart(
                f_best=best_power, n_starts=10, bounds=(0, 15), xi=xi
            )
            
            # Evaluate farm power at next point
            farm_power_base = self._evaluate_farm_power_base(next_yaw)
            farm_power_observed = farm_power_base

            # Compute residual
            residual = farm_power_observed - farm_power_base
            
            # Add to observations
            self.X_obs = np.vstack([self.X_obs, [[next_yaw]]])
            self.y_obs = np.append(self.y_obs, farm_power_observed)
            self.y_base_obs = np.append(self.y_base_obs, farm_power_base)
            
            # Update residual GP with new data
            y_residuals_all = self.y_obs - self.y_base_obs
            self.fit_residual_gp(self.X_obs, y_residuals_all)
            
            # Update best
            if farm_power_observed > best_power:
                best_power = farm_power_observed
                best_yaw = next_yaw
                if verbose:
                    print(f"  Iter {iteration+1:2d}: New best! yaw={best_yaw:.2f}°, power={best_power:.4f} MW")
            elif verbose and iteration % 5 == 0:
                print(f"  Iter {iteration+1:2d}: yaw={next_yaw:.2f}°, power={farm_power_observed:.4f} MW, EI={xi:.4f}")
            
            # Compute EI for early stopping
            ei_at_best = self.expected_improvement(best_yaw, best_power, xi)
            if ei_at_best < 1e-4:
                if verbose:
                    print(f"  Converged at iteration {iteration+1} (EI < 1e-4)")
                break
            
            results_history.append({
                'iteration': iteration + 1,
                'yaw': next_yaw,
                'power': farm_power_observed,
                'xi': xi,
                'ei': ei_at_best
            })
        
        # Calculate downstream yaw for optimal upstream yaw
        optimal_downstream_yaw = calculate_wake_deflected_downstream_yaw(
            best_yaw, self.turbine_spacing_D
        )
        
        # Evaluate at optimal point
        optimal_power = self._evaluate_farm_power_base(best_yaw)
        
        # Evaluate baseline (0° yaw)
        baseline_power = self._evaluate_farm_power_base(0.0)
        
        power_gain = optimal_power - baseline_power
        power_gain_pct = (power_gain / baseline_power) * 100 if baseline_power > 0 else 0
        
        return {
            'optimal_upstream_misalignment': best_yaw,
            'optimal_downstream_misalignment': optimal_downstream_yaw,
            'optimal_total_power': optimal_power,
            'baseline_total_power': baseline_power,
            'power_gain_MW': power_gain,
            'power_gain_percent': power_gain_pct,
            'n_evaluations': len(self.X_obs),
            'results_history': results_history,
            'optimization_method': 'Bayesian Optimization (Transfer Learning)'
        }


# =============================================================================
# Multi-Pair Turbine Optimizer (Optimizes N pairs individually)
# =============================================================================
def optimize_multiple_turbine_pairs(turbine_pairs_data, power_agent, wake_agent, 
                                    optimization_method='analytical_physics', 
                                    n_timesteps=50, max_pairs=4, verbose=False):
    """
    Optimize multiple turbine pairs individually.
    
    Parameters:
    -----------
    turbine_pairs_data : dict
        Results from Agent 2C/2D containing turbine pairs information
    power_agent : RotorPowerAgent
        Power prediction agent
    wake_agent : WakeFlowAgent
        Wake flow prediction agent
    optimization_method : str
        Optimization method ('analytical_physics', 'ml_surrogate', 'grid_search')
    n_timesteps : int
        Number of timesteps for predictions
    max_pairs : int
        Maximum number of pairs to optimize
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict
        Results containing optimization results for all pairs
    """
    if not turbine_pairs_data or 'turbine_pairs' not in turbine_pairs_data:
        return {
            'status': 'no_pairs',
            'message': 'No turbine pairs provided for optimization',
            'optimization_results': [],
            'total_power_gain': 0.0
        }
    
    turbine_pairs = turbine_pairs_data['turbine_pairs']
    turbine_chains = turbine_pairs_data.get('turbine_chains', [])
    
    # DEBUG: Print turbine pairs structure
    if verbose or True:  # Always print for debugging
        print(f"\n[DEBUG] optimize_multiple_turbine_pairs called with {len(turbine_pairs)} pairs", file=sys.stderr)
        if turbine_pairs:
            print(f"[DEBUG] First pair structure: {turbine_pairs[0]}", file=sys.stderr)
            print(f"[DEBUG] First pair keys: {list(turbine_pairs[0].keys())}", file=sys.stderr)
            if 'distance_rotor_diameters' in turbine_pairs[0]:
                print(f"[DEBUG] First pair has distance_rotor_diameters: {turbine_pairs[0]['distance_rotor_diameters']}", file=sys.stderr)
            else:
                print(f"[DEBUG] WARNING: First pair MISSING distance_rotor_diameters key!", file=sys.stderr)
    
    if not turbine_pairs:
        return {
            'status': 'no_pairs',
            'message': 'No turbine pairs found',
            'optimization_results': [],
            'total_power_gain': 0.0
        }
    
    # Limit to max pairs
    pairs_to_optimize = turbine_pairs[:max_pairs]
    
    optimization_results = []
    total_power_gain = 0.0
    
    for idx, pair in enumerate(pairs_to_optimize):
        if verbose:
            print(f"\nOptimizing Pair {idx+1}/{len(pairs_to_optimize)}...")
        
        # Extract turbine IDs (both Agent 2C and 2D use 'upstream_turbine'/'downstream_turbine')
        upstream_id = pair.get('upstream_turbine')
        downstream_id = pair.get('downstream_turbine')
        
        if upstream_id is None or downstream_id is None:
            if verbose:
                print(f"  Skipping pair {idx+1}: Missing turbine IDs")
            continue
        
        # Always treat as simple two-turbine pair (no chain merging)
        turbine_list = [upstream_id, downstream_id]

        # Extract spatial data from pair (from Agent 2D wake analysis)
        # downstream_distance_D = streamwise (wind-aligned) distance in rotor diameters
        # lateral_offset_D = cross-wind lateral offset in rotor diameters
        # distance_rotor_diameters = Euclidean distance (kept for display)
        euclidean_D = pair.get('distance_rotor_diameters', 7.0)
        downstream_distance_D = pair.get('downstream_distance_D', euclidean_D)  # Fallback to Euclidean
        lateral_offset_D = pair.get('lateral_offset_D', 0.0)  # Fallback to on-centerline

        # DEBUG: Print spatial data extraction
        print(f"\n[DEBUG] Pair {idx+1}: IDs={upstream_id}->{downstream_id}", file=sys.stderr)
        print(f"[DEBUG] Pair {idx+1}: Keys in pair dict = {list(pair.keys())}", file=sys.stderr)
        print(f"[DEBUG] Pair {idx+1}: Euclidean={euclidean_D:.2f}D, Streamwise={downstream_distance_D:.2f}D, Lateral={lateral_offset_D:.2f}D", file=sys.stderr)

        if verbose:
            print(f"  Turbine {upstream_id} -> {downstream_id}, Streamwise: {downstream_distance_D:.2f}D, Lateral offset: {lateral_offset_D:.2f}D")

        # Check baseline wake deficit at 0° yaw to determine if optimization is needed
        baseline_deficit = calculate_wake_deficit(
            turbine_spacing_D=downstream_distance_D,
            upstream_yaw_deg=0.0,
            lateral_offset_D=lateral_offset_D
        )

        # Classify wake overlap
        k_star = 0.022
        epsilon = 0.2
        sigma_D = k_star * downstream_distance_D + epsilon  # Wake width at downstream position
        rotor_radius_D = 0.5  # Half rotor diameter
        if lateral_offset_D > sigma_D * 3.0 + rotor_radius_D:
            overlap_class = 'none'
        elif lateral_offset_D > sigma_D * 1.5:
            overlap_class = 'partial'
        else:
            overlap_class = 'full'

        print(f"[DEBUG] Pair {idx+1}: Baseline deficit={baseline_deficit:.4f}, Wake σ={sigma_D:.2f}D, Overlap={overlap_class}", file=sys.stderr)

        # Short-circuit: if baseline deficit is negligible, no wake steering needed
        if baseline_deficit < 0.01:
            if verbose:
                print(f"  ⏭️  Skipping pair {idx+1}: Baseline wake deficit {baseline_deficit:.4f} < 1% — no wake steering needed")
            print(f"[DEBUG] Pair {idx+1}: SKIPPED — negligible wake influence (deficit={baseline_deficit:.4f})", file=sys.stderr)

            # Return zero-gain result for this pair
            optimization_results.append({
                'pair_index': idx + 1,
                'turbine_ids': turbine_list,
                'upstream_id': upstream_id,
                'downstream_id': downstream_id,
                'upstream_yaw': 0.0,
                'downstream_yaw': 0.0,
                'turbine_spacing_D': euclidean_D,
                'downstream_distance_D': downstream_distance_D,
                'lateral_offset_D': lateral_offset_D,
                'overlap_class': overlap_class,
                'baseline_deficit': baseline_deficit,
                'optimal_yaw_angles': {upstream_id: 0.0, downstream_id: 0.0},
                'power_gain_MW': 0.0,
                'power_gain_percent': 0.0,
                'optimization_method': f'{optimization_method} (skipped: no wake influence)',
                'baseline_power': 0.0,
                'optimized_power': 0.0,
                'sensitivity_figure': None
            })
            continue

        try:
            # Optimize this pair using streamwise distance and lateral offset
            opt_result = optimize_two_turbine_farm(
                power_agent=power_agent,
                wake_agent=wake_agent,
                turbine_spacing_D=downstream_distance_D,  # Streamwise distance (not Euclidean)
                lateral_offset_D=lateral_offset_D,  # Cross-wind offset
                n_timesteps=n_timesteps,
                verbose=verbose,
                optimization_method=optimization_method
            )
            
            # Build result for this pair
            # In wake steering: only upstream turbine gets yaw misalignment, downstream stays at 0°
            upstream_yaw = opt_result.get('optimal_upstream_misalignment', 0.0)
            power_gain = opt_result.get('power_gain_MW', 0.0)

            # Guard: if optimization produces negative gain, revert to baseline (0° yaw)
            if power_gain < 0:
                print(f"[DEBUG] Pair {idx+1}: Negative gain ({power_gain:.4f} MW) — reverting to baseline (0° yaw)", file=sys.stderr)
                upstream_yaw = 0.0
                opt_result['optimal_upstream_misalignment'] = 0.0
                opt_result['optimal_downstream_misalignment'] = 0.0
                opt_result['optimal_upstream_nacelle'] = 270.0
                opt_result['optimal_downstream_nacelle'] = 270.0
                opt_result['optimal_upstream_power'] = opt_result.get('baseline_upstream_power', 0.0)
                opt_result['optimal_downstream_power'] = opt_result.get('baseline_downstream_power', 0.0)
                opt_result['optimal_total_power'] = opt_result.get('baseline_total_power', 0.0)
                opt_result['power_gain_MW'] = 0.0
                opt_result['power_gain_percent'] = 0.0
                opt_result['optimization_method'] = opt_result.get('optimization_method', optimization_method) + ' (reverted: negative gain)'

            # DEBUG: Print optimization results
            print(f"[DEBUG] Pair {idx+1}: Optimization returned keys = {list(opt_result.keys())}", file=sys.stderr)
            print(f"[DEBUG] Pair {idx+1}: optimal_upstream_misalignment = {upstream_yaw}", file=sys.stderr)
            print(f"[DEBUG] Pair {idx+1}: optimal_total_power = {opt_result.get('optimal_total_power', 'N/A')}", file=sys.stderr)
            print(f"[DEBUG] Pair {idx+1}: power_gain_MW = {opt_result.get('power_gain_MW', 'N/A')}", file=sys.stderr)
            print(f"[DEBUG] Pair {idx+1}: optimization_method = {optimization_method}", file=sys.stderr)
            if upstream_yaw == 0.0:
                print(f"[DEBUG] Pair {idx+1}: WARNING - Upstream yaw is ZERO! Check if optimization actually ran.", file=sys.stderr)
                if 'error' in opt_result:
                    print(f"[DEBUG] Pair {idx+1}: ERROR in result: {opt_result['error']}", file=sys.stderr)
            
            pair_result = {
                'pair_index': idx + 1,
                'turbine_ids': turbine_list,
                'upstream_id': upstream_id,
                'downstream_id': downstream_id,
                'upstream_yaw': upstream_yaw,
                'downstream_yaw': 0.0,  # Always 0° for wake steering
                'turbine_spacing_D': euclidean_D,  # Euclidean spacing for display
                'downstream_distance_D': downstream_distance_D,  # Streamwise distance used
                'lateral_offset_D': lateral_offset_D,  # Cross-wind offset
                'overlap_class': overlap_class,  # 'full', 'partial', or 'none'
                'baseline_deficit': baseline_deficit,
                'optimal_yaw_angles': {
                    upstream_id: upstream_yaw,
                    downstream_id: 0.0  # Downstream stays aligned with wind
                },
                'power_gain_MW': opt_result.get('power_gain_MW', 0.0),
                'power_gain_percent': opt_result.get('power_gain_percent', 0.0),
                'optimization_method': opt_result.get('optimization_method', optimization_method),
                'baseline_power': opt_result.get('baseline_total_power', 0.0),
                'optimized_power': opt_result.get('optimal_total_power', 0.0),
                'upstream_power_baseline': opt_result.get('baseline_upstream_power', 0.0),
                'upstream_power_optimized': opt_result.get('optimal_upstream_power', 0.0),
                'downstream_power_baseline': opt_result.get('baseline_downstream_power', 0.0),
                'downstream_power_optimized': opt_result.get('optimal_downstream_power', 0.0),
                'sensitivity_figure': opt_result.get('sensitivity_figure', None)  # For ML wake extraction visualization
            }
            
            # Simple pair - no need to assign additional turbines
            # turbine_list always has exactly 2 turbines: [upstream_id, downstream_id]
            
            optimization_results.append(pair_result)
            total_power_gain += pair_result['power_gain_MW']
            
        except Exception as e:
            if verbose:
                print(f"Error optimizing pair {idx+1}: {e}")
            # Add error result
            optimization_results.append({
                'pair_index': idx + 1,
                'turbine_ids': turbine_list,
                'upstream_id': upstream_id,
                'downstream_id': downstream_id,
                'optimal_yaw_angles': {},
                'power_gain_MW': 0.0,
                'power_gain_percent': 0.0,
                'error': str(e),
                'optimization_method': optimization_method
            })
    
    return {
        'status': 'success' if optimization_results else 'failed',
        'message': f'Optimized {len(optimization_results)} turbine pairs',
        'optimization_results': optimization_results,
        'total_power_gain': total_power_gain,
        'optimization_method': optimization_method,
        'num_pairs_optimized': len(optimization_results)
    }


# =============================================================================
# ML-Based Wake Analysis Functions (User's Expert Approach)
# =============================================================================

def extract_vertical_profile_from_wake(wake_predictions, grid, downstream_distance_m, 
                                        vertical_range_m=300, hub_height_m=90):
    """
    Extract vertical velocity profile from ML wake predictions at specified downstream distance.
    
    Based on user's expert approach: Extract velocity profile along vertical line at given
    downstream distance to analyze wake deficit and lateral migration.
    
    The ML wake model provides full 3D velocity fields. This function samples a vertical
    line at a specific downstream distance (e.g., 0m at turbine, 1700m far downstream)
    to extract the velocity profile for wake analysis.
    
    Parameters:
    -----------
    wake_predictions : np.ndarray
        Shape (n_timesteps, n_space, 3) - velocity field from TT-OpInf model
    grid : pyvista.UnstructuredGrid
        Spatial grid with point coordinates
    downstream_distance_m : float
        Distance downstream from turbine to extract profile (e.g., 0m, 1700m)
    vertical_range_m : float
        Vertical extent above/below hub (default: 300m = ±300m total 600m)
    hub_height_m : float
        Hub height of turbine (default: 90m for NREL 5MW)
    
    Returns:
    --------
    dict with keys:
        'z_coords': Vertical coordinates (m)
        'velocities': Time-averaged velocity magnitudes at each z
        'lateral_positions': Corresponding y-coordinates (lateral positions)
        'x_coord': Actual downstream x-coordinate sampled
        'n_points': Number of grid points in profile
    """
    import numpy as np
    import sys
    
    # Get grid coordinates
    points = grid.points  # Shape: (n_points, 3) with columns [x, y, z]
    
    print(f"[DEBUG extract_profile] Searching for points at x={downstream_distance_m:.1f}m, grid has {len(points)} total points", file=sys.stderr)
    print(f"[DEBUG extract_profile] Grid x-range: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]", file=sys.stderr)
    
    # Find points near the specified downstream distance
    # Allow ±10m tolerance for finding points
    x_tolerance = 10.0
    mask_x = np.abs(points[:, 0] - downstream_distance_m) < x_tolerance
    
    if not np.any(mask_x):
        # No points found at this distance
        print(f"[DEBUG extract_profile] ❌ No grid points found at x={downstream_distance_m:.1f}m ±{x_tolerance}m", file=sys.stderr)
        return None
    
    # Filter to vertical range around hub height
    z_min = hub_height_m - vertical_range_m
    z_max = hub_height_m + vertical_range_m
    mask_z = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    
    # Combined mask
    mask = mask_x & mask_z
    
    print(f"[DEBUG extract_profile] Found {np.sum(mask_x)} points at x±{x_tolerance}m, {np.sum(mask)} after z-filtering [{z_min:.1f}, {z_max:.1f}]", file=sys.stderr)
    
    if np.sum(mask) < 5:
        # Too few points
        print(f"[DEBUG extract_profile] ❌ Only {np.sum(mask)} points found (need ≥5)", file=sys.stderr)
        return None
    
    # Extract points and velocities at this location
    selected_points = points[mask]
    selected_indices = np.where(mask)[0]
    
    # Time-average velocities
    vel_at_location = wake_predictions[:, selected_indices, :]  # (n_time, n_selected, 3)
    vel_mag = np.linalg.norm(vel_at_location, axis=2)  # (n_time, n_selected)
    vel_avg = np.mean(vel_mag, axis=0)  # (n_selected,) - time-averaged
    
    print(f"[DEBUG extract_profile] ✅ Successfully extracted {len(vel_avg)} points at x={np.mean(selected_points[:, 0]):.1f}m", file=sys.stderr)
    
    return {
        'z_coords': selected_points[:, 2],  # Vertical positions
        'velocities': vel_avg,  # Time-averaged velocity magnitudes
        'lateral_positions': selected_points[:, 1],  # Lateral (y) positions
        'x_coord': np.mean(selected_points[:, 0]),  # Actual x sampled
        'n_points': len(vel_avg)
    }


def calculate_wake_centerline_and_deficit(profile_start, profile_end, freestream_velocity=8.5):
    """
    Calculate wake deficit and lateral migration from two vertical profiles.
    
    User's expert approach: Track the "eye" of the wake (minimum velocity location) at both
    start and end profiles. The lateral displacement of the wake eye reveals how the wake
    centerline has migrated due to yaw misalignment.
    
    CRITICAL: Wake eye = location of MINIMUM velocity in vertical profile.
    Lateral migration = change in y-position of the wake eye between start and end.
    
    Parameters:
    -----------
    profile_start : dict
        Vertical profile at turbine location (x=0m or near-wake)
    profile_end : dict
        Vertical profile at reference downstream location (e.g., x=1000m)
    freestream_velocity : float
        Freestream wind speed (m/s)
    
    Returns:
    --------
    dict with keys:
        'deficit_start': Maximum wake deficit at start location (0 to 1)
        'deficit_end': Maximum wake deficit at end location (0 to 1)
        'lateral_migration': Lateral displacement of wake eye (m) = y_eye_end - y_eye_start
        'y_eye_start': Y-position of wake eye at start (m)
        'y_eye_end': Y-position of wake eye at end (m)
        'z_eye_start': Z-position of wake eye at start (m)
        'z_eye_end': Z-position of wake eye at end (m)
        'migration_rate': Linear migration rate (m/m) = lateral_migration / x_end
    """
    import numpy as np
    import sys
    
    if profile_start is None or profile_end is None:
        return None
    
    # Find wake EYE (minimum velocity) at START location
    idx_eye_start = np.argmin(profile_start['velocities'])
    y_eye_start = profile_start['lateral_positions'][idx_eye_start]
    z_eye_start = profile_start['z_coords'][idx_eye_start]
    vel_min_start = profile_start['velocities'][idx_eye_start]
    deficit_start = 1.0 - (vel_min_start / freestream_velocity)
    deficit_start = np.clip(deficit_start, 0, 1)
    
    # Find wake EYE (minimum velocity) at END location
    idx_eye_end = np.argmin(profile_end['velocities'])
    y_eye_end = profile_end['lateral_positions'][idx_eye_end]
    z_eye_end = profile_end['z_coords'][idx_eye_end]
    vel_min_end = profile_end['velocities'][idx_eye_end]
    deficit_end = 1.0 - (vel_min_end / freestream_velocity)
    deficit_end = np.clip(deficit_end, 0, 1)
    
    # Lateral migration = change in wake eye position
    lateral_migration = y_eye_end - y_eye_start
    
    # Migration rate for linear extrapolation
    x_end = profile_end['x_coord']
    migration_rate = lateral_migration / x_end if x_end > 0 else 0.0
    
    print(f"[DEBUG calc_wake_centerline] Wake eye at start: y={y_eye_start:.2f}m, z={z_eye_start:.1f}m, deficit={deficit_start:.3f}", file=sys.stderr)
    print(f"[DEBUG calc_wake_centerline] Wake eye at end: y={y_eye_end:.2f}m, z={z_eye_end:.1f}m, deficit={deficit_end:.3f}", file=sys.stderr)
    print(f"[DEBUG calc_wake_centerline] Lateral migration: {lateral_migration:.2f}m, rate: {migration_rate:.5f} m/m", file=sys.stderr)
    
    return {
        'deficit_start': deficit_start,
        'deficit_end': deficit_end,
        'lateral_migration': lateral_migration,
        'y_eye_start': y_eye_start,
        'y_eye_end': y_eye_end,
        'z_eye_start': z_eye_start,
        'z_eye_end': z_eye_end,
        'migration_rate': migration_rate,
        'x_end': x_end
    }


def extrapolate_wake_trajectory(wake_analysis, target_distance_m, reference_distance_m):
    """
    Extrapolate wake lateral position and deficit to arbitrary downstream distance.
    
    User's approach: Linear extrapolation using measured migration rate from wake eye tracking.
    Given wake eye positions at start (x=0) and reference distance (x=x_ref), extrapolate
    to actual turbine distance (x=x_target) using linear rate.
    
    Formula:
        migration_rate = (y_eye_end - y_eye_start) / x_ref  [m/m]
        y_at_target = y_eye_start + migration_rate × x_target
    
    Parameters:
    -----------
    wake_analysis : dict
        Result from calculate_wake_centerline_and_deficit() with keys:
        'y_eye_start', 'y_eye_end', 'lateral_migration', 'migration_rate'
    target_distance_m : float
        Distance where we want to predict wake (e.g., actual turbine spacing 2500m)
    reference_distance_m : float
        Distance where wake was measured (e.g., 1000m from grid)
    
    Returns:
    --------
    dict with keys:
        'lateral_migration_at_target': Absolute Y-position of wake eye at target (m)
        'deficit_at_target': Estimated wake deficit at target distance
        'extrapolation_ratio': target_distance / reference_distance
        'migration_rate': m/m
        'y_eye_start': Starting Y-position (m)
        'is_extrapolation': bool - True if target > reference
    """
    import numpy as np
    import sys
    
    if wake_analysis is None:
        return None
    
    # Get wake eye positions from analysis
    y_start = wake_analysis['y_eye_start']
    migration_rate = wake_analysis['migration_rate']
    
    # Linear extrapolation: y(x) = y_start + rate × x
    y_at_target = y_start + migration_rate * target_distance_m
    
    # Deficit also extrapolates (conservative linear estimate)
    extrapolation_ratio = target_distance_m / reference_distance_m
    deficit_target = wake_analysis['deficit_end'] * extrapolation_ratio
    deficit_target = np.clip(deficit_target, 0, 1)  # Physical limits
    
    print(f"[DEBUG extrapolate] Target distance: {target_distance_m:.1f}m, Reference: {reference_distance_m:.1f}m", file=sys.stderr)
    print(f"[DEBUG extrapolate] y_start={y_start:.2f}m, rate={migration_rate:.5f}m/m → y_target={y_at_target:.2f}m", file=sys.stderr)
    
    return {
        'lateral_migration_at_target': y_at_target,  # Absolute Y-position
        'deficit_at_target': deficit_target,
        'extrapolation_ratio': extrapolation_ratio,
        'migration_rate': migration_rate,
        'y_eye_start': y_start,
        'is_extrapolation': target_distance_m > reference_distance_m
    }


def check_wake_turbine_overlap(wake_at_distance, downstream_turbine_position, 
                                rotor_radius_m=63.0):
    """
    Check if wake overlaps with downstream turbine rotor.
    
    User's approach: Given known positions of both turbines and predicted wake
    trajectory, determine if wake will hit downstream turbine. This is a geometric
    calculation comparing wake centerline position to turbine position.
    
    Parameters:
    -----------
    wake_at_distance : dict
        Result from extrapolate_wake_trajectory() - wake position at downstream turbine
    downstream_turbine_position : dict
        {'x': downstream x-coord, 'y': downstream y-coord (lateral)}
    rotor_radius_m : float
        Rotor radius (default: 63m for NREL 5MW with D=126m)
    
    Returns:
    --------
    dict with keys:
        'overlaps': bool - True if wake hits turbine
        'lateral_clearance': float - Distance between wake center and turbine center (m)
        'overlap_fraction': float - Fraction of rotor disk in wake (0 to 1)
        'wake_center_y': Wake centerline y-position
        'turbine_center_y': Turbine centerline y-position
    """
    import numpy as np
    
    if wake_at_distance is None:
        # No wake data, assume overlap for safety
        return {'overlaps': True, 'lateral_clearance': 0, 'overlap_fraction': 1.0}
    
    # Lateral distance between wake centerline and turbine center
    wake_y = wake_at_distance['lateral_migration_at_target']
    turbine_y = downstream_turbine_position.get('y', 0)  # Assume 0 if not specified
    
    lateral_clearance = abs(wake_y - turbine_y)
    
    # Check if wake overlaps turbine rotor disk
    # Assume wake width ~ 1.5 * rotor_diameter (conservative estimate based on expansion)
    wake_half_width = 1.5 * rotor_radius_m  # 94.5m for NREL 5MW
    
    if lateral_clearance > (rotor_radius_m + wake_half_width):
        # Wake completely misses turbine
        overlaps = False
        overlap_fraction = 0.0
    elif lateral_clearance < (wake_half_width - rotor_radius_m):
        # Turbine completely in wake
        overlaps = True
        overlap_fraction = 1.0
    else:
        # Partial overlap - estimate fraction
        overlaps = True
        overlap_fraction = 1.0 - (lateral_clearance / (rotor_radius_m + wake_half_width))
        overlap_fraction = np.clip(overlap_fraction, 0, 1)
    
    return {
        'overlaps': overlaps,
        'lateral_clearance': lateral_clearance,
        'overlap_fraction': overlap_fraction,
        'wake_center_y': wake_y,
        'turbine_center_y': turbine_y
    }


def find_optimal_yaw_for_wake_avoidance(wake_agent, power_agent, P_base, turbine_spacing_D, rotor_diameter=126.0,
                                         downstream_turbine_position=None, 
                                         yaw_constraint=(0, 15), n_timesteps=50,
                                         reference_distance_m=None, verbose=False):
    """
    Find minimum yaw misalignment needed to steer wake away from downstream turbine.
    
    User's expert approach: Use ML wake model to find the yaw angle that causes
    sufficient lateral wake migration to miss the downstream turbine entirely or
    minimize overlap. Tests multiple yaw angles and returns the optimal one.
    
    Strategy:
    1. Test yaw angles from 0° to 15° (constraint)
    2. For each yaw, run ML wake model
    3. Extract wake profiles at 0m and 1700m
    4. Calculate lateral migration
    5. Extrapolate to actual turbine distance
    6. Check if wake misses downstream turbine
    7. Calculate power for upstream (ML model) and downstream (with wake deficit)
    8. Return minimum yaw where overlap < 10%
    
    Parameters:
    -----------
    wake_agent : WakeFlowAgent
        ML wake flow prediction agent (TT-OpInf model)
    power_agent : RotorPowerAgent
        ML power prediction agent (GP model)
    P_base : float
        Baseline power at 0° yaw (MW) from ML model
    turbine_spacing_D : float
        Actual turbine spacing in rotor diameters
    rotor_diameter : float
        Rotor diameter in meters (default: 126m for NREL 5MW)
    downstream_turbine_position : dict
        Position {'x': x_m, 'y': y_m (lateral offset)}
    yaw_constraint : tuple
        (min_yaw, max_yaw) in degrees (default: 0-15°)
    n_timesteps : int
        Timesteps for ML prediction
    reference_distance_m : float or None
        Distance to extract wake profile for analysis. If None, auto-detect from grid extent.
        Default: None (auto-detect to 80% of grid x_max or 1700m, whichever is smaller)
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict with keys:
        'optimal_yaw': float - Minimum yaw angle for wake avoidance (degrees)
        'wake_trajectory': list - Wake analysis at different yaw angles tested
        'avoidance_achieved': bool - True if wake can be steered away within constraint
        'turbine_spacing_m': Actual spacing in meters
        'turbine_spacing_D': Spacing in rotor diameters
        'reference_distance_m': Actual reference distance used (m)
    """
    import numpy as np
    import sys
    
    if downstream_turbine_position is None:
        downstream_turbine_position = {'x': turbine_spacing_D * rotor_diameter, 'y': 0}
    
    target_distance_m = turbine_spacing_D * rotor_diameter
    rotor_radius_m = rotor_diameter / 2.0
    
    # Auto-detect reference distance from grid extent if not specified
    if reference_distance_m is None:
        grid = wake_agent.grid
        if grid is not None:
            x_max = grid.points[:, 0].max()
            # Use 80% of grid extent for safety, capped at 1700m
            reference_distance_m = min(x_max * 0.8, 1700.0)
            print(f"[DEBUG find_optimal_yaw] Auto-detected reference distance: {reference_distance_m:.1f}m (grid extends to {x_max:.1f}m)", file=sys.stderr)
        else:
            reference_distance_m = 1000.0  # Conservative fallback
            print(f"[DEBUG find_optimal_yaw] Grid is None, using fallback reference distance: {reference_distance_m:.1f}m", file=sys.stderr)
    
    # Test yaw angles from minimum to maximum
    yaw_test_angles = np.linspace(yaw_constraint[0], yaw_constraint[1], 8)
    
    wake_trajectory = []
    
    # DEBUG: Initial grid check
    import sys
    print(f"\n[DEBUG find_optimal_yaw] Starting yaw tests at spacing={turbine_spacing_D:.2f}D ({target_distance_m:.1f}m)", file=sys.stderr)
    print(f"[DEBUG find_optimal_yaw] wake_agent.grid exists? {hasattr(wake_agent, 'grid')}", file=sys.stderr)
    if hasattr(wake_agent, 'grid'):
        print(f"[DEBUG find_optimal_yaw] wake_agent.grid is None? {wake_agent.grid is None}", file=sys.stderr)
    print(f"[DEBUG find_optimal_yaw] Testing {len(yaw_test_angles)} yaw angles: {yaw_test_angles}", file=sys.stderr)
    
    for yaw_deg in yaw_test_angles:
        if verbose:
            print(f"  Testing yaw = {yaw_deg:.1f}°...")
        
        try:
            # Get wake predictions from ML model
            nacelle_dir = 270.0 + yaw_deg  # Convert misalignment to nacelle direction
            wake_pred, _ = wake_agent.predict(
                yaw_angle=nacelle_dir, 
                n_timesteps=n_timesteps,
                export_vtk=False, 
                verbose=False
            )
            
            grid = wake_agent.grid
            print(f"[DEBUG find_optimal_yaw] Yaw={yaw_deg:.1f}°: grid is None? {grid is None}", file=sys.stderr)
            if grid is None:
                print(f"[DEBUG find_optimal_yaw] ❌ Grid is None at yaw={yaw_deg:.1f}°! Breaking loop.", file=sys.stderr)
                if verbose:
                    print("    Warning: No grid available, skipping ML-based analysis")
                break
            
            # Extract vertical profiles at start and reference distance
            profile_start = extract_vertical_profile_from_wake(
                wake_pred, grid, downstream_distance_m=0, 
                vertical_range_m=300, hub_height_m=90
            )
            
            profile_end = extract_vertical_profile_from_wake(
                wake_pred, grid, downstream_distance_m=reference_distance_m,
                vertical_range_m=300, hub_height_m=90
            )
            
            print(f"[DEBUG find_optimal_yaw] Yaw={yaw_deg:.1f}°: profile_start is None? {profile_start is None}, profile_end is None? {profile_end is None}", file=sys.stderr)
            if profile_start is None or profile_end is None:
                print(f"[DEBUG find_optimal_yaw] ❌ Profile extraction failed at yaw={yaw_deg:.1f}°. Skipping.", file=sys.stderr)
                if verbose:
                    print(f"    Skipping yaw={yaw_deg:.1f}° (insufficient grid points)")
                continue
            
            # Calculate wake centerline and deficit
            wake_analysis = calculate_wake_centerline_and_deficit(
                profile_start, profile_end, freestream_velocity=8.5
            )
            
            # Extrapolate to target distance (actual turbine spacing)
            wake_at_target = extrapolate_wake_trajectory(
                wake_analysis, target_distance_m, reference_distance_m
            )
            
            # Check overlap with downstream turbine
            overlap_check = check_wake_turbine_overlap(
                wake_at_target, downstream_turbine_position, rotor_radius_m
            )
            
            # Calculate power at this yaw angle
            # Upstream power: Use ML model at actual yaw
            upstream_result = power_agent.predict(yaw_angle=nacelle_dir, n_time_points=n_timesteps)
            upstream_power = extract_settled_power(upstream_result['power_mean_MW'])
            
            # Downstream power: Based on wake overlap
            if overlap_check['overlaps']:
                # Wake hits turbine - calculate power loss
                overlap_frac = overlap_check['overlap_fraction']
                deficit_at_turbine = wake_at_target['deficit_at_target'] * overlap_frac
                downstream_power = P_base * (1 - deficit_at_turbine)**3
            else:
                # Wake completely misses turbine - full power!
                downstream_power = P_base
            
            total_power = upstream_power + downstream_power
            
            wake_trajectory.append({
                'yaw_deg': yaw_deg,
                'wake_analysis': wake_analysis,
                'wake_at_target': wake_at_target,
                'overlap_check': overlap_check,
                'upstream_power_MW': upstream_power,
                'downstream_power_MW': downstream_power,
                'total_power_MW': total_power
            })
            
            print(f"[DEBUG find_optimal_yaw] ✅ Yaw={yaw_deg:.1f}° SUCCESS: migration={wake_at_target['lateral_migration_at_target']:.1f}m, overlap={overlap_check['overlap_fraction']:.1%}, total_power={total_power:.3f}MW", file=sys.stderr)
            
            if verbose:
                print(f"    Lateral migration: {wake_at_target['lateral_migration_at_target']:.1f}m, "
                      f"Overlap: {overlap_check['overlap_fraction']:.1%}, "
                      f"Total Power: {total_power:.3f} MW")
            
        except Exception as e:
            print(f"[DEBUG find_optimal_yaw] ❌ Exception at yaw={yaw_deg:.1f}°: {type(e).__name__}: {e}", file=sys.stderr)
            if verbose:
                print(f"    Error at yaw={yaw_deg:.1f}°: {e}")
            continue
    
    # Find minimum yaw where wake doesn't overlap (or has minimal overlap)
    print(f"[DEBUG find_optimal_yaw] Loop complete. wake_trajectory has {len(wake_trajectory)} entries", file=sys.stderr)
    
    optimal_yaw = yaw_constraint[0]  # Default: no yaw
    avoidance_achieved = False
    
    if not wake_trajectory:
        # No successful tests - return baseline
        print(f"[DEBUG find_optimal_yaw] ❌ wake_trajectory is EMPTY! All yaw tests failed.", file=sys.stderr)
    
    for result in wake_trajectory:
        if result['overlap_check']['overlap_fraction'] < 0.1:  # < 10% overlap
            optimal_yaw = result['yaw_deg']
            avoidance_achieved = True
            print(f"[DEBUG find_optimal_yaw] ✅ Wake avoidance achieved at {optimal_yaw:.1f}° (overlap < 10%)", file=sys.stderr)
            break
    
    if not avoidance_achieved and wake_trajectory:
        print(f"[DEBUG find_optimal_yaw] ⚠️ Wake avoidance not achieved. Using yaw with minimum overlap.", file=sys.stderr)
    
    return {
        'optimal_yaw': optimal_yaw,
        'wake_trajectory': wake_trajectory,
        'avoidance_achieved': avoidance_achieved,
        'turbine_spacing_m': target_distance_m,
        'turbine_spacing_D': turbine_spacing_D,
        'reference_distance_m': reference_distance_m
    }


# =============================================================================
# Wake Physics Models (Bastankhah & Porté-Agel 2016)
# =============================================================================
def calculate_wake_deficit(turbine_spacing_D, upstream_yaw_deg=0.0, thrust_coefficient=0.8, wake_expansion_coeff=0.022, lateral_offset_D=0.0):
    """
    Calculate wake velocity deficit at downstream turbine using Bastankhah & Porté-Agel (2016) model
    with full Gaussian lateral wake profile and yaw-induced wake deflection.

    The wake deficit depends on:
    1. Streamwise distance x/D (wake expands and centerline deficit recovers with distance)
    2. Upstream yaw angle γ (reduces thrust via C_T'=C_T·cos²γ, and deflects wake laterally)
    3. Lateral offset y/D (Gaussian decay: turbines off-centerline see less wake)
    4. Thrust coefficient and atmospheric conditions (wake expansion rate k*)

    Physics:
    - Centerline deficit:  δ_max = 1 - √(1 - C_T'/(8σ²)),  where σ/D = k*·(x/D) + ε
    - Wake deflection:     y_defl/D = θ₀·(x/D)/(1 + β·(x/D))  [Bastankhah 2016]
    - Effective offset:    y_eff = y_lateral + y_defl  (steering moves wake away)
    - Gaussian profile:    δ(x,y) = δ_max · exp(-y_eff²/(2σ²))

    Parameters:
    -----------
    turbine_spacing_D : float
        Streamwise (wind-aligned) distance between turbines in rotor diameters
    upstream_yaw_deg : float or torch.Tensor
        Upstream turbine yaw misalignment in degrees (0 = aligned with wind)
    thrust_coefficient : float
        Thrust coefficient (default: 0.8 for NREL 5MW)
    wake_expansion_coeff : float
        Wake expansion coefficient k* (default: 0.022 for neutral stability)
    lateral_offset_D : float
        Cross-wind lateral offset of downstream turbine in rotor diameters (default: 0.0 = on centerline).
        This is the absolute distance from the wind-aligned line through the upstream turbine.

    Returns:
    --------
    wake_deficit : float or torch.Tensor
        Velocity deficit fraction at the downstream turbine (0 to 0.6)
    """
    import torch
    import numpy as np

    C_T = thrust_coefficient
    k_star = wake_expansion_coeff
    x_D = turbine_spacing_D  # Streamwise distance in rotor diameters

    if isinstance(upstream_yaw_deg, torch.Tensor):
        gamma = upstream_yaw_deg * np.pi / 180.0

        # Yaw-corrected thrust coefficient: C_T' = C_T * cos²(γ)
        C_T_prime = C_T * torch.cos(gamma) ** 2

        # Wake width at downstream location: σ/D = k* * (x/D) + ε
        epsilon = 0.2  # Initial wake half-width parameter
        sigma_over_D = k_star * x_D + epsilon

        # Centerline (maximum) wake deficit at this streamwise distance
        # δ_max = 1 - √(1 - C_T' / (8(σ/D)²))
        denominator = 8.0 * (sigma_over_D ** 2)
        deficit_max = 1.0 - torch.sqrt(torch.clamp(1.0 - C_T_prime / denominator, min=0.01, max=1.0))

        # Wake lateral deflection (Bastankhah & Porté-Agel 2016)
        # θ₀ = 0.3γ/C_T · (1 - √(1-C_T))
        # β = 0.5·C_T/(1 - C_T/2)
        # y_defl/D = θ₀·(x/D) / (1 + β·(x/D))
        sqrt_1_minus_CT = torch.sqrt(torch.tensor(1.0 - C_T))
        theta_0 = (0.3 * gamma / C_T) * (1.0 - sqrt_1_minus_CT)
        beta = 0.5 * C_T / (1.0 - C_T / 2.0)
        y_deflection_D = theta_0 * x_D / (1.0 + beta * x_D)

        # Effective lateral offset: wake steering moves wake away from downstream turbine
        # y_eff = y_lateral + y_defl (deflection adds to offset, reducing wake impact)
        y_eff_D = lateral_offset_D + y_deflection_D

        # Gaussian lateral wake profile: δ(y) = δ_max · exp(-y²/(2σ²))
        lateral_factor = torch.exp(-y_eff_D ** 2 / (2.0 * sigma_over_D ** 2))

        wake_deficit = deficit_max * lateral_factor
        wake_deficit = torch.clamp(wake_deficit, 0.0, 0.6)

    else:
        # NumPy version (for grid_search, bayesian_transfer, etc.)
        gamma = np.radians(upstream_yaw_deg)
        C_T_prime = C_T * np.cos(gamma) ** 2

        epsilon = 0.2
        sigma_over_D = k_star * x_D + epsilon

        denominator = 8.0 * (sigma_over_D ** 2)
        deficit_max = 1.0 - np.sqrt(np.clip(1.0 - C_T_prime / denominator, 0.01, 1.0))

        # Wake lateral deflection
        theta_0 = (0.3 * gamma / C_T) * (1.0 - np.sqrt(1.0 - C_T))
        beta = 0.5 * C_T / (1.0 - C_T / 2.0)
        y_deflection_D = theta_0 * x_D / (1.0 + beta * x_D)

        # Effective lateral offset
        y_eff_D = lateral_offset_D + y_deflection_D

        # Gaussian lateral decay
        lateral_factor = np.exp(-y_eff_D ** 2 / (2.0 * sigma_over_D ** 2))

        wake_deficit = deficit_max * lateral_factor
        wake_deficit = np.clip(wake_deficit, 0.0, 0.6)

    return wake_deficit


def plot_ml_wake_extraction_results(wake_trajectory, baseline_total_power, turbine_spacing_D, 
                                      optimal_yaw, save_path=None):
    """
    Visualize ML wake extraction optimization results.
    
    Generates 3 subplots:
    1. Yaw angle vs Lateral wake migration
    2. Yaw angle vs Power (upstream, downstream, total)
    3. Yaw angle vs Power gain relative to baseline
    
    Parameters:
    -----------
    wake_trajectory : list
        Results from find_optimal_yaw_for_wake_avoidance containing yaw tests
    baseline_total_power : float
        Baseline total power at 0° yaw (MW)
    turbine_spacing_D : float
        Turbine spacing in rotor diameters
    optimal_yaw : float
        Optimal yaw angle found (degrees)
    save_path : str, optional
        Path to save figure (if None, returns fig object)
    
    Returns:
    --------
    matplotlib.figure.Figure or None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not wake_trajectory:
        print("No wake trajectory data to plot")
        return None
    
    # Extract data from wake_trajectory
    yaw_angles = [result['yaw_deg'] for result in wake_trajectory]
    lateral_migrations = [result['wake_at_target']['lateral_migration_at_target'] for result in wake_trajectory]
    upstream_powers = [result['upstream_power_MW'] for result in wake_trajectory]
    downstream_powers = [result['downstream_power_MW'] for result in wake_trajectory]
    total_powers = [result['total_power_MW'] for result in wake_trajectory]
    overlap_fractions = [result['overlap_check']['overlap_fraction'] for result in wake_trajectory]
    
    # Calculate power gains
    power_gains = [tp - baseline_total_power for tp in total_powers]
    power_gain_pcts = [(tp - baseline_total_power) / baseline_total_power * 100 for tp in total_powers]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Lateral Migration
    ax1 = axes[0]
    ax1.plot(yaw_angles, lateral_migrations, 'bo-', linewidth=2, markersize=8, label='Lateral Migration')
    ax1.axvline(optimal_yaw, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal Yaw: {optimal_yaw:.1f}°')
    ax1.set_xlabel('Upstream Yaw Misalignment (°)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Lateral Wake Migration (m)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Wake Deflection vs Yaw Angle\nTurbine Spacing: {turbine_spacing_D:.1f}D', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Subplot 2: Power Components
    ax2 = axes[1]
    ax2.plot(yaw_angles, upstream_powers, 'b^-', linewidth=2, markersize=8, label='Upstream Power')
    ax2.plot(yaw_angles, downstream_powers, 'gs-', linewidth=2, markersize=8, label='Downstream Power')
    ax2.plot(yaw_angles, total_powers, 'ro-', linewidth=3, markersize=10, label='Total Farm Power')
    ax2.axhline(baseline_total_power, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Baseline (0° yaw)')
    ax2.axvline(optimal_yaw, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal: {optimal_yaw:.1f}°')
    ax2.set_xlabel('Upstream Yaw Misalignment (°)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
    ax2.set_title('Power Generation vs Yaw Angle', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    # Subplot 3: Power Gain
    ax3 = axes[2]
    colors = ['green' if pg > 0 else 'red' for pg in power_gains]
    ax3.bar(yaw_angles, power_gain_pcts, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(optimal_yaw, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal: {optimal_yaw:.1f}°')
    ax3.set_xlabel('Upstream Yaw Misalignment (°)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Power Gain (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Farm Power Gain vs Baseline', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig


def calculate_wake_deflected_downstream_yaw(upstream_yaw_deg, turbine_spacing_D, thrust_coefficient=0.8):
    """
    Calculate downstream turbine yaw angle to align with deflected wake.
    
    Based on Bastankhah & Porté-Agel (2016) wake deflection model.
    When upstream turbine is yawed, the wake deflects laterally.
    Downstream turbine should align with this deflected wake for maximum power capture.
    
    Parameters:
    -----------
    upstream_yaw_deg : float or torch.Tensor
        Upstream turbine yaw misalignment in degrees
    turbine_spacing_D : float
        Distance between turbines in rotor diameters
    thrust_coefficient : float
        Thrust coefficient (default: 0.8)
    
    Returns:
    --------
    downstream_yaw_deg : float or torch.Tensor
        Recommended downstream turbine yaw to align with deflected wake
    """
    import torch
    import numpy as np
    
    # Convert to radians
    if isinstance(upstream_yaw_deg, torch.Tensor):
        upstream_yaw_rad = upstream_yaw_deg * np.pi / 180.0
        
        # Wake deflection model (Bastankhah & Porté-Agel 2016)
        # Initial wake deflection coefficient
        C_T = thrust_coefficient
        theta_0 = (0.3 * upstream_yaw_rad / C_T) * (1.0 - torch.sqrt(torch.tensor(1.0 - C_T)))
        
        # Wake expansion parameter
        beta = 0.5 * (C_T / (1.0 - C_T / 2.0))
        
        # Lateral wake deflection at downstream turbine location
        x_D = turbine_spacing_D  # Distance in rotor diameters
        deflection_lateral = theta_0 * x_D / (1.0 + beta * x_D)
        
        # Deflection angle (approximation for small angles)
        # δ_angle ≈ arctan(deflection_lateral / x_D) ≈ deflection_lateral / x_D for small angles
        deflection_angle_rad = deflection_lateral / x_D
        
        # Downstream turbine should align with deflected wake
        # For optimal power: yaw towards the deflection
        downstream_yaw_rad = deflection_angle_rad
        downstream_yaw_deg = downstream_yaw_rad * 180.0 / np.pi
        
    else:
        # NumPy version
        upstream_yaw_rad = np.radians(upstream_yaw_deg)
        
        C_T = thrust_coefficient
        theta_0 = (0.3 * upstream_yaw_rad / C_T) * (1.0 - np.sqrt(1.0 - C_T))
        beta = 0.5 * (C_T / (1.0 - C_T / 2.0))
        
        x_D = turbine_spacing_D
        deflection_lateral = theta_0 * x_D / (1.0 + beta * x_D)
        deflection_angle_rad = deflection_lateral / x_D
        
        downstream_yaw_rad = deflection_angle_rad
        downstream_yaw_deg = np.degrees(downstream_yaw_rad)
    
    # Clamp to reasonable range (-5° to +5°)
    if isinstance(downstream_yaw_deg, torch.Tensor):
        downstream_yaw_deg = torch.clamp(downstream_yaw_deg, -5.0, 5.0)
    else:
        downstream_yaw_deg = np.clip(downstream_yaw_deg, -5.0, 5.0)
    
    return downstream_yaw_deg


def extract_settled_power(power_trajectory, weight_decay=0.1):
    """
    Extract the initial power from a GP transient trajectory.

    The GP model predicts transient power dynamics over normalized time (0->1).
    Using the first time-step (t=0) captures the instantaneous yaw-dependent
    power response before transient ramp effects dilute the yaw sensitivity.

    Parameters:
        power_trajectory: np.ndarray - GP posterior mean power over time
        weight_decay: float - unused, kept for API compatibility

    Returns:
        float - power at first time-step (MW)
    """
    return float(power_trajectory[0])


# =============================================================================
# Two-Turbine Wake Steering Optimizer (Supports Both Methods)
# =============================================================================
def optimize_two_turbine_farm(power_agent, wake_agent, turbine_spacing_D: float = 7.0,
                               n_timesteps: int = 50, verbose: bool = False,
                               optimization_method: str = 'analytical_physics',
                               lateral_offset_D: float = 0.0):
    """
    Optimize yaw misalignment angles for a two-turbine wind farm.
    
    Supports two optimization approaches:
    1. 'analytical_physics': AD through analytical wake steering physics (fast)
    2. 'ml_surrogate': True AD through differentiable ML surrogate models (accurate)
    
    The optimizer finds the optimal yaw misalignment for each turbine (0-12°)
    to maximize total farm power output, considering wake effects.
    
    Parameters:
    -----------
    power_agent : RotorPowerAgent
        Power prediction agent (GP model)
    wake_agent : WakeFlowAgent
        Wake flow prediction agent (TT-OpInf model)
    turbine_spacing_D : float
        Streamwise (wind-aligned) distance between turbines in rotor diameters (default: 7D)
    n_timesteps : int
        Number of timesteps for predictions
    verbose : bool
        Print optimization progress
    lateral_offset_D : float
        Cross-wind lateral offset of downstream turbine in rotor diameters (default: 0.0).
        Used for Gaussian lateral wake profile — turbines off-centerline see reduced wake.
    optimization_method : str
        'analytical_physics': Use analytical wake steering model with AD
        'ml_surrogate': Use differentiable surrogate fitted to ML models
        'grid_search': Brute-force grid search (fallback)
    
    Returns:
    --------
    dict
        Optimization results including optimal misalignments and power values
    """
    import torch
    
    # Turbine specifications
    rotor_diameter = 126.0  # meters (NREL 5MW)
    freestream_velocity = 8.5  # Approximate freestream (m/s)
    
    results_history = []
    
    if optimization_method == 'ml_surrogate':
        # =====================================================================
        # Method 1: Direct GP Model Evaluation Grid
        # =====================================================================
        # Uses actual GP model predictions at each candidate yaw angle,
        # combined with per-pair Bastankhah wake deficit.
        # - Upstream power: actual GP prediction at each yaw
        # - Wake deficit: Bastankhah model with per-pair spacing & lateral offset
        # - Downstream power: actual GP prediction × (1 - deficit)³
        # =====================================================================
        if verbose:
            print("Using ML Surrogate grid evaluation (direct GP predictions)...")

        # Get baseline power from GP model at 0° yaw misalignment
        baseline_result_ml = power_agent.predict(yaw_angle=270.0, n_time_points=n_timesteps)
        P_base_ml = extract_settled_power(baseline_result_ml['power_mean_MW'])

        # Baseline wake deficit at 0° yaw
        baseline_deficit_ml = calculate_wake_deficit(
            turbine_spacing_D=turbine_spacing_D,
            upstream_yaw_deg=0.0,
            thrust_coefficient=0.8,
            wake_expansion_coeff=0.022,
            lateral_offset_D=lateral_offset_D
        )

        best_power = -np.inf
        best_up_power = P_base_ml
        best_down_power = P_base_ml * 0.7
        best_deficit = baseline_deficit_ml
        optimal_upstream = 0.0
        optimal_downstream = 0.0

        # Dense 1° resolution grid
        misalign_range = np.arange(0, 13, 1)

        for up_misalign in misalign_range:
            # Downstream yaw from wake deflection physics (spacing-dependent)
            down_misalign = calculate_wake_deflected_downstream_yaw(
                up_misalign, turbine_spacing_D
            )
            if hasattr(down_misalign, 'item'):
                down_misalign = down_misalign.item()

            # Upstream power: actual GP model prediction
            upstream_nacelle = 270.0 + up_misalign
            upstream_result = power_agent.predict(yaw_angle=upstream_nacelle, n_time_points=n_timesteps)
            up_power = extract_settled_power(upstream_result['power_mean_MW'])

            # Wake deficit: Bastankhah model (per-pair spacing + lateral offset)
            eff_deficit = calculate_wake_deficit(
                turbine_spacing_D=turbine_spacing_D,
                upstream_yaw_deg=float(up_misalign),
                thrust_coefficient=0.8,
                wake_expansion_coeff=0.022,
                lateral_offset_D=lateral_offset_D
            )
            if hasattr(eff_deficit, 'item'):
                eff_deficit = float(eff_deficit.item())
            else:
                eff_deficit = float(eff_deficit)

            # Downstream power: GP prediction if within range, else cos³ fallback
            downstream_nacelle = 270.0 + down_misalign
            if 270.0 <= downstream_nacelle <= 285.0:
                downstream_result = power_agent.predict(yaw_angle=downstream_nacelle, n_time_points=n_timesteps)
                down_power_base = extract_settled_power(downstream_result['power_mean_MW'])
            else:
                down_rad = np.radians(down_misalign)
                down_power_base = P_base_ml * np.cos(down_rad)**3

            down_power = down_power_base * (1 - eff_deficit)**3
            total = up_power + down_power

            results_history.append({
                'upstream_misalignment': float(up_misalign),
                'downstream_misalignment': float(down_misalign),
                'upstream_power': up_power,
                'downstream_power': down_power,
                'total_power': total,
                'wake_deficit': eff_deficit
            })

            if total > best_power:
                best_power = total
                best_up_power = up_power
                best_down_power = down_power
                best_deficit = eff_deficit
                optimal_upstream = float(up_misalign)
                optimal_downstream = float(down_misalign)

        # Compute baseline total for comparison
        baseline_downstream_power = P_base_ml * (1 - baseline_deficit_ml)**3
        baseline_total = P_base_ml + baseline_downstream_power

        return {
            'optimal_upstream_misalignment': optimal_upstream,
            'optimal_downstream_misalignment': optimal_downstream,
            'optimal_upstream_nacelle': yaw_misalignment_to_nacelle_direction(optimal_upstream),
            'optimal_downstream_nacelle': yaw_misalignment_to_nacelle_direction(optimal_downstream),
            'optimal_upstream_power': best_up_power if results_history else P_base_ml,
            'optimal_downstream_power': best_down_power if results_history else P_base_ml * 0.7,
            'optimal_total_power': best_power,
            'optimal_wake_deficit': best_deficit if results_history else baseline_deficit_ml,
            'baseline_total_power': baseline_total,
            'baseline_upstream_power': P_base_ml,
            'baseline_downstream_power': baseline_downstream_power,
            'baseline_wake_deficit': baseline_deficit_ml,
            'power_gain_MW': best_power - baseline_total,
            'power_gain_percent': ((best_power - baseline_total) / baseline_total * 100) if baseline_total > 0 else 0.0,
            'turbine_spacing_D': turbine_spacing_D,
            'optimization_method': 'ML Surrogate (Direct GP Evaluation)',
            'all_results': results_history
        }
        
    elif optimization_method == 'analytical_physics':
        # =====================================================================
        # Method 2: AD through Analytical Physics Model
        # =====================================================================
        if verbose:
            print("Using analytical physics model with AD...")
        
        # Get baseline power (constant, from ML model)
        baseline_result = power_agent.predict(yaw_angle=270.0, n_time_points=n_timesteps)
        P_base = extract_settled_power(baseline_result['power_mean_MW'])
        
        # Get baseline wake deficit (constant, from ML model)
        wake_pred, _ = wake_agent.predict(yaw_angle=270.0, n_timesteps=n_timesteps,
                                          export_vtk=False, verbose=False)
        vel_mag = np.linalg.norm(wake_pred, axis=2)
        base_deficit = 1.0 - (np.mean(vel_mag) / freestream_velocity)
        base_deficit = np.clip(base_deficit, 0, 0.5)
        
        def compute_farm_power_physics(upstream_misalignment):
            """
            Compute farm power using analytical physics with AD.
            Uses distance-dependent wake model from Bastankhah & Porté-Agel (2016).
            
            Only upstream yaw is optimized. Downstream yaw is calculated
            based on wake deflection physics to align with deflected wake.
            """
            # Calculate downstream yaw based on wake deflection
            downstream_misalign = calculate_wake_deflected_downstream_yaw(
                upstream_misalignment, turbine_spacing_D
            )
            
            # ============================================================
            # Upstream Power: P = P_base * cos³(γ)
            # ============================================================
            upstream_rad = upstream_misalignment * np.pi / 180.0
            upstream_cos_loss = torch.cos(upstream_rad) ** 3
            upstream_power = P_base * upstream_cos_loss
            
            # ============================================================
            # Wake Deficit (Distance-Dependent Bastankhah & Porté-Agel 2016)
            # KEY FIX: Wake deficit now depends on turbine_spacing_D!
            # ============================================================
            wake_deficit = calculate_wake_deficit(
                turbine_spacing_D=turbine_spacing_D,
                upstream_yaw_deg=upstream_misalignment,
                thrust_coefficient=0.8,
                wake_expansion_coeff=0.022,
                lateral_offset_D=lateral_offset_D
            )

            # ============================================================
            # Downstream Power: P = P_base * (1-δ)³ * cos³(γ)
            # ============================================================
            downstream_rad = downstream_misalign * np.pi / 180.0
            downstream_cos_loss = torch.cos(downstream_rad) ** 3
            effective_wind_ratio = 1.0 - wake_deficit
            downstream_power = P_base * (effective_wind_ratio ** 3) * downstream_cos_loss
            
            total_power = upstream_power + downstream_power
            
            return total_power, upstream_power, downstream_power, wake_deficit, downstream_misalign
        
        compute_fn = compute_farm_power_physics
        method_name = 'Analytical Physics AD'
        
    elif optimization_method == 'grid_search':
        # =====================================================================
        # Method 3: Grid Search (Fallback)
        # =====================================================================
        if verbose:
            print("Using grid search optimization...")
        
        # Get baseline values
        baseline_result = power_agent.predict(yaw_angle=270.0, n_time_points=n_timesteps)
        P_base = extract_settled_power(baseline_result['power_mean_MW'])
        
        wake_pred, _ = wake_agent.predict(yaw_angle=270.0, n_timesteps=n_timesteps,
                                          export_vtk=False, verbose=False)
        vel_mag = np.linalg.norm(wake_pred, axis=2)
        base_deficit = 1.0 - (np.mean(vel_mag) / freestream_velocity)
        base_deficit = np.clip(base_deficit, 0, 0.5)
        
        best_power = -np.inf
        optimal_upstream = 0.0
        optimal_downstream = 0.0
        
        misalign_range = np.arange(0, 13, 2)
        
        for up_misalign in misalign_range:
            # Calculate downstream yaw based on wake deflection physics
            down_misalign = calculate_wake_deflected_downstream_yaw(
                up_misalign, turbine_spacing_D
            )
            
            up_rad = np.radians(up_misalign)
            down_rad = np.radians(down_misalign)
            
            up_power = P_base * np.cos(up_rad)**3
            
            # Distance-dependent wake deficit with lateral offset
            eff_deficit = calculate_wake_deficit(
                turbine_spacing_D=turbine_spacing_D,
                upstream_yaw_deg=up_misalign,
                thrust_coefficient=0.8,
                wake_expansion_coeff=0.022,
                lateral_offset_D=lateral_offset_D
            )
            
            down_power = P_base * (1 - eff_deficit)**3 * np.cos(down_rad)**3
            total = up_power + down_power
            
            results_history.append({
                'upstream_misalignment': up_misalign,
                'downstream_misalignment': down_misalign,
                'upstream_power': up_power,
                'downstream_power': down_power,
                'total_power': total,
                'wake_deficit': eff_deficit
            })
            
            if total > best_power:
                best_power = total
                optimal_upstream = up_misalign
                optimal_downstream = down_misalign
        
        # Return grid search results
        return {
            'optimal_upstream_misalignment': optimal_upstream,
            'optimal_downstream_misalignment': optimal_downstream,
            'optimal_upstream_nacelle': yaw_misalignment_to_nacelle_direction(optimal_upstream),
            'optimal_downstream_nacelle': yaw_misalignment_to_nacelle_direction(optimal_downstream),
            'optimal_upstream_power': results_history[-1]['upstream_power'] if results_history else P_base,
            'optimal_downstream_power': results_history[-1]['downstream_power'] if results_history else P_base * 0.7,
            'optimal_total_power': best_power,
            'optimal_wake_deficit': base_deficit,
            'baseline_total_power': 2 * P_base * (1 - base_deficit)**1.5,
            'baseline_upstream_power': P_base,
            'baseline_downstream_power': P_base * (1 - base_deficit)**3,
            'baseline_wake_deficit': base_deficit,
            'power_gain_MW': best_power - 2 * P_base * (1 - base_deficit)**1.5,
            'power_gain_percent': 0.0,
            'turbine_spacing_D': turbine_spacing_D,
            'optimization_method': 'Grid Search',
            'all_results': results_history
        }
    
    elif optimization_method == 'bayesian_transfer':
        # =====================================================================
        # Method 4: Bayesian Optimization with Transfer Learning
        # =====================================================================
        if verbose:
            print("Using Bayesian Optimization with transfer learning...")
            print("Base model: Pre-trained GPR | Residual: Matérn 5/2 GP")
            print("Acquisition: Expected Improvement (EI) | Optimizer: Multi-start L-BFGS-B")
            print("Constraint: 0-15° yaw misalignment (expanded from 12° for other methods)")
        
        # Initialize Bayesian Transfer Optimizer
        bo_optimizer = BayesianTransferOptimizer(
            base_gp_agent=power_agent,
            wake_agent=wake_agent,
            turbine_spacing_D=turbine_spacing_D,
            n_initial_samples=8,
            xi_lengthscale_range=(0.01, 0.05),
            n_timesteps=n_timesteps,
            freestream_velocity=freestream_velocity,
            lateral_offset_D=lateral_offset_D
        )

        # Run Bayesian Optimization (25 iterations for better spacing-dependent convergence)
        bo_results = bo_optimizer.optimize(n_iterations=25, verbose=verbose, xi_range=(0.01, 0.05))
        
        # Extract optimal values
        optimal_upstream = bo_results['optimal_upstream_misalignment']
        optimal_downstream = bo_results['optimal_downstream_misalignment']
        optimal_power = bo_results['optimal_total_power']
        baseline_power = bo_results['baseline_total_power']
        power_gain = bo_results['power_gain_MW']
        power_gain_pct = bo_results['power_gain_percent']
        
        # Get detailed power breakdown at optimal point
        upstream_nacelle = yaw_misalignment_to_nacelle_direction(optimal_upstream)
        downstream_nacelle = yaw_misalignment_to_nacelle_direction(optimal_downstream)

        # Upstream power
        upstream_result = power_agent.predict(yaw_angle=upstream_nacelle, n_time_points=n_timesteps)
        optimal_upstream_power = extract_settled_power(upstream_result['power_mean_MW'])

        # Distance-dependent wake deficit at optimal yaw with lateral offset
        optimal_deficit = calculate_wake_deficit(
            turbine_spacing_D=turbine_spacing_D,
            upstream_yaw_deg=optimal_upstream,
            thrust_coefficient=0.8,
            wake_expansion_coeff=0.022,
            lateral_offset_D=lateral_offset_D
        )

        # Downstream power
        downstream_result = power_agent.predict(yaw_angle=downstream_nacelle, n_time_points=n_timesteps)
        downstream_power_base = extract_settled_power(downstream_result['power_mean_MW'])
        optimal_downstream_power = downstream_power_base * (1 - optimal_deficit)**3

        # Baseline values
        baseline_result = power_agent.predict(yaw_angle=270.0, n_time_points=n_timesteps)
        baseline_upstream_power = extract_settled_power(baseline_result['power_mean_MW'])

        # Baseline wake deficit at 0° yaw with lateral offset
        baseline_deficit = calculate_wake_deficit(
            turbine_spacing_D=turbine_spacing_D,
            upstream_yaw_deg=0.0,
            thrust_coefficient=0.8,
            wake_expansion_coeff=0.022,
            lateral_offset_D=lateral_offset_D
        )
        baseline_downstream_power = baseline_upstream_power * (1 - baseline_deficit)**3
        
        if verbose:
            print(f"\n✅ Bayesian Optimization Complete!")
            print(f"  Optimal upstream yaw: {optimal_upstream:.2f}°")
            print(f"  Optimal downstream yaw: {optimal_downstream:.2f}° (physics-calculated)")
            print(f"  Total farm power: {optimal_power:.4f} MW")
            print(f"  Power gain: {power_gain:.4f} MW (+{power_gain_pct:.2f}%)")
            print(f"  Total evaluations: {bo_results['n_evaluations']}")
        
        return {
            'optimal_upstream_misalignment': optimal_upstream,
            'optimal_downstream_misalignment': optimal_downstream,
            'optimal_upstream_nacelle': upstream_nacelle,
            'optimal_downstream_nacelle': downstream_nacelle,
            'optimal_upstream_power': optimal_upstream_power,
            'optimal_downstream_power': optimal_downstream_power,
            'optimal_total_power': optimal_power,
            'optimal_wake_deficit': optimal_deficit,
            'baseline_total_power': baseline_power,
            'baseline_upstream_power': baseline_upstream_power,
            'baseline_downstream_power': baseline_downstream_power,
            'baseline_wake_deficit': baseline_deficit,
            'power_gain_MW': power_gain,
            'power_gain_percent': power_gain_pct,
            'turbine_spacing_D': turbine_spacing_D,
            'optimization_method': 'Bayesian Optimization (Transfer Learning)',
            'n_evaluations': bo_results['n_evaluations'],
            'all_results': bo_results['results_history']
        }
    
    elif optimization_method == 'ml_wake_extraction':
        # =====================================================================
        # Method 5: ML Wake Extraction (Expert Approach)
        # =====================================================================
        if verbose:
            print("Using ML Wake Extraction method (Expert Approach)...")
            print("Strategy: Extract vertical profiles → Track wake centerline → Extrapolate to target → Check overlap")
        
        # CRITICAL: Check if wake_agent has grid (required for ML wake extraction)
        if not hasattr(wake_agent, 'grid') or wake_agent.grid is None:
            if verbose:
                print("❌ ERROR: ML Wake Extraction requires wake_agent.grid, but it's None!")
                print("   Falling back to analytical_physics method...")
            # Recursively call with analytical_physics method
            return optimize_two_turbine_farm(
                power_agent=power_agent,
                wake_agent=wake_agent,
                turbine_spacing_D=turbine_spacing_D,
                lateral_offset_D=lateral_offset_D,
                n_timesteps=n_timesteps,
                verbose=verbose,
                optimization_method='analytical_physics'
            )
        
        # Get baseline power
        baseline_result = power_agent.predict(yaw_angle=270.0, n_time_points=n_timesteps)
        P_base = extract_settled_power(baseline_result['power_mean_MW'])

        # Get baseline wake at 0° yaw
        baseline_wake_pred, _ = wake_agent.predict(yaw_angle=270.0, n_timesteps=n_timesteps,
                                                   export_vtk=False, verbose=False)
        baseline_vel_mag = np.linalg.norm(baseline_wake_pred, axis=2)
        baseline_deficit = 1.0 - (np.mean(baseline_vel_mag) / freestream_velocity)
        baseline_deficit = np.clip(baseline_deficit, 0, 0.5)
        
        # Downstream turbine position at actual streamwise distance and lateral offset
        turbine_spacing_m = turbine_spacing_D * rotor_diameter
        lateral_offset_m = lateral_offset_D * rotor_diameter
        downstream_turbine_position = {'x': turbine_spacing_m, 'y': lateral_offset_m}
        
        # Use the comprehensive helper function to find optimal yaw
        optimization_result = find_optimal_yaw_for_wake_avoidance(
            wake_agent=wake_agent,
            power_agent=power_agent,
            P_base=P_base,
            turbine_spacing_D=turbine_spacing_D,
            rotor_diameter=rotor_diameter,
            downstream_turbine_position=downstream_turbine_position,
            yaw_constraint=(0, 15),
            n_timesteps=n_timesteps,
            reference_distance_m=None,  # Auto-detect from grid extent
            verbose=verbose
        )
        
        # CRITICAL CHECK: If wake_trajectory is empty, all tests failed!
        if not optimization_result['wake_trajectory']:
            if verbose:
                print("❌ ERROR: ML Wake Extraction failed - wake_trajectory is empty!")
                print("   All yaw angle tests failed. Possible causes:")
                print("   - Grid points insufficient at required distances")
                print("   - Profile extraction failed")
                print("   - ML model prediction errors")
                print("   Returning baseline (no optimization possible)")
            
            # Return baseline results (no change)
            baseline_upstream_power = P_base
            baseline_downstream_power = P_base * (1 - baseline_deficit)**3
            baseline_total_power = baseline_upstream_power + baseline_downstream_power
            
            return {
                'optimal_upstream_misalignment': 0.0,
                'optimal_downstream_misalignment': 0.0,
                'optimal_upstream_nacelle': 270.0,
                'optimal_downstream_nacelle': 270.0,
                'optimal_upstream_power': baseline_upstream_power,
                'optimal_downstream_power': baseline_downstream_power,
                'optimal_total_power': baseline_total_power,
                'optimal_wake_deficit': baseline_deficit,
                'baseline_total_power': baseline_total_power,
                'baseline_upstream_power': baseline_upstream_power,
                'baseline_downstream_power': baseline_downstream_power,
                'baseline_wake_deficit': baseline_deficit,
                'power_gain_MW': 0.0,
                'power_gain_percent': 0.0,
                'turbine_spacing_D': turbine_spacing_D,
                'optimization_method': 'ML Wake Extraction (Expert Approach) - FAILED',
                'wake_avoidance_achieved': False,
                'error': 'Wake trajectory empty - all tests failed',
                'all_results': []
            }
        
        if not optimization_result['avoidance_achieved']:
            if verbose:
                print("⚠️ Wake avoidance not achievable within yaw constraints")
                print(f"   Best yaw: {optimization_result['optimal_yaw']:.2f}° (still has overlap)")
        else:
            if verbose:
                print(f"✅ Wake avoidance achieved at {optimization_result['optimal_yaw']:.2f}°")
        
        # Get optimal yaw and corresponding wake analysis
        optimal_upstream = optimization_result['optimal_yaw']
        optimal_downstream = 0.0  # Downstream stays aligned (no deflection needed)
        
        # Get optimal wake trajectory for power calculation
        optimal_wake_info = None
        for traj in optimization_result['wake_trajectory']:
            if abs(traj['yaw_deg'] - optimal_upstream) < 0.1:
                optimal_wake_info = traj
                break
        
        # Extract powers from optimal wake info (already calculated in find_optimal_yaw_for_wake_avoidance)
        if optimal_wake_info:
            optimal_upstream_power = optimal_wake_info['upstream_power_MW']
            optimal_downstream_power = optimal_wake_info['downstream_power_MW']
            
            # Get deficit for reporting
            if optimal_wake_info['overlap_check']['overlaps']:
                overlap_frac = optimal_wake_info['overlap_check']['overlap_fraction']
                optimal_deficit = optimal_wake_info['wake_at_target']['deficit_at_target'] * overlap_frac
            else:
                optimal_deficit = 0.0
        else:
            # CRITICAL BUG FIX: If optimal_yaw = 0°, use baseline powers (no change!)
            if abs(optimal_upstream) < 0.01:  # Essentially 0°
                if verbose:
                    print("⚠️ WARNING: Optimal yaw is 0° (no change from baseline)")
                    print("   Using baseline powers (no optimization benefit)")
                optimal_upstream_power = P_base
                optimal_downstream_power = P_base * (1 - baseline_deficit)**3
                optimal_deficit = baseline_deficit
            else:
                # For non-zero yaw, estimate powers (this shouldn't happen if wake_trajectory exists)
                if verbose:
                    print(f"⚠️ WARNING: Optimal yaw is {optimal_upstream:.2f}° but no trajectory data found")
                    print("   Using analytical approximation")
                optimal_upstream_rad = np.radians(optimal_upstream)
                optimal_upstream_power = P_base * np.cos(optimal_upstream_rad)**3
                # Use same deficit as baseline (conservative estimate)
                optimal_downstream_power = P_base * (1 - baseline_deficit)**3 * np.cos(optimal_upstream_rad)**0.5
                optimal_deficit = baseline_deficit * (1 - 0.3 * np.sin(optimal_upstream_rad))
        
        optimal_total_power = optimal_upstream_power + optimal_downstream_power
        
        # Baseline powers
        baseline_upstream_power = P_base
        baseline_downstream_power = P_base * (1 - baseline_deficit)**3
        baseline_total_power = baseline_upstream_power + baseline_downstream_power
        
        # Power gain
        power_gain = optimal_total_power - baseline_total_power
        power_gain_pct = (power_gain / baseline_total_power) * 100 if baseline_total_power > 0 else 0
        
        if verbose:
            print(f"\n🎯 ML Wake Extraction Optimization Complete!")
            print(f"  Optimal upstream yaw: {optimal_upstream:.2f}°")
            print(f"  Optimal downstream yaw: {optimal_downstream:.2f}° (aligned)")
            print(f"  Wake avoidance: {'YES ✅' if optimization_result['avoidance_achieved'] else 'NO ❌'}")
            print(f"  Baseline power: {baseline_total_power:.4f} MW")
            print(f"  Optimal power: {optimal_total_power:.4f} MW")
            print(f"  Power gain: {power_gain:.4f} MW (+{power_gain_pct:.2f}%)")
            print(f"  Tested {len(optimization_result['wake_trajectory'])} yaw angles")
            if abs(power_gain) < 1e-6 and abs(optimal_upstream) < 0.01:
                print(f"  ⚠️  NOTE: Zero gain is correct - no yaw change from baseline")
        
        # Generate visualization of yaw sensitivity analysis
        try:
            fig = plot_ml_wake_extraction_results(
                wake_trajectory=optimization_result['wake_trajectory'],
                baseline_total_power=baseline_total_power,
                turbine_spacing_D=turbine_spacing_D,
                optimal_yaw=optimal_upstream,
                save_path=None  # Return fig object for display in GUI
            )
            optimization_result['sensitivity_figure'] = fig
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not generate sensitivity plots: {e}")
            optimization_result['sensitivity_figure'] = None
        
        return {
            'optimal_upstream_misalignment': optimal_upstream,
            'optimal_downstream_misalignment': optimal_downstream,
            'optimal_upstream_nacelle': yaw_misalignment_to_nacelle_direction(optimal_upstream),
            'optimal_downstream_nacelle': yaw_misalignment_to_nacelle_direction(optimal_downstream),
            'optimal_upstream_power': optimal_upstream_power,
            'optimal_downstream_power': optimal_downstream_power,
            'optimal_total_power': optimal_total_power,
            'optimal_wake_deficit': optimal_deficit,
            'baseline_total_power': baseline_total_power,
            'baseline_upstream_power': baseline_upstream_power,
            'baseline_downstream_power': baseline_downstream_power,
            'baseline_wake_deficit': baseline_deficit,
            'power_gain_MW': power_gain,
            'power_gain_percent': power_gain_pct,
            'turbine_spacing_D': turbine_spacing_D,
            'optimization_method': 'ML Wake Extraction (Expert Approach)',
            'wake_avoidance_achieved': optimization_result['avoidance_achieved'],
            'all_results': optimization_result['wake_trajectory']
        }
    
    # =========================================================================
    # Gradient-Based Optimization (for both ML surrogate and physics methods)
    # =========================================================================
    # Only optimize upstream yaw; downstream is calculated from wake deflection physics
    upstream_param = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    
    lower_bound = 0.0
    upper_bound = 12.0
    
    optimizer = torch.optim.Adam([upstream_param], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_power = -np.inf
    best_upstream = upstream_param.detach().clone()
    best_downstream = 0.0
    
    n_iterations = 50
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        total_power, up_power, down_power, wake_deficit, down_misalign = compute_fn(upstream_param)
        
        loss = -total_power
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_([upstream_param], max_norm=5.0)
        optimizer.step()
        
        with torch.no_grad():
            upstream_param.clamp_(min=lower_bound, max=upper_bound)
        
        current_power = float(total_power.detach().numpy()) if torch.is_tensor(total_power) else float(total_power)
        scheduler.step(current_power)
        
        if current_power > best_power:
            best_power = current_power
            best_upstream = upstream_param.detach().clone()
            best_downstream = float(down_misalign.detach().numpy()) if torch.is_tensor(down_misalign) else float(down_misalign)
        
        results_history.append({
            'iteration': iteration,
            'upstream_misalignment': float(upstream_param.detach().numpy()),
            'downstream_misalignment': float(down_misalign.detach().numpy()) if torch.is_tensor(down_misalign) else float(down_misalign),
            'upstream_power': float(up_power.detach().numpy()) if torch.is_tensor(up_power) else float(up_power),
            'downstream_power': float(down_power.detach().numpy()) if torch.is_tensor(down_power) else float(down_power),
            'total_power': current_power,
            'wake_deficit': float(wake_deficit.detach().numpy()) if torch.is_tensor(wake_deficit) else float(wake_deficit),
            'gradient_norm': float(upstream_param.grad.numpy()) if upstream_param.grad is not None else 0.0
        })
        
        if verbose and iteration % 10 == 0:
            grad_norm = upstream_param.grad.item() if upstream_param.grad is not None else 0.0
            down_val = float(down_misalign.detach().numpy()) if torch.is_tensor(down_misalign) else float(down_misalign)
            print(f"  Iter {iteration:3d}: Upstream={upstream_param.item():.2f}°, Downstream={down_val:.2f}° (physics), "
                  f"Power={current_power:.4f} MW, |∇|={grad_norm:.4f}")
        
        if upstream_param.grad is not None and abs(upstream_param.grad.item()) < 1e-4:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
    
    optimal_upstream = float(best_upstream.numpy())
    optimal_downstream = best_downstream
    
    # Evaluate at optimal and baseline points
    with torch.no_grad():
        opt_total, opt_up, opt_down, opt_deficit, opt_down_misalign = compute_fn(best_upstream)
        baseline_total, baseline_up, baseline_down, baseline_deficit, baseline_down_misalign = compute_fn(
            torch.tensor(0.0, dtype=torch.float64)
        )
    
    opt_total = float(opt_total.numpy()) if torch.is_tensor(opt_total) else float(opt_total)
    opt_up = float(opt_up.numpy()) if torch.is_tensor(opt_up) else float(opt_up)
    opt_down = float(opt_down.numpy()) if torch.is_tensor(opt_down) else float(opt_down)
    opt_deficit = float(opt_deficit.numpy()) if torch.is_tensor(opt_deficit) else float(opt_deficit)
    baseline_total = float(baseline_total.numpy()) if torch.is_tensor(baseline_total) else float(baseline_total)
    baseline_up = float(baseline_up.numpy()) if torch.is_tensor(baseline_up) else float(baseline_up)
    baseline_down = float(baseline_down.numpy()) if torch.is_tensor(baseline_down) else float(baseline_down)
    baseline_deficit = float(baseline_deficit.numpy()) if torch.is_tensor(baseline_deficit) else float(baseline_deficit)
    
    power_gain = opt_total - baseline_total
    power_gain_pct = (power_gain / baseline_total) * 100 if baseline_total > 0 else 0
    
    return {
        'optimal_upstream_misalignment': optimal_upstream,
        'optimal_downstream_misalignment': optimal_downstream,
        'optimal_upstream_nacelle': yaw_misalignment_to_nacelle_direction(optimal_upstream),
        'optimal_downstream_nacelle': yaw_misalignment_to_nacelle_direction(optimal_downstream),
        'optimal_upstream_power': opt_up,
        'optimal_downstream_power': opt_down,
        'optimal_total_power': opt_total,
        'optimal_wake_deficit': opt_deficit,
        'baseline_total_power': baseline_total,
        'baseline_upstream_power': baseline_up,
        'baseline_downstream_power': baseline_down,
        'baseline_wake_deficit': baseline_deficit,
        'power_gain_MW': power_gain,
        'power_gain_percent': power_gain_pct,
        'turbine_spacing_D': turbine_spacing_D,
        'optimization_method': method_name,
        'all_results': results_history
    }


# =============================================================================
# Animation Creation
# =============================================================================
def create_wake_animation(predictions, output_path="wake_animation.gif", skip_frames=5):
    """Create an animation from wake flow predictions."""
    
    n_timesteps = predictions.shape[0]
    n_points = predictions.shape[1]
    
    # Calculate velocity magnitude for each timestep
    velocity_mag = np.linalg.norm(predictions, axis=2)
    
    # We'll create a 2D slice visualization
    # Assume the data is on a grid - we'll reshape if possible
    # For now, create a scatter-based visualization
    
    # Sample points for visualization (too many points would be slow)
    sample_size = min(5000, n_points)
    sample_idx = np.random.choice(n_points, sample_size, replace=False)
    
    # Get x, y coordinates (use velocity components as proxy if no grid info)
    # We'll create a simple 2D projection
    x_coords = predictions[0, sample_idx, 0]  # Ux at t=0 as x proxy
    y_coords = predictions[0, sample_idx, 1]  # Uy at t=0 as y proxy
    
    # Normalize coordinates for better visualization
    x_norm = np.linspace(0, 10, sample_size)  # Downstream distance (rotor diameters)
    y_norm = np.linspace(-2, 2, sample_size)  # Lateral distance
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Color normalization
    vmin, vmax = velocity_mag.min(), velocity_mag.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Initial scatter
    scatter = ax.scatter(x_norm, y_norm, c=velocity_mag[0, sample_idx], 
                        cmap='coolwarm', s=5, norm=norm)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Velocity Magnitude (m/s)')
    ax.set_xlabel('Downstream Distance (D)')
    ax.set_ylabel('Lateral Distance (D)')
    ax.set_title('Wind Turbine Wake Flow - Time: 0')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-3, 3)
    
    # Add turbine representation
    ax.axvline(x=0, color='black', linewidth=3, label='Turbine')
    ax.legend(loc='upper right')
    
    def update(frame):
        actual_frame = frame * skip_frames
        if actual_frame >= n_timesteps:
            actual_frame = n_timesteps - 1
        scatter.set_array(velocity_mag[actual_frame, sample_idx])
        ax.set_title(f'Wind Turbine Wake Flow - Timestep: {actual_frame}')
        return scatter,
    
    n_frames = n_timesteps // skip_frames
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
    
    # Save animation
    anim.save(output_path, writer='pillow', fps=10)
    plt.close(fig)
    
    return output_path


def create_velocity_slice_animation(predictions, output_path="wake_slice_animation.gif"):
    """Create a cleaner velocity field animation."""
    
    n_timesteps = predictions.shape[0]
    
    # Calculate velocity magnitude
    velocity_mag = np.linalg.norm(predictions, axis=2)
    
    # Create a synthetic 2D grid representation
    # Reshape the 1D spatial data into approximate 2D
    n_points = predictions.shape[1]
    
    # Try to create a reasonable grid
    grid_size = int(np.sqrt(n_points / 3))  # Approximate
    if grid_size < 50:
        grid_size = 50
    
    # Create interpolated 2D field
    x = np.linspace(0, 10, grid_size)  # Downstream (rotor diameters)
    y = np.linspace(-2, 2, grid_size)  # Lateral
    X, Y = np.meshgrid(x, y)
    
    # Sample velocity data and interpolate to grid
    sample_points = min(n_points, grid_size * grid_size)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Create synthetic wake pattern for visualization
    def create_wake_field(timestep, base_velocity=8.0):
        """Create a synthetic wake field pattern."""
        # Wake deficit model (simplified)
        wake_center = 0  # Centerline
        wake_expansion = 0.1  # Expansion rate
        
        # Time-varying fluctuation
        t_factor = np.sin(timestep * 0.2) * 0.1 + 1.0
        
        # Calculate wake deficit
        downstream_decay = np.exp(-X / 5) * 0.6
        lateral_profile = np.exp(-(Y ** 2) / (2 * (0.5 + wake_expansion * X) ** 2))
        
        # Mean velocity from actual predictions
        mean_vel = np.mean(velocity_mag[timestep])
        
        # Combine into velocity field
        velocity_field = mean_vel * (1 - downstream_decay * lateral_profile * t_factor)
        
        # Add some turbulent fluctuations
        velocity_field += np.random.randn(grid_size, grid_size) * 0.2
        
        return velocity_field
    
    # Initial plot
    vel_field = create_wake_field(0)
    vmin, vmax = velocity_mag.min(), velocity_mag.max()
    
    im = ax.contourf(X, Y, vel_field, levels=20, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label='Velocity Magnitude (m/s)')
    
    # Add turbine
    turbine_y = np.linspace(-0.5, 0.5, 10)
    ax.plot(np.zeros_like(turbine_y), turbine_y, 'k-', linewidth=5, label='Wind Turbine')
    ax.plot([0], [0], 'ko', markersize=10)
    
    ax.set_xlabel('Downstream Distance (Rotor Diameters)', fontsize=12)
    ax.set_ylabel('Lateral Distance (Rotor Diameters)', fontsize=12)
    ax.set_title('Wind Turbine Wake Flow Prediction - TT-OpInf Model', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-2, 2)
    
    plt.tight_layout()
    
    def update(frame):
        ax.clear()
        vel_field = create_wake_field(frame)
        im = ax.contourf(X, Y, vel_field, levels=20, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        
        # Re-add turbine
        ax.plot(np.zeros_like(turbine_y), turbine_y, 'k-', linewidth=5)
        ax.plot([0], [0], 'ko', markersize=10)
        
        ax.set_xlabel('Downstream Distance (Rotor Diameters)', fontsize=12)
        ax.set_ylabel('Lateral Distance (Rotor Diameters)', fontsize=12)
        ax.set_title(f'Wind Turbine Wake Flow - Timestep: {frame}/{n_timesteps-1}', fontsize=14)
        ax.set_xlim(-0.5, 10)
        ax.set_ylim(-2, 2)
        
        return [im]
    
    # Create animation with fewer frames for speed
    frame_skip = max(1, n_timesteps // 20)
    frames = list(range(0, n_timesteps, frame_skip))
    
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=200)
    anim.save(output_path, writer='pillow', fps=5)
    plt.close(fig)
    
    return output_path


# =============================================================================
# Main GUI
# =============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🌀 Wind Turbine Multi-Agent AI Analysis System  \n for Optimization and Predictive Maintenance</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('''
    <p style="text-align: center; font-size: 1.4rem; color: #666; margin-top: -10px;">
    📧  <b> Contact: mandar.tabib@sintef.no , SINTEF Digital | ⚠️ <em>Currently being tested/developed</em></b>
    </p>
    ''', unsafe_allow_html=True)
    

    st.markdown("""
    <div class="info-box">
    <b>Welcome!</b><br> 1. This system involves multiple agents (as cited below) involving ML/AI and LLMs to analyze wind turbine operations. <br> 2.This is a exploratory scientific research work. <br> 3. </b>The ML models are trained on simulation dataset and on publicly available data by mandar.tabib@sintef.no at SINTEF DIGITAL</b>.  <br>
    Regarding LLM: IF you want LLMs in this workflow as expert reviewers, as intelligent assistants or as turbine experts, <br> THEN expand the LLM Configuration section in the sidebar and select your provider/model! with your API KEY <br>
    <br>Application Mode:  Select the APPLICATION MODE ON LEFT SIDEBAR: PREDICTIVE MAINTENANCE OR WIND FARM OPTIMIZATION. <br>
        Wait For Models To Load When Selected - BOTH ML AND LLMS - This may Take a Few Moments 
    <br> Scroll Down to see the details of your selection as there is no auto-scroll. <br>
    
    <ul>
        <li><b>Agent 1:</b> Weather Station - Fetches real-time wind conditions</li>
        <li><b>Agent 2:</b> Turbine Experts-  A rule-based expert and the Large Language Models.</li>
         <ul>
            <li>&emsp;<b>Agent 2A:</b> Symbolic Rule-based Turbine Expert - Uses predefined rules for turbine analysis</li>
             <li>&emsp;<b>Agent 2B:</b> LLM-based Turbine Expert - Uses local LLM for intelligent recommendations</li>
            <li>&emsp;<b>Agent 2C:</b> Turbine Pair Selector - Rule-based to identify critical turbine pairs for wake optimization</li>
             <li>&emsp;<b>Agent 2D:</b> LLM Turbine Pair Selector - LLM to identify critical turbine pairs for wake optimization</li>
        </ul>  
        <li><b>Agent 3:</b> Optimizer Agent: Two-Turbine Wake Steering Optimizer - Finds optimal yaw misalignment for farm power maximization</li>
        <li><b>Agent 4:</b> Flow AI Agent: Wind Turbine Flow ROM Agent - Unsupervised Tensor Train Decomposition + Operator Inference model at SINTEF Digital</li>
        <li><b>Agent 5:</b> Power AI  - Gaussian Process Regressor trained at SINTEF Digital.</li>
        <li><b>Agent 6:</b> Predictive Maintenance AI Agent - A Semi-Supervised Learning Model for Health Indicator and RUL involving Autoencoder , Gaussian Mixture Model and Recurrent Neural Network.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.info(
                "**Important Notes:**\n\n"
                "1. Yaw misalignment ML training is upto 15 degrees. Beyond this, results may be less accurate.\n"
                "2. **NTNU VPN Required:** To use the LLM model at NTNU, you must be connected to the NTNU VPN.\n\n"
                "3. **Alternative Providers:** You can also try LLM with OpenAI or Anthropic using your own API key.\n\n"
                "4. **API Key Privacy:** Your API keys should be stored in your local `.env` file and never shared."
            )
    
    st.info(
               "**Internal reports (that will be published in peer-reviewed journals):**\n\n"   
                "1.  A Multi-Agent AI Framework Integrating Large Language Models and Machine Learning for Wind Turbine Predictive Maintenance and Wake Steering Optimisation. M. Tabib. Technical Report, SINTEF Digital, 2026. mandar.tabib@sintef.no\n\n"
                "2.  Gaussian process regression for wind turbine rotor power prediction under yaw misalignment. M. Tabib. Technical Report, SINTEF Digitala.\n\n"
                "3.  A semi-supervised machine learning framework for predictive maintenance of wind turbine gearbox systems using real SCADA data. M. Tabib. Technical Report, SINTEF Digital. \n\n"
                "4.  Tensor train decomposition with operator inference for parametric wind turbine wake flow prediction. M. Tabib. Technical Report,SINTEF Digital."
                 )
    st.markdown("""
    <div class="info-box">          
     <b> Select a wind farm on the left sidebar for agents to start analysis!</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        # GLOBAL LLM CONFIGURATION - Always visible
        st.header("🤖 LLM Configuration")
        
        # Initialize global config
        GlobalLLMConfig.initialize_session_state()
        
        with st.expander("⚙️ AI Model Settings", expanded=False):
            # Provider selection
            current_provider = GlobalLLMConfig.get_provider()
            provider_options = list(GlobalLLMConfig.PROVIDER_MODELS.keys())
            provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0
            
            new_provider = st.selectbox(
                "LLM Provider",
                options=provider_options,
                index=provider_index,
                help="Select the AI service provider",
                key="global_provider_select"
            )
            
            # Update provider if changed
            if new_provider != current_provider:
                GlobalLLMConfig.set_provider(new_provider)
                st.rerun()
            
            # Model selection
            available_models = GlobalLLMConfig.PROVIDER_MODELS[new_provider]
            current_model = GlobalLLMConfig.get_model()
            model_index = 0
            if current_model in available_models:
                model_index = available_models.index(current_model)
            
            new_model = st.selectbox(
                "Model",
                options=available_models,
                index=model_index,
                help=f"Select the {new_provider} model to use",
                key="global_model_select"
            )
            
            # Update model if changed
            if new_model != current_model:
                GlobalLLMConfig.set_model(new_model)
            
            # Temperature setting
            current_temperature = GlobalLLMConfig.get_temperature()
            new_temperature = st.slider(
                "Exploration vs Accuracy",
                min_value=0.0,
                max_value=1.0,
                value=current_temperature,
                step=0.1,
                help="0 = accuracy focused, 1 = exploration focused",
                key="global_temperature_select"
            )
            
            # Update temperature if changed
            if abs(new_temperature - current_temperature) > 0.01:
                GlobalLLMConfig.set_temperature(new_temperature)

            # API Key input - allow user to provide/override key from GUI
            if new_provider != "Ollama":
                st.markdown("---")
                st.markdown("**🔑 API Key:**")

                # Get the key from env/.env/secrets (pre-existing)
                _, env_api_key = get_api_config(new_provider)
                # Get any user-provided override
                user_key = GlobalLLMConfig.get_user_api_key(new_provider)

                # Determine the effective key and display state
                effective_key = user_key if user_key else env_api_key

                if effective_key and not user_key:
                    # Key found from env - show masked with info
                    masked = effective_key[:4] + "****" + effective_key[-4:] if len(effective_key) > 8 else "****"
                    st.caption(f"Auto-detected from environment: `{masked}`")

                key_name = GlobalLLMConfig.PROVIDER_API_KEY_NAMES.get(new_provider, "API_KEY")
                new_user_key = st.text_input(
                    f"{key_name}",
                    value=user_key,
                    type="password",
                    placeholder="Enter API key or leave blank to use .env",
                    help=f"Provide your {new_provider} API key here. It is stored in session memory only and never saved to disk.",
                    key=f"api_key_input_{new_provider}"
                )

                # Update user key if changed
                if new_user_key != user_key:
                    GlobalLLMConfig.set_user_api_key(new_provider, new_user_key)
                    st.rerun()

            # Connection test
            st.markdown("---")
            st.markdown("**🔍 Test Connection:**")
            test_col1, test_col2 = st.columns([1, 1])
            with test_col1:
                if st.button("🧪 Test", use_container_width=True, key="global_test_btn"):
                    with st.spinner("Testing..."):
                        result = test_llm_connection()
                        if result['success']:
                            st.success(result['message'])
                        else:
                            st.error(result['message'])
                            if result['error_details']:
                                st.caption(result['error_details'])

            with test_col2:
                # Status indicator
                api_base, api_key = GlobalLLMConfig.get_api_config()
                if api_key and api_key != "":
                    st.success("🔑 Configured")
                else:
                    if new_provider == "Ollama":
                        st.info("🔄 Local")
                    else:
                        st.warning("🔑 No API Key")
        
        st.markdown("---")
        
        # MODE SELECTOR - after LLM config
        st.header("🎯 Application Mode")
        app_mode = st.radio(
            "Select analysis mode:",
            options=[ "Predictive Maintenance","Wind Farm Optimization"],
            index=0 if st.session_state.app_mode == "Predictive Maintenance" else 1,
            help="Choose between multi-agent wind farm optimization or predictive maintenance inference"
        )
        
        # Update session state
        if app_mode != st.session_state.app_mode:
            st.session_state.app_mode = app_mode
            st.rerun()
        
        st.markdown("---")
        
        # =================================================================
        # PREDICTIVE MAINTENANCE MODE CONTROLS
        # =================================================================
        # PREDICTIVE MAINTENANCE MODE - Sidebar Controls
        # =================================================================
        if app_mode == "Predictive Maintenance":
            st.header("🏥 Predictive Maintenance")
            st.markdown("**Fuhrlander FL2500 (2.5MW)**")
            
            # Load models button
            if st.session_state.pdm_models is None:
                if st.button("📦 Load PdM Models", type="primary", use_container_width=True):
                    with st.spinner("Loading models..."):
                        result = load_pdm_models()
                        st.session_state.pdm_models = result
                        if result is not None:
                            st.rerun()
                        # If result is None, error is already displayed by load_pdm_models()
                
                st.info("Click to load the predictive maintenance models")
            else:
                # Verify models loaded successfully and contain required data
                if isinstance(st.session_state.pdm_models, dict) and 'test_data' in st.session_state.pdm_models:
                    st.success("✅ Models loaded")
                    
                    models_dict = st.session_state.pdm_models
                    test_data = models_dict['test_data']
                    
                    st.metric("Test Samples", f"{len(test_data['X_test']):,}")
                    st.caption("Turbines 83 & 84")
                else:
                    st.error("❌ Models loaded but test data is missing")
                    if st.button("🔄 Reload Models", use_container_width=True):
                        st.session_state.pdm_models = None
                        st.rerun()
                
                st.markdown("---")
                
                # Status indicator
                if st.session_state.pdm_results is not None:
                    st.success("✅ Inference complete")
                    n_analyzed = st.session_state.pdm_results['n_samples']
                    st.metric("Samples Analyzed", f"{n_analyzed:,}")
                    
                    if st.button("🔄 Reset Analysis", use_container_width=True):
                        st.session_state.pdm_results = None
                        st.rerun()
                else:
                    st.info("⚙️ Ready for inference")
                    st.caption("Configure and run in main area →")
        
        # =================================================================
        # WIND FARM OPTIMIZATION MODE CONTROLS
        # =================================================================
        else:  # Wind Farm Optimization mode
            st.header("🌪️ Wind Farm Optimization")
            st.markdown("Multi-agent AI-powered turbine optimization system")
            
            # Backward compatibility - sync global config with session state
            st.session_state.llm_provider = GlobalLLMConfig.get_provider()
            st.session_state.selected_model = GlobalLLMConfig.get_model()
            st.session_state.explore_vs_acc = GlobalLLMConfig.get_temperature()
            
            # Connection status indicator
            api_base, api_key = GlobalLLMConfig.get_api_config()
            current_provider = GlobalLLMConfig.get_provider()
            current_model = GlobalLLMConfig.get_model()
            
            st.info(f"**Active LLM:** {current_provider} - {current_model}")
            
            if api_key and api_key != "":
                st.success("✅ LLM configured and ready")
            else:
                if current_provider == "Ollama":
                    st.warning("⚠️ Using Ollama - ensure server is running")
                else:
                    st.error("❌ No API key configured - check global settings above")
            
            st.markdown("---")
            
            st.header("Norwegian Wind Farm Selection")
            st.markdown("Select a wind farm to analyze:")
            
            selected_farm = st.selectbox(
                "Wind Farm",
                options=list(NORWEGIAN_WIND_FARMS.keys()),
                index=0,  # Default to Bessaker
                help="Choose from major Norwegian wind farms"
            )
            
            # Show selected wind farm info in sidebar
            farm_info = NORWEGIAN_WIND_FARMS[selected_farm]
            st.markdown(f"**📍 Location:** {farm_info['location']}")
            st.markdown(f"**🌐 Latitude:** {farm_info['latitude']}")
            st.markdown(f"**🌐 Longitude:** {farm_info['longitude']}")
            st.markdown(f"**⚡ Capacity:** {farm_info['capacity_mw']} MW")
            st.markdown(f"**🌀 Turbines:** {farm_info['turbines']}")
            
            # Turbine Location Search
            st.markdown("---")
            st.header("🗺️ Turbine Locations")
            
            # Search for turbine locations when farm changes or button clicked
            search_turbines = st.button("🔍 Search Turbine Locations")
            
            if search_turbines or (selected_farm != st.session_state.last_searched_farm):
                with st.spinner(f"Searching for turbine locations at {selected_farm}..."):
                    search_results = search_turbine_locations(selected_farm, farm_info)
                    st.session_state.turbine_locations[selected_farm] = search_results
                    st.session_state.last_searched_farm = selected_farm
            
            # Display search results if available
            if selected_farm in st.session_state.turbine_locations:
                search_results = st.session_state.turbine_locations[selected_farm]
                
                # Status indicator
                status = search_results['search_status']
                if status == 'found_exact':
                    st.success(f"\u2705 Found {search_results['total_turbines_found']} exact locations")
                elif status == 'found_web':
                    st.info(f"\u2139\ufe0f Found {search_results['total_turbines_found']} locations from web")
                elif status == 'estimated':
                    st.warning(f"\u26a0\ufe0f Generated {search_results['total_turbines_found']} estimated locations")
                else:
                    st.error("\u274c No turbine location data available")
                
                st.markdown(f"**Data Source:** {search_results['data_source']}")
                
                # Show map button
                if search_results['turbine_locations']:
                    if st.button("📍 Show Turbine Map"):
                        st.session_state.show_turbine_map = True
            
            st.markdown("---")
            st.header("⚙️ Analysis Settings")
            
            # Agent Selection: Optional Agents
            st.markdown("**Optional Analysis Agents:**")
            col1, col2 = st.columns(2)
            with col1:
                enable_agent_2b = st.checkbox(
                    "🤖 Agent 2B (LLM Expert)",
                    value=False,
                    help="Enable LLM-based turbine control recommendations"
                )
                # Warn if enabling without successful connection test
                if enable_agent_2b:
                    if 'llm_test_result' not in st.session_state:
                        st.warning("⚠️ Connection not tested. Click 'Test Connection' in LLM Config above.")
                    elif not st.session_state.llm_test_result['success']:
                        st.error(f"❌ LLM connection failed! Agent 2B will not work.")
                        st.caption("Fix connection issue in LLM Configuration section")
            with col2:
                enable_agent_4 = st.checkbox(
                    "🌊 Agent 4 (Wake Flow ROM)",
                    value=False,
                    help="Enable wake flow simulation using ROM"
                )

            enable_agent_5 = st.checkbox(
                "⚡ Agent 5 (Power Predictor)",
                value=False,
                help="Enable power prediction using GP model"
            )
            
            # Store in session state
            st.session_state.enable_agent_2b = enable_agent_2b
            st.session_state.enable_agent_4 = enable_agent_4
            st.session_state.enable_agent_5 = enable_agent_5
            
            st.markdown("---")
            
            # Agent 2C/2D Selection
            st.markdown("**Turbine Pair Selection Agent:**")
            
            # Check if LLM connection was successfully tested
            llm_tested = 'llm_test_result' in st.session_state
            llm_success = llm_tested and st.session_state.llm_test_result.get('success', False)
            
            # Smart default: use 2C only if connection verified
            default_index = 0 if llm_success else 1
            
            agent_selection = st.radio(
                "Choose turbine pair identification method:",
                options=["Agent 2C (LLM-based)", "Agent 2D (Physical Wake Model)"],
                index=default_index,
                help="Agent 2C uses AI/LLM for intelligent analysis. Agent 2D uses physical wake models as backup when LLM is unavailable."
            )
            
            # Warn if selecting Agent 2C without successful test
            if "2C" in agent_selection:
                if not llm_tested:
                    st.warning("⚠️ LLM connection not tested. Agent 2C may fail.")
                    st.caption("Test connection in 'LLM Configuration' section above")
                elif not llm_success:
                    st.error(f"❌ LLM connection failed! Agent 2C will not work.")
                    st.info("💡 **Recommendation:** Use Agent 2D or fix LLM connection")
            
            # Store selection in session state
            st.session_state.selected_agent = "2C" if "2C" in agent_selection else "2D"
            
            if st.session_state.selected_agent == "2D":
                with st.expander("🔧 Agent 2D Wake Model Settings"):
                    wake_expansion_factor = st.slider(
                        "Wake Expansion Factor", 
                        min_value=0.05, max_value=0.2, value=0.1, step=0.01,
                        help="Controls how quickly wake expands downstream (typical: 0.05-0.15)"
                    )
                    min_influence_threshold = st.slider(
                        "Minimum Wake Influence (%)", 
                        min_value=1.0, max_value=20.0, value=5.0, step=1.0,
                        help="Minimum wake deficit percentage to consider significant"
                    ) / 100.0
                    
                    st.session_state.wake_expansion_factor = wake_expansion_factor
                    st.session_state.min_influence_threshold = min_influence_threshold
            
            st.markdown("---")
            
            # Agent 3 Optimizer Settings
            st.header("🎯 Agent 3: Optimizer Settings")
            
            # Optimization method selection
            st.markdown("**Optimization Method:**")
            opt_method = st.selectbox(
                "Select method for wake steering optimization:",
                options=['analytical_physics', 'ml_surrogate', 'grid_search', 'bayesian_transfer'], #'ml_wake_extraction' option.
                format_func=lambda x: {
                    'ml_surrogate': '🧠 ML Surrogate Auto-Diff',
                    'analytical_physics': '📐 Analytical Physics AD (Fastest)',
                   # 'ml_wake_extraction': '🎯 ML Wake Extraction (Expert Approach)',
                    'grid_search': '🔍 Grid Search (Brute-force)',
                    'bayesian_transfer': '🎯 Bayesian Optimization (Transfer Learning)'
                }[x],
                index=0,
                help="Choose optimization algorithm for yaw angle optimization"
            )
            
            # Store in session state
            st.session_state.opt_method = opt_method
            
            # Number of turbine pairs to optimize
            max_pairs_to_optimize = st.slider(
                "Max Turbine Pairs to Optimize",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="Select how many top turbine pairs to optimize (based on wake influence)"
            )
            st.session_state.max_pairs_to_optimize = max_pairs_to_optimize
            
            with st.expander("📖 Method Descriptions"):
                st.markdown("""
                **🧠 ML Surrogate AD:**
                - Fits polynomial surrogates to ML model predictions
                - Enables true gradient-based optimization through ML models
                - Most accurate but slower (~30-50 iterations)
                
                **📐 Analytical Physics AD:**
                - Uses Bastankhah-Porté-Agel wake model
                - Fast analytical gradients
                - Good balance of speed and accuracy
                
                **🎯 ML Wake Extraction (Expert Approach):**
                - Extracts vertical velocity profiles from ML wake predictions at 1700m
                - Calculates wake centerline and lateral migration
                - Extrapolates to actual turbine spacing using linear interpolation
                - Checks geometric wake-turbine overlap
                - Finds minimum yaw angle for wake avoidance
                - Data-driven: Uses location-specific wake characteristics from CFD model
                
                **🔍 Grid Search:**
                - Brute-force evaluation of all combinations
                - No gradients, guaranteed to find best in search space
                - Slowest for fine grids but most robust
                
                **🎯 Bayesian Optimization (Transfer Learning):**
                - Uses pre-trained GPR as base + residual GP for corrections
                - Expected Improvement (EI) with Matérn 5/2 kernel
                - Multi-start L-BFGS optimization of acquisition function
                - Constraint: 0-15° yaw misalignment (vs 12° for other methods)
                - Efficient exploration-exploitation trade-off (~15-20 iterations)
                """)
            
            st.markdown("---")
            
            n_timesteps = st.slider("Prediction Timesteps", min_value=10, max_value=100, value=50, step=10)
            export_vtk = st.checkbox("Export VTK Files", value=False)
            
            st.markdown("---")
            
            # Expert Reviewer Settings
            st.header("🎓 Expert Reviewer Agent")
            st.markdown("**AI-powered validation at all critical checkpoints**")
            
            enable_reviewer = st.checkbox(
                "🔍 Enable LLM Expert Reviewer",
                value=False,
                help="Expert LLM reviews outputs from all agents to catch errors and validate physical feasibility"
            )
            
            if enable_reviewer:
                reviewer_mode = st.radio(
                    "Reviewer Mode:",
                    options=["Advisory Only", "Blocking on Critical Issues"],
                    index=0,
                    help="**Advisory**: Provides feedback but never halts workflow. **Blocking**: Stops analysis on critical issues (e.g., optimization below cut-in wind speed)."
                )
                
                st.session_state.reviewer_enabled = True
                st.session_state.reviewer_mode = "advisory" if reviewer_mode == "Advisory Only" else "blocking"
                
                with st.expander("ℹ️ What does the Expert Reviewer do?"):
                    st.markdown("""
                    The Expert Reviewer is an LLM-based domain expert that validates analysis at 3 checkpoints:
                    
                    **Checkpoint 1 - After Agent 2 (Weather & Yaw):**
                    - ✓ Wind speed within operational range (3-25 m/s)
                    - ✓ Yaw angle recommendations valid
                    - ✓ Turbine pair alignment with wind direction
                    
                    **Checkpoint 2 - After Agent 3 (Power & Optimization):**
                    - ✓ Power predictions physically plausible (0-5 MW)
                    - ✓ Optimization only runs when turbine is operating
                    - ✓ Power gains realistic (<20%)
                    
                    **Checkpoint 3 - After Agent 4 (Wake Flow):**
                    - ✓ Wake flow predictions consistent with conditions
                    - ✓ Velocity fields physically reasonable
                    
                    **Final Review:**
                    - Synthesizes all findings into actionable recommendations
                    - Included in final report with severity levels
                    - Examples caught: "Optimization ran with 2 m/s wind (below cut-in)"
                    
                    **Mode Selection:**
                    - **Advisory**: Best for exploration and testing
                    - **Blocking**: Best for production/safety-critical operations
                    """)
            else:
                st.session_state.reviewer_enabled = False
            
            st.markdown("---")
            
            # What-If Analysis Mode
            st.header("🔬 What-If Analysis Mode")
            st.markdown("Override Agent 1 weather data with manual inputs for scenario testing")
            
            whatif_enabled = st.checkbox(
                "⚙️ Enable Manual Weather Override",
                value=st.session_state.get('whatif_mode_enabled', False),
                help="Bypass Agent 1 weather API and use manual wind conditions for demonstration"
            )
            
            if whatif_enabled:
                st.info("🔄 **What-If Mode Active**: Agent 1 weather fetch will be bypassed with your manual inputs")
                
                whatif_wind_speed = st.number_input(
                    "Wind Speed (m/s)",
                    min_value=3.5,
                    max_value=25.0,
                    value=st.session_state.get('whatif_wind_speed', 8.5),
                    step=0.5,
                    help="Must be between cut-in (3.5 m/s) and cut-out (25.0 m/s)"
                )
                
                whatif_wind_direction = st.number_input(
                    "Wind Direction (degrees)",
                    min_value=0.0,
                    max_value=360.0,
                    value=st.session_state.get('whatif_wind_direction', 270.0),
                    step=5.0,
                    help="0° = North, 90° = East, 180° = South, 270° = West"
                )
                
                # Store in session state
                st.session_state.whatif_mode_enabled = whatif_enabled
                st.session_state.whatif_wind_speed = whatif_wind_speed
                st.session_state.whatif_wind_direction = whatif_wind_direction
                
                # Validation warnings
                if whatif_wind_speed < 3.5:
                    st.warning(f"⚠️ Wind speed {whatif_wind_speed:.1f} m/s is barely above cut-in. Consider using ≥3.5 m/s for meaningful optimization.")
                elif whatif_wind_speed > 25.0:
                    st.error(f"⛔ Wind speed {whatif_wind_speed:.1f} m/s exceeds cut-out (25.0 m/s). Turbine will shut down.")
                
                #if whatif_wind_direction < 270 or whatif_wind_direction > 285:
                    #st.info(f"ℹ️ Wind direction {whatif_wind_direction:.0f}° is outside ROM training range (270-285°). Results may be less accurate.")
                
                # Show preview
                with st.expander("📊 Preview What-If Conditions"):
                    pcol1, pcol2 = st.columns(2)
                    with pcol1:
                        st.metric("Override Wind Speed", f"{whatif_wind_speed:.1f} m/s")
                    with pcol2:
                        st.metric("Override Wind Direction", f"{whatif_wind_direction:.0f}°")
            else:
                st.session_state.whatif_mode_enabled = False
            
            st.markdown("---")
            
            run_analysis = st.button("🚀 Run Analysis", type="primary")
            
            if st.session_state.analysis_complete:
                if st.button("🔄 Reset"):
                    st.session_state.analysis_complete = False
                    st.session_state.results = None
                    st.rerun()
    
    # =========================================================================
    # MAIN CONTENT AREA
    # =========================================================================
    
    # Handle Predictive Maintenance Mode
    if st.session_state.app_mode == "Predictive Maintenance":
        if st.session_state.pdm_results is not None:
            # Display results when available
            display_pdm_results(st.session_state.pdm_results, st.session_state.pdm_models)
        
        elif st.session_state.pdm_models is not None:
            # Models loaded but inference not yet run - show configuration interface
            # Verify models contain required data
            if not isinstance(st.session_state.pdm_models, dict) or 'test_data' not in st.session_state.pdm_models:
                st.error("❌ Models loaded but test data is missing. Please check RUL/saved_models directory.")
                st.info("Required files: `fuhrlander_fl2500_pm_models.joblib`, `test_data.npz`, `metadata.json`")
                if st.button("🔄 Reload Models"):
                    st.session_state.pdm_models = None
                    st.rerun()
                return
            
            st.markdown("### ⚙️ Configure Predictive Maintenance Inference")
            
            test_data = st.session_state.pdm_models['test_data']
            total_samples = len(test_data['X_test'])
            
            st.success(f"✅ Models loaded successfully! **{total_samples:,} test samples** available from Fuhrlander turbines 83 & 84.")
            
            st.markdown("---")
            
            # Create tabs for better organization
            config_tab, info_tab = st.tabs(["🔬 Run Inference", "📚 Model Information"])
            
            with config_tab:
                st.markdown("#### Select Test Data for Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    n_samples = st.slider(
                        "📊 Number of samples for inference",
                        min_value=100,
                        max_value=min(5000, total_samples),
                        value=1000,
                        step=100,
                        help="Select how many consecutive test samples to analyze"
                    )
                
                with col2:
                    sample_offset = st.slider(
                        "🎯 Starting sample index",
                        min_value=0,
                        max_value=max(0, total_samples - n_samples),
                        value=0,
                        step=100,
                        help="Choose where to start in the test dataset"
                    )
                
                st.info(f"""
                **Selection Summary:**
                - Analyzing samples {sample_offset:,} to {sample_offset + n_samples:,} (out of {total_samples:,} total)
                - Time span: ~{n_samples} hours of turbine operation data
                - Turbines: 83 & 84 (Fuhrlander FL2500)
                """)
                
                st.markdown("---")
                
                # Prominent run button
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("🚀 Run Predictive Maintenance Inference", type="primary", use_container_width=True):
                        with st.spinner("🔄 Running inference on all 5 models (Autoencoder, GMM, Classifiers, LSTM)..."):
                            try:
                                pdm_results = run_pdm_inference(
                                    st.session_state.pdm_models,
                                    n_samples=n_samples,
                                    sample_offset=sample_offset
                                )
                                st.session_state.pdm_results = pdm_results
                                st.success("✅ Inference complete! View results below.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Inference failed: {str(e)}")
                                import traceback
                                with st.expander("🔍 Error Details"):
                                    st.code(traceback.format_exc())
                
                st.markdown("---")
                st.markdown("""
                **What happens when you run inference:**
                1. 🧠 **Health Indicator (HI)** - Autoencoder reconstructs data to detect anomalies
                2. 🎯 **Health States** - GMM clusters HI into Healthy/Degrading/Critical states
                3. ⚠️ **Binary Classification** - Predicts presence/absence of faults
                4. 🔍 **Multi-class Classification** - Identifies specific fault types
                5. ⏱️ **RUL Prediction** - LSTM estimates remaining useful life in days
                6. 📊 **Feature Importance** - SHAP explains which SCADA features drive predictions
                """)
            
            with info_tab:
                st.markdown("#### 📚 Model Architecture Details")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("""
                    **Training Data:**
                    - Turbines: 80, 81, 82
                    - Samples: 53,810 hourly records
                    - Period: Healthy + degradation phases
                    
                    **Test Data:**
                    - Turbines: 83, 84
                    - Samples: 35,411 hourly records
                    - Purpose: Unseen turbine validation
                    """)
                
                with col_b:
                    st.markdown("""
                    **Features (27 total):**
                    - **SCADA (18):** Power, wind speed, temperatures, vibrations, blade angles
                    - **Engineered (9):** Rolling statistics, power efficiency, load factors
                    
                    **Models:**
                    - Autoencoder: Anomaly detection
                    - GMM: State clustering
                    - GradientBoosting: Binary fault classifier
                    - RandomForest: Multi-class fault classifier
                    - LSTM: Sequential RUL predictor
                    """)
                
                st.markdown("---")
                st.markdown("#### 🎯 Prediction Targets")
                st.markdown("""
                | Model | Output | Interpretation |
                |-------|--------|----------------|
                | Autoencoder | Health Indicator (0-1) | 0 = Healthy, 1 = Anomalous |
                | GMM | Health State (0-2) | 0 = Healthy, 1 = Degrading, 2 = Critical |
                | Binary Classifier | Fault (0-1) | 0 = No Fault, 1 = Fault Present |
                | Multi-class Classifier | Fault Type (0-N) | Specific component failure modes |
                | LSTM | RUL (days) | Estimated days until failure |
                """)
        
        else:
            # Welcome screen for PdM mode - no models loaded yet
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### 🏥 Predictive Maintenance Module")
                st.info("""
                **Fuhrlander FL2500 (2.5MW) Turbine Health Monitoring**
                
                This module provides comprehensive predictive maintenance analysis:
                
                📊 **Health Indicator (HI)**: Autoencoder-based anomaly detection  
                🎯 **Health States**: GMM classification (Healthy/Degrading/Critical)  
                ⚠️ **Fault Probability**: Binary and multi-class fault prediction  
                ⏱️ **Remaining Useful Life**: LSTM-based RUL estimation  
                🔍 **Feature Importance**: SHAP-based explainability
                
                👈 **Get Started:** Load the PdM models from the sidebar to begin.
                """)
                
                st.markdown("---")
                st.markdown("#### 📚 Model Information")
                st.write("""
                - **Training Data**: Turbines 80, 81, 82 (53,810 hourly samples)
                - **Test Data**: Turbines 83, 84 (35,411 hourly samples)
                - **Features**: 27 (18 SCADA + 9 engineered)
                - **Models**: Autoencoder, GMM, GradientBoosting, RandomForest, LSTM
                """)
    
    # Handle Wind Farm Optimization Mode
    elif st.session_state.app_mode == "Wind Farm Optimization":
        # =========================================================================
        # TURBINE LOCATION MAP SECTION
        # =========================================================================
        # Display turbine map if requested
        if hasattr(st.session_state, 'show_turbine_map') and st.session_state.show_turbine_map:
            if selected_farm in st.session_state.turbine_locations:
                search_results = st.session_state.turbine_locations[selected_farm]
                
                st.markdown("---")
                st.markdown(f"### 📍 Turbine Locations Map - {selected_farm}")
            
            # Create map data
            turbine_map_data = create_turbine_map(farm_info, search_results['turbine_locations'])
            
            if turbine_map_data is not None:
                # Get wind direction from weather data if available
                wind_direction = None
                wind_speed = None
                if 'results' in st.session_state and st.session_state.results:
                    weather_data = st.session_state.results.get('weather', {})
                    wind_direction = weather_data.get('wind_direction_deg')
                    wind_speed = weather_data.get('wind_speed_ms')
                
                wind_info = ""
                if wind_direction is not None and wind_speed is not None:
                    wind_info = f"\n- **Wind Conditions:** {wind_speed:.1f} m/s from {wind_direction:.0f}°"
                
                st.markdown(f"""
                **Map Information:**
                - Wind Farm: **{selected_farm}**
                - Turbines Displayed: **{len(turbine_map_data)}**
                - Data Source: **{search_results['data_source']}**
                - Search Status: **{search_results['search_status']}**{wind_info}
                """)
                
                # Prepare data for pydeck
                turbine_map_data['turbine_label'] = 'T' + turbine_map_data['turbine_id'].astype(str)
                
                # Calculate optimal zoom level based on turbine spread
                lat_spread = turbine_map_data['lat'].max() - turbine_map_data['lat'].min()
                lon_spread = turbine_map_data['lon'].max() - turbine_map_data['lon'].min()
                max_spread = max(lat_spread, lon_spread)
                
                # Calculate zoom level (higher zoom for smaller spread)
                if max_spread < 0.01:  # Very close turbines
                    optimal_zoom = 14
                elif max_spread < 0.05:  # Close turbines  
                    optimal_zoom = 13
                elif max_spread < 0.1:   # Medium spread
                    optimal_zoom = 12
                else:                    # Wide spread
                    optimal_zoom = 11
                
                # Create turbine scatter layer with adaptive sizing
                turbine_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=turbine_map_data,
                    get_position=["lon", "lat"],
                    get_fill_color=[0, 120, 255, 200],
                    get_radius=100,
                    radius_min_pixels=15,  # Minimum size when zoomed out
                    radius_max_pixels=100, # Maximum size when zoomed in
                    pickable=True,
                    auto_highlight=True,
                )
                
                # Create text layer for turbine numbers with adaptive sizing
                text_layer = pdk.Layer(
                    "TextLayer",
                    data=turbine_map_data,
                    get_position=["lon", "lat"],
                    get_text="turbine_label",
                    get_color=[255, 255, 255, 255],
                    get_size=16,
                    size_min_pixels=12,    # Minimum text size when zoomed out
                    size_max_pixels=24,    # Maximum text size when zoomed in
                    get_alignment_baseline="'center'",
                    get_text_anchor="'middle'",
                )
                
                layers = [turbine_layer, text_layer]
                
                # Add wind direction arrow if available
                if wind_direction is not None:
                    # Position wind arrow near the turbine cluster center
                    center_lat = turbine_map_data['lat'].mean()  # Use turbine center instead of farm center
                    center_lon = turbine_map_data['lon'].mean()
                    
                    # Arrow length in degrees (larger for better visibility)
                    arrow_length = max(0.02, max_spread * 0.5)  # Scale with turbine spread
                    
                    # Convert wind direction to arrow direction (wind blows FROM wind_direction TO opposite)
                    # Arrow should point in the direction wind is BLOWING TO
                    arrow_direction = (wind_direction + 180) % 360
                    arrow_rad = np.radians(arrow_direction)
                    
                    # Calculate arrow end point
                    end_lat = center_lat + arrow_length * np.cos(arrow_rad)
                    end_lon = center_lon + arrow_length * np.sin(arrow_rad) / np.cos(np.radians(center_lat))
                    
                    wind_arrow_data = pd.DataFrame([{
                        'start_lat': center_lat,
                        'start_lon': center_lon,
                        'end_lat': end_lat,
                        'end_lon': end_lon,
                        'wind_dir': wind_direction
                    }])
                    
                    # Create arrow layer using LineLayer with better visibility
                    arrow_layer = pdk.Layer(
                        "LineLayer",
                        data=wind_arrow_data,
                        get_source_position=["start_lon", "start_lat"],
                        get_target_position=["end_lon", "end_lat"],
                        get_color=[255, 0, 0, 255],  # Full opacity
                        get_width=10,
                        width_min_pixels=5,
                        width_max_pixels=15,
                    )
                    
                    # Add larger arrow head using ScatterplotLayer
                    arrow_head_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=wind_arrow_data,
                        get_position=["end_lon", "end_lat"],
                        get_fill_color=[255, 0, 0, 255],  # Full opacity
                        get_radius=150,
                        radius_min_pixels=8,
                        radius_max_pixels=25,
                        pickable=False,
                    )
                    
                    # Add text label for wind direction (positioned offset from turbines)
                    wind_label_data = pd.DataFrame([{
                        'lat': center_lat + arrow_length * 0.8,  # Position at arrow end
                        'lon': center_lon + arrow_length * 0.2,
                        'label': f'Wind: {wind_direction:.0f}° ({wind_speed:.1f} m/s)'
                    }])
                    
                    wind_text_layer = pdk.Layer(
                        "TextLayer",
                        data=wind_label_data,
                        get_position=["lon", "lat"],
                        get_text="label",
                        get_color=[255, 0, 0, 255],
                        get_size=16,
                        size_min_pixels=14,
                        size_max_pixels=20,
                        get_alignment_baseline="'center'",
                        get_text_anchor="'middle'",
                        background=True,
                        get_background_color=[255, 255, 255, 220],
                    )
                    
                    layers.extend([arrow_layer, arrow_head_layer, wind_text_layer])
                
                # Create the deck with optimal view
                view_state = pdk.ViewState(
                    latitude=turbine_map_data['lat'].mean(),  # Center on turbines
                    longitude=turbine_map_data['lon'].mean(),
                    zoom=optimal_zoom,
                    pitch=0,
                )
                
                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        "text": "Turbine {turbine_id}\nLat: {lat:.6f}\nLon: {lon:.6f}\nType: {type}"
                    },
                    map_style="road",
                )
                
                # Display the enhanced map
                st.pydeck_chart(deck)
                
                # Add legend
                st.markdown("""
                **Map Legend:**
                - 🔵 **Blue circles with numbers** = Turbine locations (T1, T2, ...)
                - 🔴 **Red arrow** = Wind direction (arrow points in direction wind is blowing)
                - **Hover over turbines** for detailed information
                """)
                
                # Show turbine details in expandable section
                with st.expander(f"📋 Turbine Details ({len(turbine_map_data)} turbines)"):
                    st.dataframe(
                        turbine_map_data[['turbine_id', 'lat', 'lon', 'type']],
                        column_config={
                            "turbine_id": st.column_config.NumberColumn("Turbine ID", help="Unique turbine identifier used by agents"),
                            "lat": st.column_config.NumberColumn("Latitude", format="%.6f"),
                            "lon": st.column_config.NumberColumn("Longitude", format="%.6f"),
                            "type": st.column_config.TextColumn("Data Type")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                # Close map button
                if st.button("❌ Close Map"):
                    st.session_state.show_turbine_map = False
                    st.rerun()
            else:
                st.error("Unable to create map - no valid turbine location data.")
        
        # =========================================================================
        # ML MODEL DEMONSTRATION SECTION
        # =========================================================================
        st.markdown("---")
        st.markdown("### 🔬 ML Model Demonstration: Explore Wake & Power vs Yaw Misalignment")
        st.markdown('''
        <p style="font-size: 0.95rem; color: #666; margin-top: -10px;">
        <b>Interact with the trained ML models</b> to see how yaw misalignment affects wake flow and power output.
        Adjust the slider below to explore the model predictions.
        </p>
        ''', unsafe_allow_html=True)
        
        # User input for yaw misalignment
        #demo_col1, demo_col2 = st.columns([1, 2])
        with st.container():
            st.markdown("""
            **ML Models Developed by SINTEF Digital (mandar.tabib@sintef.no):**
            
            | Model | Description | Input | Output |
            |-------|-------------|-------|--------|
            | **TT-OpInf ROM** | Tensor Train Decomposition + Operator Inference | Yaw Direction (0\u00b0-15\u00b0) |  Wake Velocity Field |
            | **GP Regressor** | Gaussian Process trained on CFD data | Yaw Direction (0\u00b0-15\u00b0)  | Rotor Power (MW) with uncertainty |
           
            *Models trained at SINTEF on high-fidelity CFD simulations of NREL 5MW turbine.*
            """)
        with st.container(): #demo_col1:
            user_yaw_misalignment = st.slider(
                "🎚️ Yaw Misalignment (degrees)",
                min_value=0.0,
                max_value=15.0,
                value=0.0,
            step=1.0,
            help="0° = aligned with wind, 15° = maximum misalignment"
        )
        
        # Convert to nacelle direction for ML models
        user_nacelle_direction = yaw_misalignment_to_nacelle_direction(user_yaw_misalignment)
        
        st.markdown(f"""
        **Current Settings:**
        - Yaw Misalignment: **{user_yaw_misalignment:.0f}°**      
        """)
        #  - Nacelle Direction: **{user_nacelle_direction:.0f}°**
        # Expected power loss due to misalignment
        cos_loss = np.cos(np.radians(user_yaw_misalignment)) ** 3
        st.metric(
            "Expected Power Factor", 
            f"{cos_loss*100:.1f}%",
            delta=f"{(cos_loss-1)*100:.1f}%" if user_yaw_misalignment > 0 else None
        )
        
        run_demo = st.button("🔄 Run Model Prediction", type="secondary")
        
        
        #Nacelle Direction (270°-285°) for ML models AS proxy for yaw direction.
        # Run demonstration when button is clicked or slider changes
        if run_demo or ('demo_results' not in st.session_state):
            with st.spinner("Running ML model predictions..."):
                try:
                    power_agent, wake_agent = load_agents()
                    
                    # Store demo results in session state
                    st.session_state.demo_yaw = user_yaw_misalignment
                    st.session_state.demo_nacelle = user_nacelle_direction
                    
                    # ============================================================
                    # Power Prediction (GP Model)
                    # ============================================================
                    power_result = power_agent.predict(
                        yaw_angle=user_nacelle_direction,
                        n_time_points=30,
                        return_samples=True,
                        n_samples=20
                    )
                    
                    # ============================================================
                    # Wake Flow Prediction (TT-OpInf ROM)
                    # ============================================================
                    wake_predictions, _ = wake_agent.predict(
                        yaw_angle=user_nacelle_direction,
                        n_timesteps=30,
                        export_vtk=False,
                        verbose=False
                    )
                    
                    st.session_state.demo_power = power_result
                    st.session_state.demo_wake = wake_predictions
                    st.session_state.demo_results = True
                    
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    st.session_state.demo_results = False
        
        # Display demonstration results
        if st.session_state.get('demo_results', False):
            st.markdown("---")
            
            demo_tab1, demo_tab2 = st.tabs([
                "⚡ Power Prediction (GP Regressor)", 
                "🌊 Wake Flow (TT-OpInf ROM)"
            ])
            
            with demo_tab1:
                st.markdown(f"#### Gaussian Process Power Prediction at Yaw Misalignment = {st.session_state.get('demo_yaw', 0):.0f}°")
                
                power_result = st.session_state.get('demo_power', {})
                
                if power_result:
                    power_time_series = power_result['power_mean_MW']
                    std_time_series = power_result['power_std_MW']
                    t = power_result['normalized_time']
                    
                    # Use time range 0.1 to 0.9 for statistics
                    valid_idx = (t >= 0.1) & (t <= 0.9)
                    t_valid = t[valid_idx]
                    
                    # Calculate statistics from power_mean_MW vector (analytical GP mean)
                    mean_power = np.mean(power_time_series[valid_idx])
                    min_power = np.min(power_time_series[valid_idx])
                    max_power = np.max(power_time_series[valid_idx])
                    power_variation = max_power - min_power
                    
                    # Display statistics from analytical GP mean (t=0.1-0.9)
                    st.markdown("**📈 Power Statistics from Analytical GP Mean (t=0.1-0.9):**")
                    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                    with pcol1:
                        st.metric("Time-Averaged", f"{mean_power:.3f} MW", help="np.mean(power_mean_MW)")
                    with pcol2:
                        st.metric("Min Power", f"{min_power:.3f} MW", help="np.min(power_mean_MW)")
                    with pcol3:
                        st.metric("Max Power", f"{max_power:.3f} MW", help="np.max(power_mean_MW)")
                    with pcol4:
                        st.metric("Variation (ΔP)", f"{power_variation:.3f} MW", help="Max - Min")
                    
                    # Power time series plot (t=0.1-0.9)
                    fig_power, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                    
                    # Left plot: Time series with samples (t=0.1-0.9)
                    if 'samples' in power_result and power_result['samples'] is not None:
                        samples = power_result['samples']
                        samples_valid = samples[:, valid_idx]
                        for i in range(min(15, samples.shape[0])):
                            ax1.plot(t_valid, samples_valid[i], 'steelblue', alpha=0.12, linewidth=0.7,
                                    label='GP Posterior Samples' if i == 0 else '')
                    
                    ax1.plot(t_valid, power_time_series[valid_idx], 'darkblue', linewidth=2.5, 
                            label='GP Mean Prediction (power_mean_MW)')
                    ax1.fill_between(t_valid, power_result['power_lower_95_MW'][valid_idx], 
                                    power_result['power_upper_95_MW'][valid_idx],
                                   alpha=0.15, color='gray', label='95% CI')
                    
                    ax1.axhline(y=mean_power, color='red', linestyle='--', linewidth=2, alpha=0.8,
                               label=f'Time-Average of GP Mean: {mean_power:.3f} MW')
                    
                    ax1.set_xlabel('Normalized Time')
                    ax1.set_ylabel('Power (MW)')
                    ax1.set_title(f'GP Power Prediction (t=0.1-0.9)\nYaw Misalignment = {st.session_state.get("demo_yaw", 0):.0f}°')
                    ax1.legend(loc='best', fontsize=8)
                    ax1.grid(True, alpha=0.3)
                    
                    # Right plot: Power distribution histogram (using valid data)
                    ax2.hist(power_time_series[valid_idx], bins=20, density=True, alpha=0.7, color='steelblue', 
                            edgecolor='black', label='Power distribution')
                    ax2.axvline(x=mean_power, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_power:.3f} MW')
                    ax2.axvline(x=min_power, color='green', linestyle='--', linewidth=1.5, label=f'Min: {min_power:.3f} MW')
                    ax2.axvline(x=max_power, color='orange', linestyle='--', linewidth=1.5, label=f'Max: {max_power:.3f} MW')
                    ax2.set_xlabel('Power (MW)')
                    ax2.set_ylabel('Probability Density')
                    ax2.set_title(f'Power Distribution (t=0.1-0.9)\n(ΔP = {power_variation:.3f} MW)')
                    ax2.legend(loc='best', fontsize=8)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig_power)
                    plt.close()
                    
                    st.markdown(f"""
                    **📊 Plot Legend:**
                    | Line Type | Description | Computation |
                    |-----------|-------------|-------------|
                    | **Light blue lines** | GP posterior samples | Random draws from GP distribution (samples array) |
                    | **Dark blue line** | GP mean prediction | `power_mean_MW` - GP's expected value at each time point |
                    | **Red dashed line** | Time-average of dark blue line | `np.mean(power_mean_MW)` = {mean_power:.3f} MW |
                    
                    **Statistics Computed (t=0.1-0.9):**
                    - Time-Averaged = `np.mean(power_mean_MW)` = {mean_power:.3f} MW
                    - Min Power = `np.min(power_mean_MW)` = {min_power:.3f} MW  
                    - Max Power = `np.max(power_mean_MW)` = {max_power:.3f} MW
                    - Variation = Max - Min = {power_variation:.3f} MW
                    """)
            
            with demo_tab2:
                st.markdown(f"#### TT-OpInf Wake Flow ROM at Yaw Misalignment = {st.session_state.get('demo_yaw', 0):.0f}°")
                
                wake_predictions = st.session_state.get('demo_wake', None)
                
                if wake_predictions is not None:
                    velocity_mag = np.linalg.norm(wake_predictions, axis=2)
                    
                    wcol1, wcol2, wcol3 = st.columns(3)
                    with wcol1:
                        st.metric("Spatial Points", f"{wake_predictions.shape[1]:,}")
                    #with wcol2:
                        #st.metric("Mean Velocity", f"{np.mean(velocity_mag):.2f} m/s")
                    with wcol3:
                        st.metric("Timesteps", wake_predictions.shape[0])
                    
                    # Create wake visualization
                    try:
                        from wake_animation import create_wake_contour_animation
                        
                        demo_anim_path = os.path.join(SCRIPT_DIR, f"wake_demo_yaw_{st.session_state.get('demo_yaw', 0):.0f}.gif")
                        grid_path = str(SCRIPT_DIR / "data" / "Grid_data.vtk")
                        
                        create_wake_contour_animation(
                            predictions=wake_predictions,
                            grid_path=grid_path,
                            output_path=demo_anim_path,
                            fps=8,
                            frame_skip=max(1, wake_predictions.shape[0] // 15),
                            cmap="RdYlBu_r",
                            verbose=False,
                            yaw_misalignment=st.session_state.get('demo_yaw', 0)
                        )
                        
                        st.image(demo_anim_path, caption=f"Wake Flow | Yaw Misalignment = {st.session_state.get('demo_yaw', 0):.0f}° | Nacelle = {st.session_state.get('demo_nacelle', 270):.0f}°")
                        
                    except Exception as e:
                        # Fallback: show velocity statistics
                        st.warning(f"Animation unavailable: {e}")
                        
                        fig_wake, ax = plt.subplots(figsize=(10, 4))
                        
                        # Plot velocity magnitude over time for a few sample points
                        n_samples = min(100, velocity_mag.shape[1])
                        sample_idx = np.random.choice(velocity_mag.shape[1], n_samples, replace=False)
                        
                        for i in sample_idx[:10]:
                            ax.plot(velocity_mag[:, i], alpha=0.3, linewidth=0.5)
                        
                        ax.plot(np.mean(velocity_mag, axis=1), 'r-', linewidth=2, label='Mean velocity')
                        ax.fill_between(
                            range(velocity_mag.shape[0]),
                            np.percentile(velocity_mag, 5, axis=1),
                            np.percentile(velocity_mag, 95, axis=1),
                            alpha=0.2, color='gray', label='5-95 percentile'
                        )
                        
                        # Add turbine marker with rotation based on yaw misalignment
                        yaw_misalignment = st.session_state.get('demo_yaw', 0)  # Get yaw misalignment
                        turbine_marker = plt.Line2D([0], [0], color='black', linewidth=5, label='Turbine')
                        ax.add_line(turbine_marker)
                        rotate_line(turbine_marker, -yaw_misalignment, ax)
                        # Apply rotation transformation
                        #from matplotlib.transforms import Affine2D
                        #rotation = Affine2D().rotate_deg(-yaw_misalignment)  # Rotate clockwise by yaw_misalignment
                        #turbine_marker.set_transform(rotation + ax.transData)

                        ax.set_xlabel('Timestep')
                        ax.set_ylabel('Velocity Magnitude (m/s)')
                        ax.set_title(f'Wake Velocity Evolution | Yaw Misalignment = {yaw_misalignment:.0f}°')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_wake)
                        plt.close()
            
            st.markdown("---")
        
        # Main content area - Wind Farm Analysis flow
        if run_analysis and selected_farm:
            run_full_analysis(selected_farm, n_timesteps, export_vtk)
        
        elif st.session_state.analysis_complete and st.session_state.results:
            display_results(st.session_state.results)
        
        else:
            # Welcome screen
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Windmills_D1-D4_%28Thornton_Bank%29.jpg/1280px-Windmills_D1-D4_%28Thornton_Bank%29.jpg", 
                        caption="Offshore Wind Turbines")
                
                st.info("👈 Enter the wind turbine location in the sidebar and click **Run Analysis** to begin.")


def run_reviewer_checkpoint(reviewer_agent, checkpoint_name, review_func, *args, **kwargs):
    """
    Helper function to run reviewer checkpoints and handle async calls.
    
    Args:
        reviewer_agent: WindTurbineReviewerAgent instance or None
        checkpoint_name: Name of checkpoint for display
        review_func: Async review function to call
        *args, **kwargs: Arguments to pass to review_func
    
    Returns:
        Review dict or None if reviewer not enabled
    """
    if reviewer_agent is None:
        return None
    
    try:
        with st.spinner(f"🎓 Expert Reviewer analyzing {checkpoint_name}..."):
            # Run async function in event loop
            review = asyncio.run(review_func(*args, **kwargs))
            
            # Check if workflow should continue
            if not review.get("allow_continue", True):
                severity = review.get("severity", "unknown")
                summary = review.get("summary", "Critical issues detected")
                
                st.error(f"🔴 **Review Failed**: {summary}")
                st.error(f"**Severity**: {severity}")
                
                # Show findings
                findings = review.get("findings", [])
                for finding in findings:
                    if finding.get("type") == "critical":
                        st.error(f"🔴 {finding.get('message', '')}")
                
                st.warning("⚠️ **Workflow halted by ExpertReviewer in blocking mode.** Fix issues and try again.")
                return review
            
            # Display review summary (non-blocking)
            severity = review.get("severity", "info")
            if severity == "critical" and review.get("allow_continue"):
                st.warning(f"⚠️ Expert Reviewer found critical issues in {checkpoint_name} (advisory mode - continuing)")
            elif severity == "warning":
                st.info(f"ℹ️ Expert Reviewer found warnings in {checkpoint_name}")
            
            return review
            
    except Exception as e:
        st.warning(f"⚠️ Expert review failed for {checkpoint_name}: {e}")
        return {"checkpoint": checkpoint_name, "status": "error", "error": str(e), "allow_continue": True}


def run_full_analysis(selected_farm: str, n_timesteps: int, export_vtk: bool):
    """Run the complete multi-agent analysis with GUI updates."""
    
    # Get wind farm info
    farm_info = NORWEGIAN_WIND_FARMS[selected_farm]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "location": f"{selected_farm}, {farm_info['location']}",
        "wind_farm": selected_farm,
        "farm_info": farm_info,
        "reviews": {}  # Store expert reviews
    }
    
    # Initialize reviewer agent if enabled
    reviewer_agent = None
    if st.session_state.get('reviewer_enabled', False) and REVIEWER_AGENT_AVAILABLE:
        try:
            # Load config
            config_path = os.path.join(SCRIPT_DIR, "config.yaml")
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
            
            reviewer_mode = st.session_state.get('reviewer_mode', 'advisory')
            reviewer_agent = WindTurbineReviewerAgent(
                config=config,
                mode=reviewer_mode,
                enabled=True
            )
            st.info(f"🎓 Expert Reviewer initialized in **{reviewer_mode}** mode")
        except Exception as e:
            st.warning(f"⚠️ Could not initialize Expert Reviewer: {e}. Continuing without reviewer.")
            reviewer_agent = None
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # =========================================================================
    # AGENT 1: Weather Station : to fetch wind data at the wind farm location
    # =========================================================================
    st.markdown("### 🌤️ Agent 1: Weather Station: to fetch wind data at the wind farm location.")
    
    with st.spinner(f"Fetching weather data for {selected_farm}..."):
        status_text.text("Agent 1: Connecting to weather service...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # Check if What-If mode is enabled
        if st.session_state.get('whatif_mode_enabled', False):
            # Override with manual inputs
            weather = {
                "location": f"{selected_farm}, {farm_info['location']}",
                "latitude": farm_info["latitude"],
                "longitude": farm_info["longitude"],
                "wind_speed_ms": st.session_state.whatif_wind_speed,
                "wind_direction_deg": st.session_state.whatif_wind_direction,
                "temperature_c": 10.0,  # Default reasonable value
                "data_source": "What-If Manual Override",
                "farm_name": selected_farm
            }
            yr_weather = {
                "wind_speed_ms": st.session_state.whatif_wind_speed,
                "wind_direction_deg": st.session_state.whatif_wind_direction,
                "data_source": "What-If Manual Override (yr.no display)"
            }
            st.info("🔬 **What-If Mode Active**: Using manual weather inputs instead of Agent 1 API fetch")
        else:
            # Normal path: fetch from API
            weather = fetch_weather_for_farm(selected_farm, farm_info)
            # Agent 1A: Fetch weather from yr.no (Met.no)
            yr_weather = fetch_weather_yr_no(farm_info["latitude"], farm_info["longitude"])
        
        results["weather"] = weather
        progress_bar.progress(20)
        results["yr_weather"] = yr_weather
    
    # Display wind farm information with tabs
    tab1, tab2 = st.tabs(["🏭 Wind Farm Info", "🌍 Weather Data"])
    
    with tab1:
        st.success(f"**{selected_farm}**")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(f"📍 **Location:** {farm_info['location']}, Norway")
            st.markdown(f"🌐 **Coordinates:** {farm_info['latitude']:.4f}°N, {farm_info['longitude']:.4f}°E")
        with col_info2:
            st.markdown(f"⚡ **Installed Capacity:** {farm_info['capacity_mw']} MW")
            st.markdown(f"🌀 **Number of Turbines:** {farm_info['turbines']}")
        st.info(f"ℹ️ {farm_info['description']}")
    
    with tab2:
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            st.markdown("#### Agent 1: Open-Meteo Weather Data")
            st.write(results["weather"])
            st.metric("🧭 Wind Direction", f"{results['weather']['wind_direction_deg']:.0f}°")
            st.metric("💨 Wind Speed", f"{results['weather']['wind_speed_ms']:.1f} m/s")
            st.metric("🌡️ Temperature", f"{results['weather']['temperature_c']:.1f}°C")
            st.caption(f"Data source: {results['weather']['data_source']}")
            
            # Show What-If mode indicator
            if st.session_state.get('whatif_mode_enabled', False):
                st.warning("🔬 **WHAT-IF MODE ACTIVE**: Weather data manually overridden. Agent 1 weather API was bypassed.")
        with wcol2:
            st.markdown("#### Agent 1A: yr.no (Met.no) Weather Data")
            # Debug: Show raw API response for troubleshooting
            st.code(str(results["yr_weather"]))
            wind_dir = results['yr_weather'].get('wind_direction_deg', None)
            wind_spd = results['yr_weather'].get('wind_speed_ms', None)
            st.metric("🧭 Wind Direction", f"{wind_dir if wind_dir is not None else 'N/A'}°")
            st.metric("💨 Wind Speed", f"{wind_spd if wind_spd is not None else 'N/A'} m/s")
            st.caption(f"Data source: {results['yr_weather']['data_source']}")
    
    # Arrow from Agent 1 to Agent 2
    st.markdown('''
    <div style="text-align: center; font-size: 2.5rem; color: #1E88E5; margin: 15px 0;">
    ⬇️
    </div>
    <div style="text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 15px;">
    Wind Direction & Speed → Agent 2
    </div>
    ''', unsafe_allow_html=True)
    
    # Pause to simulate handoff between agents
    time.sleep(0.9)
    
    # =========================================================================
    # AGENT 2: Turbine Expert : to recommend yaw angle based on turbine manual.
    # =========================================================================
    st.markdown("### 📖 Agent 2: Turbine Experts for SINGLE TURBINE OPERATION ")    
    status_text.text("🔄 Agent 2: Starting turbine manual consultation...")
    time.sleep(0.5)
    with st.spinner("Consulting NREL 5MW turbine manual based on wind conditions..."):
        status_text.text("Agent 2: Analyzing wind conditions against turbine specifications...")
        progress_bar.progress(35)
        time.sleep(1.0)
        
        expert = get_expert_recommendation(
            weather["wind_speed_ms"], 
            weather["wind_direction_deg"]
        )
        results["expert"] = expert
        progress_bar.progress(45)
    
    # Display actual real-world recommendation
    st.markdown("#### 📋 Turbine Recommendation")
    st.markdown(expert['actual_recommendation'])
    
    # Show recommended values
    acol1, acol2, acol3 = st.columns(3)
    with acol1:
        st.metric("Recommended Yaw", f"{expert['actual_yaw']:.1f}°")
    with acol2:
        st.metric("Recommended Pitch", f"{expert['actual_pitch']:.0f}°")
    with acol3:
        st.metric("Operating Region", expert['operating_region'])
    
    with st.expander("View Expert Reasoning"):
        for reason in expert['reasoning']:
            st.write(f"• {reason}")
    
    # CHECKPOINT 1 REVIEW: After Agent 2 completes
    if reviewer_agent is not None:
        try:
            with st.spinner("🎓 Expert Reviewer validating Agent 2..."):
                review1 = asyncio.run(reviewer_agent.review_agent2(
                    weather_data=weather,
                    expert_analysis=expert,
                    agent2b_result=results.get('llm_expert'),
                    turbine_pairs=None
                ))
                results["reviews"]["checkpoint1_agent2"] = review1
                display_checkpoint_review_short(review1, "Agent 2")
                
                # Check if workflow should halt (blocking mode)
                if not review1.get("allow_continue", True):
                    st.error("⛔ **Workflow Halted**: Critical issues detected by Expert Reviewer.")
                    st.stop()
        except Exception as e:
            st.warning(f"⚠️ Checkpoint 1 review failed: {e}")
    
    # Arrow from Agent 2 to Agent 2B
    st.markdown('''
    <div style="text-align: center; font-size: 2.5rem; color: #1E88E5; margin: 15px 0;">
    ⬇️
    </div>
    <div style="text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 15px;">
    Wind Conditions → Agent 2B (LLM Expert)
    </div>
    ''', unsafe_allow_html=True)
    
    time.sleep(0.9)
    
    # =========================================================================
    # AGENT 2B: LLM-based Turbine Expert (OPTIONAL)
    # =========================================================================
    enable_agent_2b = st.session_state.get('enable_agent_2b', True)
    
    if enable_agent_2b:
        st.markdown("### 🤖 Agent 2B: LLM-based Turbine Expert")
        st.markdown('''
        <p style="font-size: 1.0rem; color: #666; margin-top: -10px; font-style: italic;">
        Uses local LLM to provide intelligent turbine control recommendations.
        </p>
        ''', unsafe_allow_html=True)
        
        # LLM Configuration - Use GUI selections from session state
        # Don't load from config.yaml - use the selections from the sidebar
        llm_config = None  # This will trigger get_llm_expert_recommendation to use session state
        
        # Show current LLM configuration being used
        current_provider = st.session_state.get('llm_provider', 'NTNU')
        current_model = st.session_state.get('selected_model', 'moonshotai/Kimi-K2.5')
        st.info(f"🤖 Using LLM: **{current_provider}** - **{current_model}**")
        
        status_text.text("🔄 Agent 2B: Loading LLM configuration and querying for expert recommendations...")
        time.sleep(0.5)
        
        with st.spinner("Agent 2B: Consulting LLM for turbine control recommendations..."):
            try:
                llm_result = get_llm_expert_recommendation(
                    wind_speed=weather['wind_speed_ms'],
                    wind_direction=weather['wind_direction_deg'],
                    config=llm_config  # Pass None to use session state configuration
                )
                
                results['llm_expert'] = llm_result
                progress_bar.progress(0.4)
                time.sleep(0.3)
                
            except Exception as e:
                st.warning(f"⚠️ Agent 2B: Could not connect to LLM. Error: {e}")
                st.info("💡 Tip: Ensure your LLM server is running and accessible at the configured API base URL. "
                       "Set the LLM_API_KEY and LLM_API_BASE environment variables or update them in the code.")
                llm_result = None
        
        # Display LLM recommendations
        if llm_result and "llm_response" in llm_result:
            st.markdown("#### 🤖 LLM Expert Recommendation")
            st.markdown(llm_result['llm_response'])
            
            with st.expander("ℹ️ About Agent 2B"):
                config_info = llm_result.get('config', {})
                st.markdown(f"""
                **Agent 2B Configuration (from GUI selections):**
                - **Provider:** {config_info.get('provider', 'N/A')}
                - **Model:** {config_info.get('model', 'N/A')}
                - **API Base:** {config_info.get('api_base', 'N/A')}
                - **Temperature:** {config_info.get('temperature', 0.1)}
                - **Max Tokens:** {config_info.get('max_tokens', 200000):,}
                - **Timeout:** {config_info.get('timeout', 1000.0)}s
                - **Input:** Wind Speed ({llm_result['wind_speed']:.2f} m/s), Wind Direction ({llm_result['wind_direction']:.1f}°)
                
                **How it works:**
                1. Agent 2B uses the LLM provider and model selected in the GUI sidebar
                2. Queries the LLM server using OpenAI-compatible API
                3. The LLM is prompted with wind conditions and turbine specifications
                4. The LLM generates expert recommendations based on its training
                5. This provides an AI-powered second opinion alongside the rule-based Agent 2
                
                **Note:** The LLM's recommendations may differ from the rule-based expert (Agent 2). 
                Both perspectives can be valuable for decision-making.
                
                **Configuration:** Change the provider and model in the sidebar to use different AI services.
                """)
    else:
        st.info("⏭️ Agent 2B (LLM Expert) is disabled. Enable it in Analysis Settings to run.")
        results['llm_expert'] = None
    
    # Update checkpoint 1 review with Agent 2B if available
    if reviewer_agent is not None and results.get('llm_expert'):
        try:
            with st.spinner("🎓 Expert Reviewer re-validating with Agent 2B..."):
                review1_updated = asyncio.run(reviewer_agent.review_agent2(
                    weather_data=weather,
                    expert_analysis=expert,
                    agent2b_result=results.get('llm_expert'),
                    turbine_pairs=results.get('turbine_pairs')
                ))
                results["reviews"]["checkpoint1_agent2"] = review1_updated
                display_checkpoint_review_short(review1_updated, "Agent 2B")
        except Exception as e:
            st.warning(f"⚠️ Checkpoint 1B review failed: {e}")
    
    # =========================================================================
    # AGENT 2C: Turbine Pair Selection for Multi-Turbine Wake Optimization
    # =========================================================================
    # Add Agent 2C analysis if turbine locations are available
    if selected_farm in st.session_state.turbine_locations:
        search_results = st.session_state.turbine_locations[selected_farm]
        if search_results and search_results['turbine_locations']:
            # Get selected agent from user choice
            selected_agent = st.session_state.get('selected_agent', '2C')
            
            # Arrow from Agent 2B to Agent 2C/2D
            st.markdown(f'''
            <div style="text-align: center; font-size: 2.5rem; color: #1E88E5; margin: 15px 0;">
            ⬇️
            </div>
            <div style="text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 15px;">
            Wind & Turbine Data → Agent {selected_agent} (Turbine Pair Selector)
            </div>
            ''', unsafe_allow_html=True)
            
            if selected_agent == "2C":
                st.markdown("### 🎯 Agent 2C: LLM-Based Turbine Pair Selector")
                st.markdown('''
                <p style="font-size: 1.0rem; color: #666; margin-top: -10px; font-style: italic;">
                Analyzes turbine locations and wind conditions using LLM to identify critical turbine pairs 
                most affected by wake effects. These pairs will be optimized by the downstream agents.
                </p>
                ''', unsafe_allow_html=True)
            else:
                st.markdown("### 🎯 Agent 2D: Physical Wake Model Turbine Pair Selector")  
                st.markdown('''
                <p style="font-size: 1.0rem; color: #666; margin-top: -10px; font-style: italic;">
                Analyzes turbine locations and wind conditions using physical wake models to identify critical 
                turbine pairs most affected by wake deficits. Backup agent when LLM services are unavailable.
                </p>
                ''', unsafe_allow_html=True)
            
            progress_bar.progress(35)
            status_text.text(f"🔄 Agent {selected_agent}: Analyzing turbine pairs for wake optimization...")
            
            spinner_text = f"Agent {selected_agent}: {'Using LLM' if selected_agent == '2C' else 'Using physical wake models'} to identify critical turbine pairs..."
            with st.spinner(spinner_text):
                try:
                    if selected_agent == "2C":
                        # Add delay to avoid rate limiting from previous Agent 2B call
                        time.sleep(2)  # 2 second delay to prevent hitting rate limits
                        
                        # Get current LLM configuration from session state
                        current_provider = st.session_state.get('llm_provider', 'NTNU')
                        current_model = st.session_state.get('selected_model', 'moonshotai/Kimi-K2.5')
                        
                        try:
                            turbine_pair_analysis = get_turbine_pair_recommendations(
                                search_results['turbine_locations'],
                                weather['wind_speed_ms'],
                                weather['wind_direction_deg'],
                                current_provider,
                                current_model
                            )
                        except Exception as llm_error:
                            # Detect error type and provide specific guidance
                            error_msg = str(llm_error)
                            
                            # Show detailed error with suggestions
                            if '401' in error_msg:
                                st.error("❌ Agent 2C Failed: **Authentication Error (401)**")
                                st.warning("""**Suggestions:**
- ✏️ **Check API Key**: Verify your API key is correct in the configuration
- 🔄 **Try Different LLM**: Switch to another provider in the sidebar (OpenAI, Anthropic, Ollama)
- 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' below as backup
""")
                            elif '403' in error_msg:
                                st.error("❌ Agent 2C Failed: **Permission Denied (403)**")
                                st.warning("""**Suggestions:**
- 🔑 **API Access**: Your API key may not have access to this model
- 🔄 **Try Different Model**: Choose a different model in the sidebar
- 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' below as backup
""")
                            elif '429' in error_msg:
                                st.error("❌ Agent 2C Failed: **Rate Limit Exceeded (429)**")
                                st.warning("""**Suggestions:**
- ⏰ **Wait**: Try again in a few moments
- 🔄 **Try Different Provider**: Switch to another LLM provider in the sidebar
- 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' below as backup
""")
                            elif '500' in error_msg or '502' in error_msg or '503' in error_msg:
                                st.error("❌ Agent 2C Failed: **Server Error (5xx)**")
                                st.warning("""**Suggestions:**
- 🔄 **Try Again**: The LLM service may be temporarily down
- 🔄 **Try Different Provider**: Switch to another LLM provider in the sidebar
- 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' below as backup
""")
                            else:
                                st.error(f"❌ Agent 2C Failed: {error_msg[:150]}")
                                st.warning("""**Suggestions:**
- 🔄 **Try Different LLM**: Switch to another provider/model in the sidebar
- 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' below as backup
- 📝 **Check Configuration**: Verify your API settings
""")
                            
                            # Auto-fallback to Agent 2D
                            st.info("🔄 **Automatically falling back to Agent 2D (Physical Wake Model)**...")
                            
                            turbine_pair_analysis = get_wake_influenced_turbine_pairs(
                                search_results['turbine_locations'],
                                weather['wind_speed_ms'],
                                weather['wind_direction_deg'],
                                rotor_diameter=126.0,
                                hub_height=90.0,
                                wake_expansion_factor=st.session_state.get('wake_expansion_factor', 0.1),
                                min_influence_threshold=st.session_state.get('min_influence_threshold', 0.05)
                            )
                            
                            # Convert Agent 2D output to match Agent 2C format for compatibility
                            if turbine_pair_analysis and 'turbine_pairs' in turbine_pair_analysis:
                                turbine_pair_analysis.update({
                                    'status': 'success',
                                    'agent': f'Agent 2D (Fallback from 2C) - {turbine_pair_analysis.get("method", "Physical Wake Model")}',
                                    'provider': 'Physical Model (Fallback)',
                                    'model': 'Jensen Wake Model',
                                    'total_turbines': len(search_results['turbine_locations']),
                                    'wind_conditions': f"{weather['wind_speed_ms']:.1f} m/s from {weather['wind_direction_deg']:.0f}°",
                                    'analysis_summary': f"Auto-fallback to Agent 2D: {turbine_pair_analysis.get('wake_analysis', 'Wake analysis completed')}",
                                    'optimization_strategy': (
                                        f"Optimize {len(turbine_pair_analysis['turbine_pairs'])} wake-influenced pairs. "
                                        f"Max wake deficit: {turbine_pair_analysis.get('max_wake_deficit', 0):.1%}"
                                    )
                                })
                                
                                # Convert raw wake pairs to format compatible with downstream processing
                                if 'raw_wake_pairs' in turbine_pair_analysis:
                                    formatted_pairs = []
                                    for pair in turbine_pair_analysis['raw_wake_pairs'][:10]:  # Limit to top 10
                                        formatted_pairs.append({
                                            'upstream_turbine': pair['upstream_turbine'],
                                            'downstream_turbine': pair['downstream_turbine'],
                                            'distance_km': pair['distance_m'] / 1000.0,
                                            'distance_rotor_diameters': pair['distance_D'],
                                            'downstream_distance_D': pair['downstream_distance_m'] / 126.0,
                                            'lateral_offset_D': pair['lateral_distance_m'] / 126.0,
                                            'wake_strength': (
                                                'high' if pair['wake_deficit'] > 0.15 else
                                                'medium' if pair['wake_deficit'] > 0.08 else 'low'
                                            ),
                                            'wake_deficit': pair['wake_deficit'],
                                            'priority': len(formatted_pairs) + 1
                                        })
                                    turbine_pair_analysis['turbine_pairs'] = formatted_pairs
                    else:
                        # Use Agent 2D (Physical Wake Model)
                        turbine_pair_analysis = get_wake_influenced_turbine_pairs(
                            search_results['turbine_locations'],
                            weather['wind_speed_ms'],
                            weather['wind_direction_deg'],
                            rotor_diameter=126.0,
                            hub_height=90.0,
                            wake_expansion_factor=st.session_state.get('wake_expansion_factor', 0.1),
                            min_influence_threshold=st.session_state.get('min_influence_threshold', 0.05)
                        )
                        
                        # Convert Agent 2D output to match Agent 2C format for compatibility
                        if turbine_pair_analysis and 'turbine_pairs' in turbine_pair_analysis:
                            turbine_pair_analysis.update({
                                'status': 'success',
                                'agent': f'Agent 2D ({turbine_pair_analysis.get("method", "Physical Wake Model")})',
                                'provider': 'Physical Model',
                                'model': 'Jensen Wake Model',
                                'total_turbines': len(search_results['turbine_locations']),
                                'wind_conditions': f"{weather['wind_speed_ms']:.1f} m/s from {weather['wind_direction_deg']:.0f}°",
                                'analysis_summary': turbine_pair_analysis.get('wake_analysis', 'Wake analysis completed'),
                                'optimization_strategy': (
                                    f"Optimize {len(turbine_pair_analysis['turbine_pairs'])} wake-influenced pairs. "
                                    f"Max wake deficit: {turbine_pair_analysis.get('max_wake_deficit', 0):.1%}"
                                )
                            })
                            
                            # Convert raw wake pairs to format compatible with downstream processing
                            if 'raw_wake_pairs' in turbine_pair_analysis:
                                formatted_pairs = []
                                for pair in turbine_pair_analysis['raw_wake_pairs'][:10]:  # Limit to top 10
                                    formatted_pairs.append({
                                        'upstream_turbine': pair['upstream_turbine'],
                                        'downstream_turbine': pair['downstream_turbine'],
                                        'distance_km': pair['distance_m'] / 1000.0,
                                        'distance_rotor_diameters': pair['distance_D'],
                                        'downstream_distance_D': pair['downstream_distance_m'] / 126.0,
                                        'lateral_offset_D': pair['lateral_distance_m'] / 126.0,
                                        'wake_strength': (
                                            'high' if pair['wake_deficit'] > 0.15 else
                                            'medium' if pair['wake_deficit'] > 0.08 else 'low'
                                        ),
                                        'wake_deficit': pair['wake_deficit'],
                                        'priority': len(formatted_pairs) + 1
                                    })
                                turbine_pair_analysis['turbine_pairs'] = formatted_pairs
                        else:
                            # Fallback if no pairs found
                            turbine_pair_analysis = {
                                'status': 'no_pairs',
                                'agent': 'Agent 2D (Physical Wake Model)',
                                'provider': 'Physical Model',
                                'model': 'Jensen Wake Model',
                                'total_turbines': len(search_results['turbine_locations']),
                                'turbine_pairs': [],
                                'wind_conditions': f"{weather['wind_speed_ms']:.1f} m/s from {weather['wind_direction_deg']:.0f}°",
                                'analysis_summary': 'No significant wake interactions found at current wind conditions',
                                'optimization_strategy': 'No optimization needed - turbines operate independently'
                            }
                    
                    results["turbine_pairs"] = turbine_pair_analysis
                    progress_bar.progress(40)
                    
                    if turbine_pair_analysis['status'] == 'success':
                        st.success(f"✅ {turbine_pair_analysis['agent']} analysis complete using {turbine_pair_analysis['provider']} - {turbine_pair_analysis['model']}")
                        
                        # Display analysis results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Turbines", turbine_pair_analysis['total_turbines'])
                        with col2:
                            st.metric("Critical Pairs", len(turbine_pair_analysis['turbine_pairs']))
                        with col3:
                            st.metric("Wind Conditions", turbine_pair_analysis['wind_conditions'])
                        
                        # Show analysis summary
                        if turbine_pair_analysis.get('analysis_summary'):
                            st.markdown(f"**Analysis:** {turbine_pair_analysis['analysis_summary']}")
                        
                        if turbine_pair_analysis.get('optimization_strategy'):
                            st.markdown(f"**Strategy:** {turbine_pair_analysis['optimization_strategy']}")
                        
                        # Display critical turbine pairs
                        if turbine_pair_analysis['turbine_pairs']:
                            st.markdown("**Critical Turbine Pairs for Optimization:**")
                            
                            pair_cols = st.columns(min(3, len(turbine_pair_analysis['turbine_pairs'])))
                            for i, pair in enumerate(turbine_pair_analysis['turbine_pairs']):
                                col_idx = i % 3
                                with pair_cols[col_idx]:
                                    upstream = pair.get('upstream_turbine', 'N/A')
                                    downstream = pair.get('downstream_turbine', 'N/A')
                                    distance = pair.get('distance_km', 'N/A')
                                    strength = pair.get('wake_strength', 'unknown')
                                    priority = pair.get('priority', i+1)
                                    
                                    if strength == 'high':
                                        emoji = "🔴"
                                        color = "#ff4444"
                                    elif strength == 'medium':
                                        emoji = "🟡"
                                        color = "#ffaa00"
                                    else:
                                        emoji = "🟢"
                                        color = "#44aa44"
                                    
                                    distance_text = f"{distance:.2f}km" if isinstance(distance, (int, float)) else "N/A"
                                    
                                    st.markdown(f"""
                                    <div style="border: 1px solid {color}; border-radius: 8px; padding: 10px; margin: 5px 0;">
                                    <div style="font-weight: bold; color: {color};">
                                    {emoji} Priority {priority}
                                    </div>
                                    <div style="font-size: 1.1em; margin: 5px 0;">
                                    T{upstream} → T{downstream}
                                    </div>
                                    <div style="font-size: 0.9em; color: #666;">
                                    Distance: {distance_text}<br>
                                    Wake: {strength}
                                    </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No critical turbine pairs identified for current wind conditions")
                        
                        # Show rejected pairs if any (Agent 2C validation)
                        if selected_agent == "2C" and turbine_pair_analysis.get('rejected_pairs'):
                            with st.expander(f"⚠️ Rejected Pairs ({len(turbine_pair_analysis['rejected_pairs'])} pairs not aligned with wind)"):
                                st.markdown("""**These pairs were identified by the LLM but rejected by physical validation:**
                                
The pairs below were suggested by the AI model but did not pass wind direction validation. 
They are not aligned with the current wind direction (±45° tolerance).""")
                                
                                for rejected in turbine_pair_analysis['rejected_pairs']:
                                    upstream = rejected.get('upstream_turbine', 'N/A')
                                    downstream = rejected.get('downstream_turbine', 'N/A')
                                    reason = rejected.get('reason', 'Unknown')
                                    bearing = rejected.get('bearing_deg', 'N/A')
                                    
                                    st.markdown(f"""
                                    <div style="border: 1px solid #ff6b6b; border-radius: 8px; padding: 10px; margin: 5px 0; background-color: #ffe0e0;">
                                    <div style="font-weight: bold; color: #c92a2a;">
                                    ❌ T{upstream} → T{downstream}
                                    </div>
                                    <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                                    <b>Rejection Reason:</b> {reason}<br>
                                    <b>Bearing:</b> {bearing}° (Wind: {turbine_pair_analysis['wind_conditions'].split('from')[1].split('°')[0].strip()}°)
                                    </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.info(f"""**Validation Info:** 
✅ Accepted: {turbine_pair_analysis.get('validated_pairs_count', 0)} pairs
❌ Rejected: {len(turbine_pair_analysis['rejected_pairs'])} pairs
📊 Total LLM suggestions: {turbine_pair_analysis.get('llm_identified_pairs', 0)} pairs
                                
Physical validation ensures all pairs are aligned with wind direction from Agent 1.""")
                        
                        with st.expander("ℹ️ About Agent 2C"):
                            st.markdown(f"""
                            **Agent 2C Configuration (from GUI selections):**
                            - **LLM Provider:** {turbine_pair_analysis['provider']}
                            - **LLM Model:** {turbine_pair_analysis['model']}
                            - **Total Turbines Analyzed:** {turbine_pair_analysis['total_turbines']}
                            - **Wind Conditions:** {turbine_pair_analysis['wind_conditions']}
                            
                            **How it works:**
                            1. Agent 2C receives turbine locations (latitude/longitude) from the location search
                            2. **NEW:** Converts lat/lon to Cartesian coordinates (x, y in meters) for easier LLM analysis
                            3. Sends turbine positions in meters (e.g., T1 at origin, T2 at x=350m, y=120m)
                            4. LLM uses the provider and model selected in GUI sidebar (same as Agent 2B)
                            5. LLM analyzes wind direction, turbine spacing, and geometry to identify wake interactions
                            6. **Physical validation:** Each LLM-suggested pair is validated against actual wind direction
                            7. Only pairs aligned with wind (±45° tolerance) are passed to Agent 3
                            
                            **Wind Direction Validation:**
                            - Calculates actual bearing between turbine pairs
                            - Compares bearing with wind direction from Agent 1
                            - Rejects pairs where "downstream" turbine is not actually downwind
                            - Ensures upstream→downstream relationship is physically correct
                            
                            **Purpose:** This agent bridges single-turbine analysis (Agent 2B) with 
                            multi-turbine farm optimization by intelligently selecting which turbine 
                            pairs will benefit most from wake steering control.
                            
                            **Configuration:** Change the provider and model in the sidebar to use different AI services.
                            
                            **Performance Optimizations (Latest):**
                            - **Cartesian coordinates**: Lat/lon converted to x,y meters for clearer LLM input
                            - 2-second delay after Agent 2B to prevent rate limiting
                            - **4000 tokens** output limit for complete JSON responses
                            - **60-second timeout** for complex computational analysis
                            - **Clear prompt format**: Positions in meters (e.g., "T1(x=0m, y=0m)")
                            - **Temperature 0.1** for deterministic JSON output
                            - Triple-fallback strategy: (1) JSON with system msg, (2) JSON without, (3) Plain text
                            - Comprehensive debug logging (check terminal/console)
                            """)
                    
                    elif turbine_pair_analysis['status'] == 'partial':
                        st.warning(f"⚠️ {turbine_pair_analysis['agent']} analysis partially complete")
                        st.markdown(f"**Analysis:** {turbine_pair_analysis.get('analysis_summary', 'Partial results available')}")
                        
                        # Show debug information for partial results
                        with st.expander("🐛 Debug Information (Click to expand)"):
                            st.markdown(f"""
                            **LLM Configuration:**
                            - Provider: {turbine_pair_analysis.get('provider', 'N/A')}
                            - Model: {turbine_pair_analysis.get('model', 'N/A')}
                            
                            **Issue:**
                            {turbine_pair_analysis.get('parse_error', 'LLM response could not be parsed as JSON')}
                            
                            **Raw LLM Response:**
                            """)
                            st.code(turbine_pair_analysis.get('raw_response', 'No response captured')[:1000], language="text")
                            
                            if len(turbine_pair_analysis.get('raw_response', '')) > 1000:
                                st.caption("(Response truncated to 1000 characters)")
                        
                        st.info("""**💡 Solutions:**
1. 🔄 **Try Different Model**: The current model may not support JSON format - try a different model in the sidebar
2. 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' above for reliable physics-based analysis
3. 📝 **Check LLM Settings**: Some models require specific formatting - verify your model selection
""")
                        
                        results["turbine_pairs"] = turbine_pair_analysis
                    
                    else:
                        # Handle error status with specific guidance
                        error_msg = turbine_pair_analysis.get('message', 'Unknown error')
                        error_code = turbine_pair_analysis.get('error_code')
                        
                        st.error(f"❌ {turbine_pair_analysis['agent']} analysis failed")
                        st.code(error_msg, language="text")
                        
                        # Show debug information for errors that might be LLM-related
                        if turbine_pair_analysis.get('raw_response') and selected_agent == "2C":
                            with st.expander("🐛 Debug Information (Click to expand)"):
                                st.markdown(f"""
                                **LLM Configuration:**
                                - Provider: {turbine_pair_analysis.get('provider', 'N/A')}
                                - Model: {turbine_pair_analysis.get('model', 'N/A')}
                                - Total Turbines Sent: {turbine_pair_analysis.get('total_turbines', 'N/A')}
                                
                                **Recovery Attempts:**
                                {turbine_pair_analysis.get('recovery_attempts', 'N/A')}
                                
                                **Prompt Information:**
                                - Prompt Length: {turbine_pair_analysis.get('prompt_length', 'N/A')} characters
                                - Estimated Tokens: ~{turbine_pair_analysis.get('prompt_length', 0)//4}
                                
                                **Prompt Preview (first 500 chars):**
                                """)
                                st.code(turbine_pair_analysis.get('prompt_preview', 'No prompt captured')[:500], language="text")
                                
                                st.markdown("**Raw LLM Response:**")
                                raw_resp = turbine_pair_analysis.get('raw_response', 'No response captured')
                                if raw_resp.startswith('ERROR_'):
                                    st.code(raw_resp, language="text")
                                else:
                                    st.code(raw_resp[:1000], language="text")
                                    if len(raw_resp) > 1000:
                                        st.caption("(Response truncated to 1000 characters)")
                                
                                st.markdown("""
                                **Troubleshooting Empty Responses:**
                                - Model may have insufficient output tokens configured
                                - Prompt may be too complex or too long
                                - Model may not support JSON format well
                                - Check terminal/console for detailed debug logs
                                """)
                        
                        # Provide helpful suggestions based on error
                        if error_code == '401':
                            st.warning("""**💡 Solutions:**
1. ✏️ **Check API Key**: Verify your API credentials in the configuration
2. 🔄 **Switch LLM**: Try a different provider/model in the sidebar (Settings → LLM Configuration)
3. 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' above instead
""")
                        elif error_code == '403':
                            st.warning("""**💡 Solutions:**
1. 🔑 **Model Access**: Your API key may not have access to `{model}`
2. 🔄 **Try Different Model**: Select another model in the sidebar
3. 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' above instead
""".format(model=turbine_pair_analysis.get('model', 'this model')))
                        elif error_code == '429':
                            st.warning("""**💡 Solutions for Rate Limiting:**
1. ⏰ **Wait & Retry**: Rate limit exceeded - the system has automatic delays now
2. 🔄 **Try Again**: Click 'Run Analysis' again - Agent 2C now includes 2s delay after Agent 2B
3. 🔄 **Switch Provider**: Use a different LLM provider in the sidebar
4. 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' above instead

**Note:** Agent 2C now includes automatic rate-limit protection:
- 2-second delay after Agent 2B to prevent conflicts
- Exponential backoff (5s, 10s, 20s) on rate limit errors
- Extended 45-second timeout for complex queries
""")
                        elif error_code == '5xx':
                            st.warning("""**💡 Solutions:**
1. 🔄 **Try Again**: Server error - the service may be temporarily down
2. 🔄 **Different Provider**: Switch to another LLM provider in the sidebar
3. 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' above instead
""")
                        else:
                            # Check if it's an empty response error
                            if 'EMPTY_RESPONSE' in error_msg:
                                st.warning("""**💡 Solutions for Empty Response Error:**

**This model is not responding to Agent 2C requests despite 3 recovery attempts.**

Agent 2C tried:
1. ✅ Standard JSON request with system message
2. ✅ JSON request WITHOUT system message (some models don't handle it)
3. ✅ Simple plain text request

All attempts returned empty/null responses. **This suggests model incompatibility.**

**Recommended Actions:**
1. 🔧 **Use Agent 2D**: Click 'Agent 2D (Physical Wake Model)' above - **This always works!**
2. 🔄 **Try Different Model**: Switch to a model known to support structured outputs:
   - GPT-4 or GPT-3.5-turbo (OpenAI)
   - Claude 3 (Anthropic)
   - Llama models via Ollama locally
3. 📝 **Check Model**: Current model may not support:
   - JSON output format
   - Complex reasoning tasks
   - Long prompts with multiple turbines

**Note:** Agent 2D uses physics-based wake models and is 100% reliable.
""")
                        
                        results["turbine_pairs"] = None
                        
                except Exception as e:
                    st.error(f"❌ Agent {selected_agent}: Critical error in turbine pair analysis")
                    st.code(str(e), language="text")
                    st.warning("""**💡 Solutions:**
1. 🔄 **Try Different LLM**: Switch provider/model in the sidebar if using Agent 2C
2. 🔧 **Use Agent 2D**: Select 'Agent 2D (Physical Wake Model)' above for reliable physics-based analysis
3. 📝 **Check Configuration**: Verify all settings are correct
""")
                    results["turbine_pairs"] = None
        else:
            st.info("🎯 Agent 2C: Turbine pair analysis requires turbine location data. Agent 2C will be skipped.")
            results["turbine_pairs"] = None
    else:
        st.info("🎯 Agent 2C: No turbine location data available. Use 'Search Turbine Locations' in the sidebar to enable multi-turbine analysis.")
        results["turbine_pairs"] = None

    # Arrow from Agent 2B/2C to Agent 3
    st.markdown('''
    <div style="text-align: center; font-size: 2.5rem; color: #1E88E5; margin: 15px 0;">
    ⬇️
    </div>
    <div style="text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 15px;">
    Wind Conditions & Turbine Pairs → Agent 3 (Two-Turbine Optimizer)
    </div>
    ''', unsafe_allow_html=True)
    
    # Pause to simulate agents loading ML models
    time.sleep(1.0)
    
    # =========================================================================
    # AGENT 3: Two-Turbine Wake Steering Optimizer
    # =========================================================================
    st.markdown("### 🎯 Agent 3: Two-Turbine Wake Steering Optimizer")
    st.markdown('''
    <p style="font-size: 1.0rem; color: #666; margin-top: -10px; font-style: italic;">
    Optimizes yaw misalignment for turbine pairs identified by Agent 2C/2D to maximize total power output.
    The optimizer adjusts yaw angles to steer wakes laterally, reducing downstream wake impact.
    </p>
    ''', unsafe_allow_html=True)
    
    # Get optimization method from session state (set in sidebar)
    opt_method = st.session_state.get('opt_method', 'analytical_physics')
    max_pairs = st.session_state.get('max_pairs_to_optimize', 4)
    
    # Display selected method
    st.info(f"**Selected Method:** {opt_method.replace('_', ' ').title()} | **Pairs to Optimize:** {max_pairs}")
    
    # Show explanation of selected method
    if opt_method == 'ml_surrogate':
        with st.expander("ℹ️ About ML Surrogate AD Method"):
            st.markdown("""
            **True Automatic Differentiation through ML Models**
            
            This method creates **differentiable polynomial surrogates** fitted to the actual ML model 
            predictions (GP Power model and TT-OpInf Wake ROM), enabling genuine gradient-based 
            optimization through the ML models.
        
        **How it works:**
        
        1. **Surrogate Construction**: 
           - Sample the GP power model at 16 yaw angles → fit cubic polynomial $P(\\gamma)$
           - Sample the TT-OpInf wake ROM at 8 yaw angles → fit quadratic deficit $\\delta(\\gamma)$
           - These polynomials are PyTorch `nn.Module` objects with differentiable coefficients
        
        2. **Differentiable Farm Power**: The surrogates enable true AD:
           ```
           upstream_power = power_surrogate(γ₁)           # Differentiable!
           wake_deficit = wake_surrogate(γ₁)              # Differentiable!
           downstream_power = power_surrogate(γ₂) × (1-δ)³
           ```
        
        3. **Gradient Computation**: PyTorch computes exact gradients:
           $$\\nabla P = \\left[\\frac{\\partial P_{total}}{\\partial \\gamma_1}, \\frac{\\partial P_{total}}{\\partial \\gamma_2}\\right]$$
           through backpropagation from loss to parameters through the surrogate models.
        
        **Advantages:**
        - ✅ Gradients reflect actual ML model behavior (not just analytical approximations)
        - ✅ Captures non-linear relationships learned by ML models from CFD data
        - ✅ More accurate optima when ML models capture physics better than analytical models
        
        **Trade-offs:**
        - ⏱️ Requires initial surrogate fitting (~2-5 seconds)
        - 📊 Polynomial approximation introduces small fitting error
        """)
    
    with st.expander("📐 Method 2: AD through Analytical Physics Model"):
        st.markdown("""
        **Automatic Differentiation through Analytical Wake Steering Physics**
        
        This method uses **analytical physics equations** that are directly differentiable, 
        while ML models provide baseline power and wake deficit values (as constants).
        
        **How it works:**
        
        1. **ML Models as Constants**: 
           - GP model provides base power $P_{base}$ (evaluated once at 270°)
           - TT-OpInf ROM provides base wake deficit $\\delta_{base}$ (evaluated once at 270°)
        
        2. **Differentiable Physics Layer**: Analytical equations are differentiated:
           - *Yaw power loss*: $P = P_{base} \\cdot \\cos^3(\\gamma)$
           - *Wake steering* (Bastankhah & Porté-Agel 2016): 
             $$\\delta_{eff} = \\delta_{base} \\cdot (1 - k \\cdot \\sin(\\gamma) \\cdot \\cos^2(\\gamma))$$
           - *Downstream power*: $P_2 = P_{base} \\cdot (1-\\delta_{eff})^3 \\cdot \\cos^3(\\gamma_2)$
        
        3. **Gradient Computation**: PyTorch differentiates through cos, sin operations:
           $$\\frac{\\partial P}{\\partial \\gamma_1} = P_{base} \\cdot \\left(-3\\cos^2(\\gamma_1)\\sin(\\gamma_1) + \\text{wake steering terms}\\right)$$
        
        **Advantages:**
        - ✅ Fast (no surrogate fitting needed)
        - ✅ Physics-interpretable gradients
        - ✅ Well-validated analytical models
        
        **Trade-offs:**
        - ⚠️ ML models are treated as constants (not differentiated through)
        - ⚠️ May miss complex relationships captured by ML models
        """)
    
    with st.expander("🔍 Method 3: Grid Search (Reference)"):
        st.markdown("""
        **Brute-Force Grid Search**
        
        Evaluates all combinations on a discrete grid. No gradients used.
        
        - **Grid**: Upstream [0°, 2°, 4°, ..., 12°] × Downstream [0°, 2°, 4°]
        - **Evaluations**: 7 × 3 = 21 combinations
        - **Advantage**: Guaranteed to find global optimum on grid
        - **Disadvantage**: Limited to grid resolution, doesn't find exact optimum
        """)
    
    # Explain the conversion
    with st.expander("ℹ️ Understanding Yaw Misalignment ↔ Nacelle Direction"):
        st.markdown("""
        **Conversion between Yaw Misalignment and Nacelle Direction:**
        
        The ML models (Wake Flow ROM and Power Predictor) use **nacelle direction** as input, 
        while the optimizer works with **yaw misalignment** angles.
        
        | Yaw Misalignment | Nacelle Direction | Description |
        |-----------------|-------------------|-------------|
        | 0° | 270° | Aligned with wind (no steering) |
        | 5° | 275° | Mild wake steering |
        | 10° | 280° | Moderate wake steering |
        | 12° | 282° | Aggressive wake steering |
        | 15° | 285° | Maximum misalignment (training limit) |
        
        **Formula:** `nacelle_direction = 270° + yaw_misalignment`
        
        The optimizer searches for optimal misalignment in the range **0° - 12°** 
        (nacelle direction 270° - 282°) to stay within well-validated ROM range.
        """)
    
    status_text.text("🔄 Agent 3: Running wake steering optimization for turbine pairs...")
    time.sleep(0.5)
    
    # Check if we have turbine pairs from Agent 2C/2D
    turbine_pairs_data = results.get('turbine_pairs', None)
    
    # Validate that Agent 2C/2D produced valid output before proceeding
    if turbine_pairs_data is None:
        st.error("⛔ **Agent 3 Cannot Run: No valid input from Agent 2C/2D**")
        st.warning("""**Agent 3 requires turbine pair data from Agent 2C or 2D to proceed.**

**Please ensure:**
- ✅ Agent 2C (LLM) or Agent 2D (Physical Model) completed successfully
- ✅ At least one turbine pair was identified for optimization

**If Agent 2C failed:**
- Try selecting a different LLM provider/model in the sidebar
- Switch to 'Agent 2D (Physical Wake Model)' in the dropdown above
- Ensure your API credentials are configured correctly

**Skipping Agent 3 optimization...**
""")
        results["optimizer"] = None
    elif not turbine_pairs_data.get('turbine_pairs'):
        st.warning("⚠️ **Agent 3 Cannot Run: No turbine pairs found**")
        st.info("""Agent 2C/2D completed but found no significant wake interactions for the current wind conditions.

**This can happen when:**
- Turbines are too far apart for wake effects
- Wind direction doesn't align turbines in wake patterns  
- Wake influence is below the detection threshold

**No optimization needed - turbines operate independently.**

Skipping Agent 3 optimization...
""")
        results["optimizer"] = None
    elif turbine_pairs_data.get('status') != 'success':
        st.error("⛔ **Agent 3 Cannot Run: Agent 2C/2D status is not successful**")
        st.warning(f"""Agent 2C/2D status: **{turbine_pairs_data.get('status', 'unknown')}**

**Agent 3 requires successful turbine pair analysis to proceed.**

Please resolve the Agent 2C/2D issues above before running Agent 3.

Skipping Agent 3 optimization...
""")
        results["optimizer"] = None
    elif turbine_pairs_data and turbine_pairs_data.get('turbine_pairs'):
        # Multi-pair optimization
        num_pairs = len(turbine_pairs_data['turbine_pairs'])
        st.info(f"📋 Found **{num_pairs} turbine pairs** from Agent 2C/2D. Optimizing top **{max_pairs}** pairs...")
        
        with st.spinner(f"Optimizing {min(max_pairs, num_pairs)} turbine pairs using {opt_method.replace('_', ' ').title()} method..."):
            progress_bar.progress(48)
            
            try:
                power_agent, wake_agent = load_agents()
                
                # Run multi-pair optimizer
                opt_results = optimize_multiple_turbine_pairs(
                    turbine_pairs_data=turbine_pairs_data,
                    power_agent=power_agent,
                    wake_agent=wake_agent,
                    optimization_method=opt_method,
                    n_timesteps=min(n_timesteps, 30),
                    max_pairs=max_pairs,
                    verbose=True  # Enable debug output
                )
                
                results["optimizer"] = opt_results
                progress_bar.progress(52)
                
                if opt_results['status'] == 'success':
                    st.success(f"✅ Optimized **{opt_results['num_pairs_optimized']}** turbine pairs! Total power gain: **{opt_results['total_power_gain']:.3f} MW**")
                else:
                    st.warning(f"⚠️ Optimization completed with issues: {opt_results.get('message', 'Unknown')}")
                    
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                results["optimizer"] = None
                import traceback
                st.code(traceback.format_exc())
    else:
        # Fallback: single two-turbine optimization (original behavior)
        st.warning("⚠️ No turbine pairs from Agent 2C/2D. Running default two-turbine optimization...")
        
        with st.spinner(f"Optimizing yaw misalignment using {opt_method.replace('_', ' ').title()} method..."):
            progress_bar.progress(48)
            
            try:
                power_agent, wake_agent = load_agents()
                
                # Run single pair optimizer (original method)
                opt_results = optimize_two_turbine_farm(
                    power_agent=power_agent,
                    wake_agent=wake_agent,
                    turbine_spacing_D=7.0,
                    n_timesteps=min(n_timesteps, 30),
                    verbose=False,
                    optimization_method=opt_method
                )
                
                results["optimizer"] = opt_results
                progress_bar.progress(52)
                
                st.success(f"✅ Wake steering optimization complete! Method: **{opt_results['optimization_method']}**")
                
                # Convert to multi-pair format for unified display
                upstream_yaw = opt_results['optimal_upstream_misalignment']
                opt_results = {
                    'status': 'success',
                    'optimization_method': opt_results['optimization_method'],
                    'num_pairs_optimized': 1,
                    'total_power_gain': opt_results['power_gain_MW'],
                    'optimization_results': [{
                        'pair_index': 1,
                        'turbine_ids': [1, 2],
                        'upstream_id': 1,
                        'downstream_id': 2,
                        'upstream_yaw': upstream_yaw,
                        'downstream_yaw': 0.0,  # Wake steering: downstream stays at 0°
                        'optimal_yaw_angles': {
                            1: upstream_yaw,
                            2: 0.0  # Downstream at 0° for wake steering
                        },
                        'power_gain_MW': opt_results['power_gain_MW'],
                        'power_gain_percent': opt_results['power_gain_percent'],
                        'baseline_power': opt_results['baseline_total_power'],
                        'optimized_power': opt_results['optimal_total_power'],
                        'upstream_power_baseline': opt_results.get('baseline_upstream_power', 0.0),
                        'upstream_power_optimized': opt_results.get('optimal_upstream_power', 0.0),
                        'downstream_power_baseline': opt_results.get('baseline_downstream_power', 0.0),
                        'downstream_power_optimized': opt_results.get('optimal_downstream_power', 0.0)
                    }]
                }
                results["optimizer"] = opt_results
            
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                results["optimizer"] = None
                import traceback
                st.code(traceback.format_exc())
    
    # =========================================================================
    # Display Optimization Results (Unified for both multi-pair and single-pair)
    # =========================================================================
    if "optimizer" in results and results["optimizer"] is not None:
        opt_results = results["optimizer"]
        
        if opt_results.get('status') == 'success' and 'optimization_results' in opt_results:
            # Display optimization results
            st.markdown("#### 📊 Optimization Results")
            st.markdown(f"**Method:** {opt_results['optimization_method']}")
            
            optimization_data = opt_results['optimization_results']
            
            # Get wind direction from weather agent (Agent 1)
            wind_dir = weather.get('wind_direction_deg', 270)
            
            # Create results table with upstream/downstream columns
            st.markdown("**💡 Wake Steering Optimization Results:**")
            st.info("""**Wake Steering Principle:** Only the upstream turbine is misaligned to steer the wake away. 
The downstream turbine stays aligned at 0° to capture maximum power outside the wake.""")
            
            # Build table data
            table_data = []
            for result in optimization_data:
                upstream_id = result.get('upstream_id')
                downstream_id = result.get('downstream_id')
                upstream_yaw = result.get('upstream_yaw', result['optimal_yaw_angles'].get(upstream_id, 0.0))
                downstream_yaw = result.get('downstream_yaw', result['optimal_yaw_angles'].get(downstream_id, 0.0))
                spacing_D = result.get('turbine_spacing_D', 7.0)  # Euclidean distance for display
                ds_dist_D = result.get('downstream_distance_D', spacing_D)  # Streamwise distance
                lat_off_D = result.get('lateral_offset_D', 0.0)  # Cross-wind offset
                overlap = result.get('overlap_class', 'full')  # Wake overlap classification

                row = {
                    'Pair #': result['pair_index'],
                    'Upstream Turbine': f"T{upstream_id}",
                    'Streamwise (D)': f"{ds_dist_D:.2f}",
                    'Lateral (D)': f"{lat_off_D:.2f}",
                    'Wake Overlap': overlap.capitalize(),
                    'Upstream Yaw (°)': f"{upstream_yaw:.1f}",
                    'Downstream Turbine': f"T{downstream_id}",
                    'Downstream Yaw (°)': f"{downstream_yaw:.1f}",
                    'Power Gain (MW)': f"{result['power_gain_MW']:+.4f}",
                    'Gain %': f"{result['power_gain_percent']:+.2f}%"
                }
                
                table_data.append(row)
            
            # Create DataFrame and display
            df_results = pd.DataFrame(table_data)
            
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Pair #': st.column_config.NumberColumn('Pair #', help='Priority rank from Agent 2C/2D'),
                    'Upstream Turbine': st.column_config.TextColumn('Upstream Turbine', help='Upwind turbine (applies wake steering)'),
                    'Streamwise (D)': st.column_config.TextColumn('Streamwise (D)', help='Wind-aligned distance in rotor diameters (D=126m)'),
                    'Lateral (D)': st.column_config.TextColumn('Lateral (D)', help='Cross-wind lateral offset in rotor diameters'),
                    'Wake Overlap': st.column_config.TextColumn('Wake Overlap', help='Full=centerline, Partial=edge of wake, None=outside wake'),
                    'Upstream Yaw (°)': st.column_config.TextColumn('Upstream Yaw (°)', help='Yaw misalignment for wake steering'),
                    'Downstream Turbine': st.column_config.TextColumn('Downstream Turbine', help='Downwind turbine (in wake region)'),
                    'Downstream Yaw (°)': st.column_config.TextColumn('Downstream Yaw (°)', help='Physics-based yaw to align with deflected wake'),
                    'Power Gain (MW)': st.column_config.TextColumn('Power Gain (MW)', help='Net power improvement for this pair'),
                    'Gain %': st.column_config.TextColumn('Gain %', help='Percentage power improvement')
                }
            )
            
            # Explanation of wake steering
            with st.expander("ℹ️ Understanding Wake Steering Results"):
                st.markdown("""
                **Wake Steering Concept:**
                
                Wake steering is a control strategy where the **upstream turbine is intentionally misaligned** 
                with the wind to steer its wake away from downstream turbines.
                
                **Physics-Based Approach:**
                - **Upstream Turbine Yaw**: Optimized to find best wake deflection angle
                - **Downstream Turbine Yaw**: Calculated using Bastankhah & Porté-Agel (2016) wake deflection model
                  - When upstream yaws, wake deflects laterally
                  - Downstream turbine aligns with deflected wake centerline for maximum power capture
                  - This is NOT an optimization variable, but a physics-based calculation
                
                **Power Trade-off:**
                - **Upstream Turbine**: Loses power due to yaw misalignment (cos³ loss)
                - **Downstream Turbine**: Gains power by operating in deflected wake with optimal alignment
                - **Net Result**: Total farm power increases (positive gain)
                
                **How to Read the Table:**
                1. **Pair #**: Priority ranking from Agent 2C/2D
                2. **Upstream Turbine**: The turbine ID that will be yawed (from Agent 2C/2D)
                3. **Upstream Yaw**: Optimized yaw misalignment angle (in degrees)
                4. **Downstream Turbine**: The turbine ID responding to wake (from Agent 2C/2D)
                5. **Downstream Yaw**: Physics-calculated alignment (typically small, 0-5°)
                6. **Power Gain**: Net power improvement for this turbine pair
                
                **Important Notes:**
                - Turbine IDs are taken directly from Agent 2C/2D recommendations
                - **Each pair uses its actual physical spacing** (not hardcoded 7D)
                - Different spacings lead to different optimal yaw angles
                - Each pair is optimized individually (not simultaneously)
                - Downstream yaw is computed from wake deflection physics (not optimized independently)
                - Positive power gain confirms wake steering is beneficial
                - Negative gains indicate wake steering is not effective for that pair
                """)
            
            # Summary metrics
            st.markdown("#### ⚡ Overall Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Pairs Optimized", opt_results['num_pairs_optimized'])
            with summary_col2:
                st.metric("Total Power Gain", f"{opt_results['total_power_gain']:.4f} MW")
            with summary_col3:
                avg_gain = opt_results['total_power_gain'] / opt_results['num_pairs_optimized'] if opt_results['num_pairs_optimized'] > 0 else 0
                st.metric("Avg Gain per Pair", f"{avg_gain:.4f} MW")
            
            # Detailed view in expander
            with st.expander("📋 Detailed Power Analysis for Each Pair"):
                for result in optimization_data:
                    upstream_id = result.get('upstream_id')
                    downstream_id = result.get('downstream_id')
                    upstream_yaw = result.get('upstream_yaw', result['optimal_yaw_angles'].get(upstream_id, 0.0))
                    spacing_D = result.get('turbine_spacing_D', 7.0)
                    spacing_m = spacing_D * 126.0  # Convert to meters (NREL 5MW rotor diameter)
                    
                    st.markdown(f"**Pair {result['pair_index']}:** T{upstream_id} (Upstream) → T{downstream_id} (Downstream) | **Spacing: {spacing_D:.2f}D ({spacing_m:.0f}m)**")
                    
                    detail_cols = st.columns(4)
                    
                    # Upstream turbine
                    with detail_cols[0]:
                        actual_yaw_upstream = wind_dir + upstream_yaw
                        st.metric(
                            f"T{upstream_id} (Upstream)",
                            f"{upstream_yaw:.1f}° misalign",
                            delta=f"Actual: {actual_yaw_upstream:.1f}°",
                            help="Wake steering yaw angle"
                        )
                        if 'upstream_power_baseline' in result and 'upstream_power_optimized' in result:
                            up_diff = result['upstream_power_optimized'] - result['upstream_power_baseline']
                            st.caption(f"Power: {result['upstream_power_baseline']:.3f} → {result['upstream_power_optimized']:.3f} MW ({up_diff:+.3f})")
                    
                    # Downstream turbine
                    with detail_cols[1]:
                        st.metric(
                            f"T{downstream_id} (Downstream)",
                            "0° aligned",
                            delta=f"Actual: {wind_dir:.1f}°",
                            help="Stays aligned with wind"
                        )
                        if 'downstream_power_baseline' in result and 'downstream_power_optimized' in result:
                            down_diff = result['downstream_power_optimized'] - result['downstream_power_baseline']
                            st.caption(f"Power: {result['downstream_power_baseline']:.3f} → {result['downstream_power_optimized']:.3f} MW ({down_diff:+.3f})")
                    
                    # Net gain
                    with detail_cols[2]:
                        st.metric(
                            "Net Pair Gain",
                            f"{result['power_gain_MW']:.4f} MW",
                            delta=f"{result['power_gain_percent']:+.2f}%",
                            help="Total power gain for this pair"
                        )
                    
                    # Explanation
                    with detail_cols[3]:
                        if result['power_gain_MW'] > 0:
                            st.success("✅ Net Positive")
                            st.caption("Downstream gain > Upstream loss")
                        elif result['power_gain_MW'] < 0:
                            st.error("❌ Net Negative")
                            st.caption("Upstream loss > Downstream gain")
                        else:
                            st.info("➖ No Change")
                    
                    if 'baseline_power' in result and 'optimized_power' in result:
                        st.caption(f"**Total Pair Power:** {result['baseline_power']:.4f} MW → {result['optimized_power']:.4f} MW")
                    
                    st.markdown("---")
            
            # Display ML Wake Extraction Sensitivity Figures (if available)
            if opt_results['optimization_method'] == 'ML Wake Extraction (Expert Approach)':
                st.markdown("#### 📈 Yaw Sensitivity Analysis (ML Wake Extraction)")
                st.info("These figures show how lateral wake migration and power vary with upstream yaw angle for each optimized turbine pair.")
                
                for result in optimization_data:
                    # Check if this pair has sensitivity figure
                    if 'sensitivity_figure' in result and result['sensitivity_figure'] is not None:
                        upstream_id = result.get('upstream_id')
                        downstream_id = result.get('downstream_id')
                        spacing_D = result.get('turbine_spacing_D', 7.0)
                        
                        st.markdown(f"**Pair {result['pair_index']}: T{upstream_id} → T{downstream_id} (Spacing: {spacing_D:.2f}D)**")
                        st.pyplot(result['sensitivity_figure'])
                        plt.close(result['sensitivity_figure'])
                        st.markdown("---")
            
            # Store optimal nacelle directions for Agent 4 (wake simulation)
            # Use first pair for downstream agents
            if optimization_data:
                first_pair = optimization_data[0]
                turbine_ids = first_pair['turbine_ids']
                optimal_yaws = first_pair['optimal_yaw_angles']
                
                # Store upstream (first) and downstream (second) turbine info
                if len(turbine_ids) >= 1:
                    upstream_id = turbine_ids[0]
                    upstream_misalign = optimal_yaws.get(upstream_id, 0.0)
                    results["optimal_nacelle_upstream"] = yaw_misalignment_to_nacelle_direction(upstream_misalign)
                    results["actual_yaw_upstream"] = wind_dir + upstream_misalign
                    results["optimal_upstream_misalignment"] = upstream_misalign
                
                if len(turbine_ids) >= 2:
                    downstream_id = turbine_ids[1]
                    downstream_misalign = optimal_yaws.get(downstream_id, 0.0)
                    results["optimal_nacelle_downstream"] = yaw_misalignment_to_nacelle_direction(downstream_misalign)
                    results["actual_yaw_downstream"] = wind_dir + downstream_misalign
                    results["optimal_downstream_misalignment"] = downstream_misalign
        
        else:
            st.warning("⚠️ Optimization completed but no results available.")
    
    # =========================================================================
    # Analysis Summary Report (after optimizer, before demonstrators)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📊 Analysis Summary Report")
    st.markdown("*Summary of wind farm analysis based on optimizer results (Agents 1-3)*")
    
    display_summary_report(results)
    
    # =========================================================================
    # Chatbot Interface for Analysis Questions
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🤖 Ask Questions About the Analysis")
    st.markdown("*Use the chatbot to ask questions about the optimization results and recommendations*")
    
    display_analysis_chatbot(results)
    
    # =========================================================================
    # Optional Demonstrations (Agent 4 & Agent 5)
    # =========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;'>
        <h2 style='color: white; text-align: center; margin-bottom: 1rem;'>🎯 Optional Demonstrators</h2>
        <p style='color: white; text-align: center; font-size: 1.1rem;'>
            The following agents demonstrate wake flow simulation and power prediction<br>
            using the <b>recommended yaw angles</b> from the optimizer above.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Store indicator that we're entering demonstrator mode
    if "optimizer" in results and results["optimizer"] is not None:
        opt_results = results["optimizer"]
        if opt_results.get('status') == 'success' and 'optimization_results' in opt_results:
            optimization_data = opt_results['optimization_results']
            first_pair = optimization_data[0]
            turbine_ids = first_pair['turbine_ids']
            optimal_yaws = first_pair['optimal_yaw_angles']
            st.info(f"ℹ️ Using Pair 1 (Turbines {' → '.join([f'T{tid}' for tid in turbine_ids])}) for demonstrations with recommended yaw misalignment")
    
    # Arrow from Agent 3 to Agent 4
    st.markdown('''
    <div style="text-align: center; font-size: 2.5rem; color: #1E88E5; margin: 15px 0;">
    ⬇️
    </div>
    <div style="text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 15px;">
    Optimal Nacelle Directions → Agent 4 (Wake Flow Simulation)
    </div>
    ''', unsafe_allow_html=True)
    
    time.sleep(0.8)
    
    # =========================================================================
    # AGENT 4: Wind Turbine Wake Flow at Recommended Yaw Angle: A Reduced Order Model (OPTIONAL)
    # =========================================================================
    enable_agent_4 = st.session_state.get('enable_agent_4', False)
    
    if enable_agent_4:
        st.markdown("### 🌊 Agent 4: Wind Turbine Wake Flow: A Reduced Order Model")
        st.markdown('''
        <p style="font-size: 1.0rem; color: #666; margin-top: -10px; font-style: italic;">
        <b>Parametric Reduced Order Model</b> based on hi-fidelity CFD data.
        based on <b>Tensor Decomposition + Radial Basis Function + Operator Inference</b>.
        Contact: mandar.tabib@sintef.no for details.
    </p>
        ''', unsafe_allow_html=True)
        
        # Use the optimal nacelle direction from the optimizer (Agent 3)
        # Note: This is the ML model input (270°-285°), which is a proxy for yaw misalignment (0°-15°)
        optimal_yaw_ml = results.get("optimal_nacelle_upstream", expert['suggested_yaw'])
        actual_yaw_upstream = results.get("actual_yaw_upstream", weather.get('wind_direction_deg', 270))
        optimal_misalign_upstream = results.get("optimal_upstream_misalignment", 0)
        
        st.markdown(f"""**Running wake simulation:**
    - ML Model Input (Nacelle Dir): **{optimal_yaw_ml:.1f}°** (proxy for {optimal_misalign_upstream:.0f}° misalignment)
    - Actual Turbine Yaw: **{actual_yaw_upstream:.0f}°** (= {weather.get('wind_direction_deg', 270):.0f}° wind dir + {optimal_misalign_upstream:.0f}° misalign)
    """)
        
        status_text.text("🔄 Agent 4: Initializing TD-RBF-OpInf model...")
        time.sleep(0.5)
        with st.spinner("Running ROM: A TD-RBF-OpInf wake flow simulator..."):
            status_text.text("Agent 4: Computing velocity field using TD-RBF-OpInf...")
            progress_bar.progress(60)
            
            try:
                power_agent, wake_agent = load_agents()
                
                predictions, vtk_dir = wake_agent.predict(
                    yaw_angle=optimal_yaw_ml,
                    n_timesteps=n_timesteps,
                    export_vtk=export_vtk,
                    export_every=5,
                    verbose=False
                )
                
                velocity_mag = np.linalg.norm(predictions, axis=2)
                
                results["wake"] = {
                    "predictions": predictions,
                    "velocity_mag": velocity_mag,
                    "vtk_dir": vtk_dir,
                    "shape": predictions.shape
                }
                
                progress_bar.progress(65)
                
                st.success(f"✅ Wake simulation complete!")
                
                wcol1, wcol2 = st.columns(2)
                with wcol1:
                    st.metric("Spatial Points", f"{predictions.shape[1]:,}")
                with wcol2:
                    st.metric("Timesteps", predictions.shape[0])
                
                if vtk_dir:
                    st.info(f"📁 VTK files saved to: `{vtk_dir}`")
                
                # Generate wake flow animation
                progress_bar.progress(70)
                st.markdown("#### 🎬 Wake Flow Animation")
                
                with st.spinner("Generating wake flow animation..."):
                    status_text.text("Creating ParaView-style visualization...")
                    
                    try:
                        from wake_animation import create_wake_contour_animation
                        
                        anim_path = os.path.join(SCRIPT_DIR, "wake_animation_optimal.gif")
                        grid_path = str(SCRIPT_DIR / "data" / "Grid_data.vtk")
                        
                        create_wake_contour_animation(
                            predictions=predictions,
                            grid_path=grid_path,
                            output_path=anim_path,
                            fps=8,
                            frame_skip=max(1, predictions.shape[0] // 20),
                            cmap="RdYlBu_r",
                            verbose=False,
                            yaw_misalignment=optimal_misalign_upstream
                        )
                       
                        
                        st.image(anim_path, caption=f"Wake Flow (ML input: {optimal_yaw_ml:.0f}°, Actual Yaw: {actual_yaw_upstream:.0f}°)")
                        
                        with open(anim_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Animation",
                                data=f,
                                file_name=f"wake_animation_misalign_{optimal_misalign_upstream:.0f}deg.gif",
                                mime="image/gif",
                                key="download_wake_optimal"
                            )
                            
                    except Exception as anim_e:
                        st.warning(f"Could not create animation: {anim_e}")
                    
            except Exception as e:
                st.error(f"Wake simulation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                results["wake"] = None
    else:
        st.info("⏭️ Agent 4 (Wake Flow ROM) is disabled. Enable it in Analysis Settings to run.")
        results["wake"] = None
    
    # Arrow from Agent 4 to Agent 5
    st.markdown('''
    <div style="text-align: center; font-size: 2.5rem; color: #1E88E5; margin: 15px 0;">
    ⬇️
    </div>
    <div style="text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 15px;">
    Yaw Angle → Agent 5 (Power Prediction)
    </div>
    ''', unsafe_allow_html=True)
    
    time.sleep(0.9)
    
    # =========================================================================
    # AGENT 5: Power Predictor (OPTIONAL)
    # =========================================================================
    enable_agent_5 = st.session_state.get('enable_agent_5', False)
    
    if enable_agent_5:
        st.markdown("### ⚡ Agent 5: Power Predictor")
        st.markdown('''
        <p style="font-size: 1.0rem; color: #666; margin-top: -10px; font-style: italic;">
        Gaussian Process Regression trained on high-fidelity CFD simulation data.
        </p>
        ''', unsafe_allow_html=True)
        
        # Use the optimal nacelle direction from the optimizer (Agent 3)
        # Note: This is the ML model input (270°-285°), which is a proxy for yaw misalignment (0°-15°)
        optimal_yaw_power_ml = results.get("optimal_nacelle_upstream", expert['suggested_yaw'])
        
        st.markdown(f"""**Running power prediction:**
    - ML Model Input (Nacelle Dir): **{optimal_yaw_power_ml:.1f}°** (proxy for {optimal_misalign_upstream:.0f}° misalignment)
    - Actual Turbine Yaw: **{actual_yaw_upstream:.0f}°** (= {weather.get('wind_direction_deg', 270):.0f}° wind dir + {optimal_misalign_upstream:.0f}° misalign)
    """)
        
        status_text.text("🔄 Agent 5: Loading SINTEF ML models...")
        time.sleep(0.5)
        with st.spinner("Running Gaussian Process power prediction..."):
            status_text.text("Agent 5: Computing power output with uncertainty quantification...")
            progress_bar.progress(80)
            
            try:
                power_agent, wake_agent = load_agents()
                
                power_results = power_agent.predict(
                    yaw_angle=optimal_yaw_power_ml,
                    n_time_points=n_timesteps,
                    return_samples=True,
                    n_samples=50
                )
                
                # Use time range 0.1 to 0.9 for all computations
                t = power_results['normalized_time']
                valid_idx = (t >= 0.1) & (t <= 0.9)
                
                # Calculate statistics from power_mean_MW vector (analytical GP mean)
                mean_power = np.mean(power_results['power_mean_MW'][valid_idx])
                min_power = np.min(power_results['power_mean_MW'][valid_idx])
                max_power = np.max(power_results['power_mean_MW'][valid_idx])
                power_variation = max_power - min_power
                uncertainty = np.mean(power_results['power_std_MW'][valid_idx])
                
                results["power"] = {
                    "mean_MW": mean_power,
                    "min_MW": min_power,
                    "max_MW": max_power,
                    "variation_MW": power_variation,
                    "uncertainty_MW": uncertainty,
                    "time_series": power_results
                }
                
                progress_bar.progress(85)
                
                st.success(f"✅ Power prediction complete!")
                
                # Display power metrics from analytical GP mean (t=0.1-0.9)
                st.markdown("**📈 Power Statistics from Analytical GP Mean (t=0.1-0.9):**")
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    st.metric("Time-Averaged", f"{mean_power:.3f} MW", help="np.mean(power_mean_MW)")
                with pcol2:
                    st.metric("Min Power", f"{min_power:.3f} MW", help="np.min(power_mean_MW)")
                with pcol3:
                    st.metric("Max Power", f"{max_power:.3f} MW", help="np.max(power_mean_MW)")
                with pcol4:
                    st.metric("Variation (ΔP)", f"{power_variation:.3f} MW", help="Max - Min")
                
                # Power plot (t=0.1-0.9)
                fig, ax = plt.subplots(figsize=(10, 4))
                t_valid = t[valid_idx]
                
                # Plot posterior samples (light blue)
                if 'samples' in power_results and power_results['samples'] is not None:
                    samples = power_results['samples']
                    samples_valid = samples[:, valid_idx]
                    n_plot_samples = min(20, samples.shape[0])
                    for i in range(n_plot_samples):
                        ax.plot(t_valid, samples_valid[i], 'steelblue', alpha=0.12, linewidth=0.7,
                               label='GP Posterior Samples' if i == 0 else '')
                
                # Plot GP mean (power_mean_MW) - dark blue line
                ax.plot(t_valid, power_results['power_mean_MW'][valid_idx], 'darkblue', linewidth=2.5, 
                       label='GP Mean Prediction (power_mean_MW)', zorder=5)
                
                # Plot 95% CI
                ax.fill_between(t_valid, power_results['power_lower_95_MW'][valid_idx], 
                              power_results['power_upper_95_MW'][valid_idx],
                              alpha=0.15, color='gray', label='95% CI', zorder=3)
                
                # Plot time-averaged value of GP mean
                ax.axhline(y=mean_power, color='red', linestyle='--', linewidth=2, alpha=0.8,
                          label=f'Time-Average of GP Mean: {mean_power:.3f} MW', zorder=4)
                
                ax.set_xlabel('Normalized Time')
                ax.set_ylabel('Power (MW)')
                ax.set_title(f'GP Power Prediction (t=0.1-0.9): Misalign={optimal_misalign_upstream:.0f}°, Yaw={actual_yaw_upstream:.0f}° | ΔP={power_variation:.3f} MW')
                ax.legend(loc='best', fontsize=8, ncol=2)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown(f"""
                **📊 Plot Legend:**
                | Line Type | Description | Computation |
                |-----------|-------------|-------------|
                | **Light blue lines** | GP posterior samples | Random draws from GP distribution (samples array) |
                | **Dark blue line** | GP mean prediction | `power_mean_MW` - GP's expected value at each time point |
                | **Red dashed line** | Time-average of dark blue line | `np.mean(power_mean_MW)` = {mean_power:.3f} MW |
                
                **Statistics Computed from Analytical GP Mean (t=0.1-0.9):**
                - **Time-Averaged** = `np.mean(power_mean_MW)` = {mean_power:.3f} MW
                - **Min Power** = `np.min(power_mean_MW)` = {min_power:.3f} MW  
                - **Max Power** = `np.max(power_mean_MW)` = {max_power:.3f} MW
                - **Variation** = Max - Min = {power_variation:.3f} MW
                """)
                
            except Exception as e:
                st.error(f"Power prediction failed: {e}")
                results["power"] = None
    else:
        st.info("⏭️ Agent 5 (Power Predictor) is disabled. Enable it in Analysis Settings to run.")
        results["power"] = None
    
    # CHECKPOINT 3 REVIEW: After Agent 4/5 (Power/Wake) completes
    if reviewer_agent is not None and results.get("power"):
        try:
            with st.spinner("🎓 Expert Reviewer validating power predictions..."):
                review3 = asyncio.run(reviewer_agent.review_agent4(
                    weather_data=weather,
                    expert_analysis=expert,
                    wake_prediction=results.get("wake", {})
                ))
                results["reviews"]["checkpoint3_agent4"] = review3
                display_checkpoint_review_short(review3, "Agent 4/5")
        except Exception as e:
            st.warning(f"⚠️ Checkpoint 3 review failed: {e}")
    
    progress_bar.progress(95)
    
    # Generate final expert review
    if reviewer_agent is not None:
        try:
            with st.spinner("🎓 Expert Reviewer generating final assessment..."):
                final_review = asyncio.run(reviewer_agent.generate_final_review(results))
                results["reviews"]["final_review"] = final_review
                
                # Display final review status
                status = final_review.get("overall_status", "UNKNOWN")
                if status == "FAILED":
                    st.error(f"🔴 **Expert Review**: {final_review.get('status_message', 'Analysis contains critical issues')}")
                elif status == "WARNING":
                    st.warning(f"⚠️ **Expert Review**: {final_review.get('status_message', 'Analysis completed with warnings')}")
                else:
                    st.success(f"✅ **Expert Review**: {final_review.get('status_message', 'All validation checks passed')}")
        except Exception as e:
            st.warning(f"⚠️ Final expert review failed: {e}")
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    # Store results
    st.session_state.results = results
    st.session_state.analysis_complete = True


def display_analysis_chatbot(results):
    """Display an interactive chatbot for asking questions about the analysis."""
    
    # Initialize chat history in session state
    if 'analysis_chat_history' not in st.session_state:
        st.session_state.analysis_chat_history = []
    
    # Get analysis context
    weather = results.get("weather", {})
    expert = results.get("expert", {})
    optimizer = results.get("optimizer", {})
    turbine_pairs = results.get("turbine_pairs", {})
    
    # Build context for LLM
    wind_speed = weather.get('wind_speed_ms', 0)
    wind_dir = weather.get('wind_direction_deg', 0)
    operating_region = expert.get('operating_region', 'N/A')
    
    # Enhanced system message for wind turbine operations expert
    system_message = """You are an expert in wind turbine operations and wind farm optimization with deep knowledge of:
- Wake steering and yaw control strategies for wind farms
- Aerodynamic interactions and wake effects between turbines
- Wind turbine power curves and operating regions
- Wind farm layout optimization and control strategies
- Trade-offs between individual turbine power loss and total farm power gain
- NREL 5MW Reference Wind Turbine specifications and performance

Provide clear, actionable answers with technical depth. Be specific about:
- Physical principles behind wake steering effects
- Quantitative analysis of power gains and losses
- Practical implementation considerations
- Operating conditions and their impact on optimization

Use technical terminology appropriately but explain complex concepts clearly."""

    context = f"""
WEATHER CONDITIONS:
- Wind Speed: {wind_speed:.1f} m/s
- Wind Direction: {wind_dir:.0f}°
- Temperature: {weather.get('temperature_c', 0):.1f}°C

TURBINE:
- Type: NREL 5MW Reference Wind Turbine
- Operating Region: {operating_region}

"""
    
    if optimizer:
        opt_method = optimizer.get('optimization_method', 'N/A')
        num_pairs = optimizer.get('num_pairs_optimized', 0)
        total_gain = optimizer.get('total_power_gain', 0)
        
        context += f"""OPTIMIZATION RESULTS:
- Method: {opt_method}
- Number of Pairs Optimized: {num_pairs}
- Total Power Gain: {total_gain:.4f} MW

"""
        
        if 'optimization_results' in optimizer:
            context += "PAIR-BY-PAIR RESULTS:\n"
            for result in optimizer['optimization_results']:
                upstream_id = result.get('upstream_id', result['turbine_ids'][0])
                downstream_id = result.get('downstream_id', result['turbine_ids'][1])
                upstream_yaw = result['optimal_yaw_angles'].get(upstream_id, 0.0)
                power_gain = result['power_gain_MW']
                gain_pct = result['power_gain_percent']
                
                context += f"""  Pair {result['pair_index']}: T{upstream_id} → T{downstream_id}
    - Upstream Yaw Misalignment: {upstream_yaw:.1f}°
    - Power Gain: {power_gain:.4f} MW ({gain_pct:+.2f}%)
"""
    
    # Suggested questions
    st.markdown("**💡 Suggested Questions:**")
    suggested_questions = [
        "What is the recommended wake steering strategy?",
        "Why does wake steering improve total farm power?",
        "How do the wind conditions affect the optimization?",
        "What are the trade-offs for each turbine pair?",
        "How much power gain can we expect?",
        "What happens if we don't apply wake steering?"
    ]
    
    # Display suggested questions as clickable buttons
    cols = st.columns(3)
    for i, question in enumerate(suggested_questions):
        with cols[i % 3]:
            if st.button(question, key=f"suggest_{i}", use_container_width=True):
                # Directly add the question and trigger processing
                st.session_state.analysis_selected_question = question
                st.session_state.analysis_trigger_ask = True
    
    st.markdown("---")
    
    # Determine if we should process a question
    process_analysis_question = False
    user_question = ""
    
    # Check if triggered by suggested question
    if st.session_state.get('analysis_trigger_ask', False):
        user_question = st.session_state.get('analysis_selected_question', '')
        st.session_state.analysis_trigger_ask = False
        if user_question:
            process_analysis_question = True
    
    # Chat input (shown when not actively processing)
    if not process_analysis_question:
        user_question = st.text_input(
            "Ask a question about the analysis:",
            key="user_analysis_question",
            placeholder="e.g., Why is the power gain positive?"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Ask", type="primary", key="ask_analysis_btn"):
                if user_question:
                    process_analysis_question = True
        with col2:
            if st.button("Clear Chat", key="clear_analysis_chat"):
                st.session_state.analysis_chat_history = []
                st.rerun()
    
    # Process question if triggered
    if process_analysis_question and user_question:
        # Add user question to history
        st.session_state.analysis_chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                # Construct detailed prompt with supplementary research references
                reports_context = load_research_reports_context()
                detailed_prompt = f"{context}\n{reports_context}\n\nUser Question: {user_question}\n\nProvide a detailed, technical answer based on the wind farm analysis context above:"
                
                response = query_unified_llm(
                    prompt=detailed_prompt,
                    system_message=system_message,
                    temperature=0.3,  # Lower temperature for consistent responses
                    max_tokens=1200,  # Reduced for faster responses
                    timeout=60.0  # Increased timeout
                )
                st.session_state.analysis_chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.session_state.analysis_selected_question = ''
            except Exception as e:
                error_msg = f"⚠️ Error querying LLM: {str(e)}"
                st.session_state.analysis_chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.session_state.analysis_selected_question = ''
        
        st.rerun()  # Rerun to display the new response
    
    # Always display chat history at the end
    if st.session_state.analysis_chat_history:
        st.markdown("**💬 Chat History:**")
        for message in st.session_state.analysis_chat_history:
            if message["role"] == "user":
                st.markdown(f"**👤 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {message['content']}")
            st.markdown("")


def display_summary_report(results):
    """Display a summary report of the analysis (Agents 1-3, excluding demonstrators)."""
    
    weather = results.get("weather", {})
    expert = results.get("expert", {})
    optimizer = results.get("optimizer", {})
    
    # Get wind direction for actual yaw calculation
    wind_dir = weather.get('wind_direction_deg', 270)
    
    # Basic info
    report_md = f"""
| Parameter | Value |
|-----------|-------|
| **Location** | {results.get('location', 'N/A')} |
| **Timestamp** | {results.get('timestamp', 'N/A')} |
| **Wind Speed** | {weather.get('wind_speed_ms', 0):.1f} m/s |
| **Wind Direction** | {weather.get('wind_direction_deg', 0):.0f}° |
| **Operating Region** | {expert.get('operating_region', 'N/A')} |
"""
    
    st.markdown(report_md)
    
    # Display optimizer results table if available
    if optimizer and optimizer.get('status') == 'success':
        if 'optimization_results' in optimizer:
            # Multi-pair format
            st.markdown("#### Wake Steering Optimization Results")
            st.markdown(f"**Method:** {optimizer.get('optimization_method', 'N/A')}")
            
            opt_results = optimizer['optimization_results']
            
            # Create summary table
            table_rows = []
            for result in opt_results:
                upstream_id = result.get('upstream_id', result['turbine_ids'][0] if result['turbine_ids'] else 'N/A')
                downstream_id = result.get('downstream_id', result['turbine_ids'][1] if len(result['turbine_ids']) > 1 else 'N/A')
                upstream_yaw = result.get('upstream_yaw', result['optimal_yaw_angles'].get(upstream_id, 0.0))
                downstream_yaw = result.get('downstream_yaw', result['optimal_yaw_angles'].get(downstream_id, 0.0))
                
                table_rows.append(f"| Pair {result['pair_index']} | T{upstream_id} | {upstream_yaw:.1f}° | T{downstream_id} | {downstream_yaw:.1f}° | {result['power_gain_MW']:+.4f} MW | {result['power_gain_percent']:+.2f}% |")
            
            table_md = """| Pair # | Upstream Turbine | Upstream Yaw | Downstream Turbine | Downstream Yaw | Power Gain | Gain % |
|--------|------------------|--------------|--------------------|--------------------|------------|--------|
"""
            table_md += "\n".join(table_rows)
            
            st.markdown(table_md)
            
            # Overall summary
            st.markdown("**Overall Summary:**")
            total_gain_mw = optimizer.get('total_power_gain', 0)
            num_pairs = optimizer.get('num_pairs_optimized', 0)
            avg_gain = total_gain_mw / num_pairs if num_pairs > 0 else 0
            
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Pairs Optimized", num_pairs)
            with summary_cols[1]:
                st.metric("Total Power Gain", f"{total_gain_mw:.4f} MW")
            with summary_cols[2]:
                st.metric("Avg Gain per Pair", f"{avg_gain:.4f} MW")
        else:
            # Legacy single-pair format
            st.markdown("#### Wake Steering Optimization Results")
            st.markdown(f"**Method:** {optimizer.get('optimization_method', 'N/A')}")
            
            total_gain_mw = optimizer.get('power_gain_MW', 0)
            gain_pct = optimizer.get('power_gain_percent', 0)
            
            # Create simple table
            opt_up_misalign = optimizer.get('optimal_upstream_misalignment', 0)
            opt_down_misalign = optimizer.get('optimal_downstream_misalignment', 0)
            actual_yaw_T1 = wind_dir + opt_up_misalign
            actual_yaw_T2 = wind_dir + opt_down_misalign
            
            table_md = f"""| Turbine | Role | Yaw Misalignment | Actual Yaw Angle | Power Output |
|---------|------------|------------------|------------------|---------------|
| T1 | Upstream | {opt_up_misalign:.1f}° | {actual_yaw_T1:.1f}° | {optimizer.get('optimal_upstream_power', 0):.3f} MW |
| T2 | Downstream | {opt_down_misalign:.1f}° | {actual_yaw_T2:.1f}° | {optimizer.get('optimal_downstream_power', 0):.3f} MW |
"""
            st.markdown(table_md)
            
            # Overall summary
            st.markdown("**Overall Summary:**")
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Baseline Power", f"{optimizer.get('baseline_total_power', 0):.3f} MW")
            with summary_cols[1]:
                st.metric("Optimized Power", f"{optimizer.get('optimal_total_power', 0):.3f} MW")
            with summary_cols[2]:
                st.metric("Net Power Gain", f"{total_gain_mw:.4f} MW", delta=f"{gain_pct:+.2f}%")
    
    st.markdown("*Note: Summary based on Agents 1-3 (Weather, Expert, Optimizer). Wake flow and power predictions are optional demonstrators below.*")
    
    # Export button
    
    # Export button
    if st.button("📄 Export Full Report"):
        # Build optimizer section for export
        if optimizer:
            opt_up_misalign = optimizer.get('optimal_upstream_misalignment', 0)
            opt_down_misalign = optimizer.get('optimal_downstream_misalignment', 0)
            # Calculate actual yaw = wind direction + misalignment
            actual_yaw_T1_export = wind_dir + opt_up_misalign
            actual_yaw_T2_export = wind_dir + opt_down_misalign
            
            # Handle both single-pair and multi-pair optimization results
            if 'optimization_results' in optimizer:
                # Multi-pair format - clean table summary
                opt_results = optimizer['optimization_results']
                total_gain_mw = optimizer.get('total_power_gain', 0)
                num_pairs = optimizer.get('num_pairs_optimized', 0)
                avg_gain = total_gain_mw / num_pairs if num_pairs > 0 else 0
                
                optimizer_section = f"""
WAKE STEERING OPTIMIZATION SUMMARY
----------------------------------
Optimization Method: {optimizer.get('optimization_method', 'N/A')}
Number of Pairs Optimized: {num_pairs}
Total Power Gain: {total_gain_mw:.4f} MW
Average Gain per Pair: {avg_gain:.4f} MW

Optimization Results Table:
------------------------------------------------------------------------------------------
| Pair # | Upstream    | Upstream Yaw | Downstream  | Downstream Yaw | Power Gain     |
|--------|-------------|--------------|-------------|----------------|----------------|
"""
                for result in opt_results:
                    upstream_id = result.get('upstream_id', result['turbine_ids'][0] if result['turbine_ids'] else 'N/A')
                    downstream_id = result.get('downstream_id', result['turbine_ids'][1] if len(result['turbine_ids']) > 1 else 'N/A')
                    upstream_yaw = result.get('upstream_yaw', result['optimal_yaw_angles'].get(upstream_id, 0.0))
                    downstream_yaw = result.get('downstream_yaw', result['optimal_yaw_angles'].get(downstream_id, 0.0))
                    
                    optimizer_section += f"| {result['pair_index']:6d} | T{str(upstream_id):<10} | {upstream_yaw:11.1f}° | T{str(downstream_id):<10} | {downstream_yaw:13.1f}° | {result['power_gain_MW']:+.4f} MW ({result['power_gain_percent']:+.2f}%) |\n"
                
                optimizer_section += f"""
------------------------------------------------------------------------------------------

Note: Upstream yaw values are optimized for wake steering.
      Downstream yaw values are physics-based wake alignment.
      Actual Yaw Angle = Wind Direction ({wind_dir:.0f}°) + Yaw Misalignment
"""
            else:
                # Legacy single-pair format - clean table summary
                total_gain_mw = optimizer.get('power_gain_MW', 0)
                gain_pct = optimizer.get('power_gain_percent', 0)
                
                optimizer_section = f"""
WAKE STEERING OPTIMIZATION SUMMARY
----------------------------------
Optimization Method: {optimizer.get('optimization_method', 'N/A')}
Number of Pairs Optimized: 1
Total Power Gain: {total_gain_mw:.4f} MW ({gain_pct:+.2f}%)

Optimization Results:
------------------------------------------------------------------------------------------
| Turbine | Role       | Yaw Misalignment | Actual Yaw Angle       | Power Output     |
|---------|------------|------------------|------------------------|------------------|
| T1      | Upstream   | {opt_up_misalign:15.1f}° | {actual_yaw_T1_export:21.1f}° | {optimizer.get('optimal_upstream_power', 0):15.3f} MW |
| T2      | Downstream | {opt_down_misalign:15.1f}° | {actual_yaw_T2_export:21.1f}° | {optimizer.get('optimal_downstream_power', 0):15.3f} MW |
------------------------------------------------------------------------------------------

Total Farm Power:
  - Baseline (no wake steering): {optimizer.get('baseline_total_power', 0):.3f} MW
  - Optimized (with wake steering): {optimizer.get('optimal_total_power', 0):.3f} MW
  - Net Power Gain: {total_gain_mw:.4f} MW ({gain_pct:+.2f}%)

Note: Actual Yaw Angle = Wind Direction ({wind_dir:.0f}°) + Yaw Misalignment
"""
        else:
            optimizer_section = ""
        
        # Generate turbine pair analysis section
        turbine_pairs = results.get('turbine_pairs', {})
        if turbine_pairs and turbine_pairs.get('status') == 'success':
            pairs_text = []
            for pair in turbine_pairs.get('turbine_pairs', []):
                upstream = pair.get('upstream_turbine', 'N/A')
                downstream = pair.get('downstream_turbine', 'N/A')
                strength = pair.get('wake_strength', 'unknown')
                distance = pair.get('distance_km', 'N/A')
                if isinstance(distance, (int, float)):
                    pairs_text.append(f"  T{upstream} → T{downstream}: {distance:.2f}km ({strength} wake)")
                else:
                    pairs_text.append(f"  T{upstream} → T{downstream}: {strength} wake")
            
            turbine_pair_section = f"""
Total Turbines: {turbine_pairs.get('total_turbines', 'N/A')}
Critical Pairs Identified: {len(turbine_pairs.get('turbine_pairs', []))}
Wind Conditions: {turbine_pairs.get('wind_conditions', 'N/A')}
Analysis: {turbine_pairs.get('analysis_summary', 'N/A')}
Optimization Strategy: {turbine_pairs.get('optimization_strategy', 'N/A')}

Critical Turbine Pairs:
{chr(10).join(pairs_text) if pairs_text else '  No critical pairs identified'}
"""
        else:
            turbine_pair_section = """
Turbine pair analysis not available or failed.
"""
        
        report_text = f"""
WIND TURBINE MULTI-AGENT ANALYSIS REPORT
========================================
Generated: {results.get('timestamp', 'N/A')}
Location: {results.get('location', 'N/A')}

WEATHER CONDITIONS (Agent 1)
----------------------------
Wind Speed: {weather.get('wind_speed_ms', 0):.1f} m/s
Wind Direction: {weather.get('wind_direction_deg', 0):.0f}°
Temperature: {weather.get('temperature_c', 0):.1f}°C
Data Source: {weather.get('data_source', 'N/A')}

EXPERT RECOMMENDATION (Agent 2A/2B)
------------------------------------
Turbine: NREL 5MW Reference Wind Turbine
Operating Region: {expert.get('operating_region', 'N/A')}
Real-World Yaw (aligned): {expert.get('actual_yaw', 0):.1f}°
Expected Efficiency: {expert.get('expected_efficiency', 0)*100:.1f}%
{optimizer_section}
TURBINE PAIR ANALYSIS (Agent 2C/2D)
------------------------------------
{turbine_pair_section}
========================================
Report generated by Wind Turbine Multi-Agent System
Developed at SINTEF Digital by mandar.tabib.

Note: This report includes results from Agents 1-3 (Weather, Expert, Optimizer).
      Wake flow (Agent 4) and power predictions (Agent 5) are optional demonstrators.
"""
        st.download_button(
            label="Download Report (TXT)",
            data=report_text,
            file_name="wind_turbine_report.txt",
            mime="text/plain"
        )


def display_checkpoint_review_short(review, checkpoint_name):
    """Display a short bullet-point checkpoint review immediately after agent completes."""
    if review is None or not review.get("findings"):
        return
    
    severity = review.get("severity", "info")
    findings = review.get("findings", [])
    
    # Count critical and warning findings
    critical_count = sum(1 for f in findings if f.get("type") == "critical")
    warning_count = sum(1 for f in findings if f.get("type") == "warning")
    
    # Display header with color code
    if severity == "critical":
        st.error(f"🔴 **Expert Review - {checkpoint_name}**: {critical_count} critical issue(s)")
    elif severity == "warning":
        st.warning(f"⚠️ **Expert Review - {checkpoint_name}**: {warning_count} warning(s)")
    else:
        st.success(f"✅ **Expert Review - {checkpoint_name}**: All checks passed")
    
    # Show top 3 findings as bullets
    display_findings = findings[:3]
    for finding in display_findings:
        finding_type = finding.get("type", "info")
        message = finding.get("message", "")[:100]  # Truncate to 100 chars
        
        if finding_type == "critical":
            st.markdown(f"- 🔴 {message}")
        elif finding_type == "warning":
            st.markdown(f"- ⚠️ {message}")
        else:
            st.markdown(f"- ✅ {message}")
    
    if len(findings) > 3:
        st.caption(f"... and {len(findings) - 3} more finding(s). See comprehensive review for details.")


def generate_review_export_report(reviews, results):
    """Generate a comprehensive review report for export."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = results.get("location", "Unknown")
    
    report = f"""╔═══════════════════════════════════════════════════════════════════════════════╗
║                    EXPERT REVIEW COMPREHENSIVE REPORT                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Generated: {timestamp:<64} ║
║  Location:  {location:<64} ║
╚═══════════════════════════════════════════════════════════════════════════════╝

"""
    
    # Final Review Summary
    final_review = reviews.get("final_review", {})
    if final_review:
        status = final_review.get("overall_status", "UNKNOWN")
        status_message = final_review.get("status_message", "")[:76]
        
        report += f"""┌───────────────────────────────────────────────────────────────────────────────┐
│ OVERALL ASSESSMENT                                                            │
├───────────────────────────────────────────────────────────────────────────────┤
│  Status: {status:<70} │
│  {status_message:<76} │
│                                                                               │
│  Critical Issues: {final_review.get('critical_count', 0):<58} │
│  Warnings:        {final_review.get('warning_count', 0):<58} │
│  Info Messages:   {final_review.get('info_count', 0):<58} │
│  Checkpoints:     {final_review.get('checkpoint_count', 0):<58} │
└───────────────────────────────────────────────────────────────────────────────┘

"""
    
    # Detailed Checkpoint Reviews
    checkpoint_names = [
        ("checkpoint1_agent2", "Checkpoint 1: Agent 2 (Weather & Yaw)"),
        ("checkpoint2_agent3", "Checkpoint 2: Agent 3 (Power & Optimization)"),
        ("checkpoint3_agent4", "Checkpoint 3: Agent 4 (Wake Flow)")
    ]
    
    for checkpoint_key, checkpoint_title in checkpoint_names:
        review = reviews.get(checkpoint_key)
        if review:
            severity = review.get("severity", "info").upper()
            findings = review.get("findings", [])
            llm_assessment = review.get("llm_assessment", "")
            
            report += f"""┌───────────────────────────────────────────────────────────────────────────────┐
│ {checkpoint_title:<77} │
├───────────────────────────────────────────────────────────────────────────────┤
│  Severity: {severity:<69} │
│                                                                               │
│  Findings:                                                                    │
"""
            
            for finding in findings:
                finding_type = finding.get("type", "info").upper()
                message = finding.get("message", "")
                rule = finding.get("rule", "unknown")
                
                # Wrap long messages
                lines = [message[i:i+70] for i in range(0, len(message), 70)]
                report += f"│  [{finding_type:8}] {lines[0]:<65} │\n"
                for line in lines[1:]:
                    report += f"│              {line:<65} │\n"
                report += f"│              Rule: {rule:<57} │\n"
            
            if llm_assessment:
                report += f"│                                                                               │\n"
                report += f"│  LLM Expert Assessment:                                                       │\n"
                # Wrap LLM assessment
                assessment_lines = [llm_assessment[i:i+72] for i in range(0, len(llm_assessment), 72)]
                for line in assessment_lines[:10]:  # Limit to 10 lines
                    report += f"│  {line:<77} │\n"
            
            report += f"└───────────────────────────────────────────────────────────────────────────────┘\n\n"
    
    # Recommendations
    if final_review and final_review.get("recommendations"):
        report += f"""┌───────────────────────────────────────────────────────────────────────────────┐
│ EXPERT RECOMMENDATIONS                                                        │
├───────────────────────────────────────────────────────────────────────────────┤
"""
        for i, rec in enumerate(final_review["recommendations"], 1):
            # Strip emojis
            clean_rec = rec.replace("✅", "").replace("⚠️", "").replace("🔴", "").strip()
            # Wrap long recommendations
            lines = [clean_rec[i:i+72] for i in range(0, len(clean_rec), 72)]
            report += f"│  {i}. {lines[0]:<75} │\n"
            for line in lines[1:]:
                report += f"│     {line:<75} │\n"
        
        report += f"└───────────────────────────────────────────────────────────────────────────────┘\n\n"
    
    report += f"""═══════════════════════════════════════════════════════════════════════════════
                         END OF EXPERT REVIEW REPORT
═══════════════════════════════════════════════════════════════════════════════
"""
    
    return report


def display_results(results):
    """Display previously computed results."""
    st.info("📊 Showing results from previous analysis. Click **Reset** to run a new analysis.")
    
    # Display summary report
    st.markdown("---")
    st.markdown("### 📊 Analysis Summary Report")
    st.markdown("*Summary of wind farm analysis based on optimizer results (Agents 1-3)*")
    display_summary_report(results)
    
    # Display Expert Review if available
    if "reviews" in results and results["reviews"].get("final_review"):
        st.markdown("---")
        st.markdown("### 🎓 Expert Review Assessment")
        st.markdown("*AI-powered validation of analysis outputs*")
        display_expert_review(results["reviews"])
        
        # Handle review export if triggered
        if st.session_state.get('trigger_review_export', False):
            st.session_state.trigger_review_export = False
            
            with st.spinner("Generating comprehensive review report..."):
                review_report = generate_review_export_report(results["reviews"], results)
                
                # Provide download button
                st.download_button(
                    label="💾 Download Expert Review Report",
                    data=review_report,
                    file_name=f"expert_review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    type="primary"
                )
                st.success("✅ Review report generated! Click above to download.")
    
    # Display chatbot interface
    st.markdown("---")
    st.markdown("### 🤖 Ask Questions About the Analysis")
    st.markdown("*Use the chatbot to ask questions about the optimization results and recommendations*")
    display_analysis_chatbot(results)
    
    # Show animation if available
    anim_path = os.path.join(SCRIPT_DIR, "wake_animation.gif")
    if os.path.exists(anim_path):
        st.markdown("---")
        st.markdown("### 🎬 Wake Flow Animation")
        st.image(anim_path, caption="Wake Flow Evolution (TT-OpInf Model)")


def display_checkpoint_review_short(review, checkpoint_name):
    """Display a short bullet-point checkpoint review immediately after agent completes."""
    if review is None or not review.get("findings"):
        return
    
    severity = review.get("severity", "info")
    findings = review.get("findings", [])
    
    # Count critical and warning findings
    critical_count = sum(1 for f in findings if f.get("type") == "critical")
    warning_count = sum(1 for f in findings if f.get("type") == "warning")
    
    # Display header with color code
    if severity == "critical":
        st.error(f"🔴 **Expert Review - {checkpoint_name}**: {critical_count} critical issue(s)")
    elif severity == "warning":
        st.warning(f"⚠️ **Expert Review - {checkpoint_name}**: {warning_count} warning(s)")
    else:
        st.success(f"✅ **Expert Review - {checkpoint_name}**: All checks passed")
    
    # Show top 3 findings as bullets
    display_findings = findings[:3]
    for finding in display_findings:
        finding_type = finding.get("type", "info")
        message = finding.get("message", "")[:100]  # Truncate to 100 chars
        
        if finding_type == "critical":
            st.markdown(f"- 🔴 {message}")
        elif finding_type == "warning":
            st.markdown(f"- ⚠️ {message}")
        else:
            st.markdown(f"- ✅ {message}")
    
    if len(findings) > 3:
        st.caption(f"... and {len(findings) - 3} more finding(s). See comprehensive review for details.")


def generate_review_export_report(reviews, results):
    """Generate a comprehensive review report for export."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = results.get("location", "Unknown")
    
    report = f"""╔═══════════════════════════════════════════════════════════════════════════════╗
║                    EXPERT REVIEW COMPREHENSIVE REPORT                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Generated: {timestamp:<64} ║
║  Location:  {location:<64} ║
╚═══════════════════════════════════════════════════════════════════════════════╝

"""
    
    # Final Review Summary
    final_review = reviews.get("final_review", {})
    if final_review:
        status = final_review.get("overall_status", "UNKNOWN")
        status_message = final_review.get("status_message", "")
        
        report += f"""┌───────────────────────────────────────────────────────────────────────────────┐
│ OVERALL ASSESSMENT                                                            │
├───────────────────────────────────────────────────────────────────────────────┤
│  Status: {status:<70} │
│  {status_message:<76} │
│                                                                               │
│  Critical Issues: {final_review.get('critical_count', 0):<58} │
│  Warnings:        {final_review.get('warning_count', 0):<58} │
│  Info Messages:   {final_review.get('info_count', 0):<58} │
│  Checkpoints:     {final_review.get('checkpoint_count', 0):<58} │
└───────────────────────────────────────────────────────────────────────────────┘

"""
    
    # Detailed Checkpoint Reviews
    checkpoint_names = [
        ("checkpoint1_agent2", "Checkpoint 1: Agent 2 (Weather & Yaw)"),
        ("checkpoint2_agent3", "Checkpoint 2: Agent 3 (Power & Optimization)"),
        ("checkpoint3_agent4", "Checkpoint 3: Agent 4 (Wake Flow)")
    ]
    
    for checkpoint_key, checkpoint_title in checkpoint_names:
        review = reviews.get(checkpoint_key)
        if review:
            severity = review.get("severity", "info").upper()
            findings = review.get("findings", [])
            llm_assessment = review.get("llm_assessment", "")
            
            report += f"""┌───────────────────────────────────────────────────────────────────────────────┐
│ {checkpoint_title:<77} │
├───────────────────────────────────────────────────────────────────────────────┤
│  Severity: {severity:<69} │
│                                                                               │
│  Findings:                                                                    │
"""
            
            for finding in findings:
                finding_type = finding.get("type", "info").upper()
                message = finding.get("message", "")
                rule = finding.get("rule", "unknown")
                
                # Wrap long messages
                lines = [message[i:i+70] for i in range(0, len(message), 70)]
                report += f"│  [{finding_type:8}] {lines[0]:<65} │\n"
                for line in lines[1:]:
                    report += f"│              {line:<65} │\n"
                report += f"│              Rule: {rule:<57} │\n"
            
            if llm_assessment:
                report += f"│                                                                               │\n"
                report += f"│  LLM Expert Assessment:                                                       │\n"
                # Wrap LLM assessment
                assessment_lines = [llm_assessment[i:i+72] for i in range(0, len(llm_assessment), 72)]
                for line in assessment_lines[:10]:  # Limit to 10 lines
                    report += f"│  {line:<77} │\n"
            
            report += f"└───────────────────────────────────────────────────────────────────────────────┘\n\n"
    
    # Recommendations
    if final_review and final_review.get("recommendations"):
        report += f"""┌───────────────────────────────────────────────────────────────────────────────┐
│ EXPERT RECOMMENDATIONS                                                        │
├───────────────────────────────────────────────────────────────────────────────┤
"""
        for i, rec in enumerate(final_review["recommendations"], 1):
            # Strip emojis
            clean_rec = rec.replace("✅", "").replace("⚠️", "").replace("🔴", "").strip()
            # Wrap long recommendations
            lines = [clean_rec[i:i+72] for i in range(0, len(clean_rec), 72)]
            report += f"│  {i}. {lines[0]:<75} │\n"
            for line in lines[1:]:
                report += f"│     {line:<75} │\n"
        
        report += f"└───────────────────────────────────────────────────────────────────────────────┘\n\n"
    
    report += f"""═══════════════════════════════════════════════════════════════════════════════
                         END OF EXPERT REVIEW REPORT
═══════════════════════════════════════════════════════════════════════════════
"""
    
    return report


def display_expert_review(reviews):
    """Display expert reviewer assessment with findings and recommendations."""
    final_review = reviews.get("final_review", {})
    
    if not final_review:
        return
    
    # Overall status
    status = final_review.get("overall_status", "UNKNOWN")
    status_message = final_review.get("status_message", "")
    
    # Color-coded status display
    if status == "APPROVED":
        st.success(f"✅ **{status}**: {status_message}")
    elif status == "WARNING":
        st.warning(f"⚠️ **{status}**: {status_message}")
    else:  # FAILED
        st.error(f"🔴 **{status}**: {status_message}")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Critical Issues", final_review.get("critical_count", 0))
    with col2:
        st.metric("Warnings", final_review.get("warning_count", 0))
    with col3:
        st.metric("Info Messages", final_review.get("info_count", 0))
    with col4:
        st.metric("Checkpoints", final_review.get("checkpoint_count", 0))
    
    # Key Findings
    key_findings = final_review.get("key_findings", [])
    if key_findings:
        with st.expander("🔍 Key Findings", expanded=True):
            for finding in key_findings:
                finding_type = finding.get("type", "info")
                message = finding.get("message", "")
                
                if finding_type == "critical":
                    st.error(f"🔴 **CRITICAL**: {message}")
                elif finding_type == "warning":
                    st.warning(f"⚠️ **WARNING**: {message}")
                else:
                    st.info(f"ℹ️ {message}")
    
    # Recommendations
    recommendations = final_review.get("recommendations", [])
    if recommendations:
        with st.expander("💡 Recommendations", expanded=True):
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    # Export Comprehensive Review Report
    st.markdown("---")
    st.markdown("**📥 Export Comprehensive Review Report**")
    
    col_export1, col_export2 = st.columns([3, 1])
    with col_export1:
        st.caption("Generate a detailed report with all checkpoint findings, LLM assessments, and recommendations (separate from main analysis report)")
    with col_export2:
        # Get parent results to pass context
        if st.button("📄 Generate & Download Report", type="secondary"):
            # This will be handled by the parent function that has access to full results
            st.session_state.trigger_review_export = True
            st.rerun()


if __name__ == "__main__":
    main()

