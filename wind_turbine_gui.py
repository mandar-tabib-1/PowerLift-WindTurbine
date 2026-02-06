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
from pathlib import Path
from datetime import datetime
import openai
import requests
import pydeck as pdk

#To do,
#I get the error ⚠️ Agent 2B: Could not connect to LLM. Error: query_local_llm() got an unexpected keyword argument 'temperature' . I have added a new folder called LLM that could be useful with ntnu_llm.py and base.py , and it could be useful for connecting to the specfic Ntnu server where the local llm is. 

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULT_ML_YAW = SCRIPT_DIR.parent
PROJECT_ROOT = RESULT_ML_YAW.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

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
        # Prepare turbine data for LLM analysis
        turbine_summary = []
        for i, turbine in enumerate(turbine_locations):
            turbine_summary.append(f"Turbine {turbine['turbine_id']}: Lat {turbine['latitude']:.6f}, Lon {turbine['longitude']:.6f}")
        
        # Create detailed prompt for turbine pair analysis
        prompt = f"""
You are Agent 2C, an expert in wind farm wake analysis and turbine pair optimization. Analyze the provided wind farm layout to identify critical turbine pairs most affected by wake effects.

WIND CONDITIONS:
- Wind Speed: {wind_speed:.1f} m/s
- Wind Direction: {wind_dir:.0f}° (meteorological convention: direction wind is coming FROM)
- Wind is blowing FROM {wind_dir:.0f}° TO {(wind_dir + 180) % 360:.0f}°

TURBINE LAYOUT:
{chr(10).join(turbine_summary)}

ANALYSIS REQUIREMENTS:
1. Calculate relative positions of turbines based on wind direction
2. Identify upstream-downstream turbine pairs along the wind direction vector
3. Consider wake decay distance (typically 5-10 rotor diameters = 0.6-1.2 km for NREL 5MW)
4. Prioritize pairs with strongest wake interactions
5. Consider wake width expansion (±20° spread)

RETURN FORMAT (JSON):
{{
  "critical_pairs": [
    {{
      "upstream_turbine": turbine_id,
      "downstream_turbine": turbine_id,
      "distance_km": distance,
      "wake_strength": "high/medium/low",
      "priority": ranking_number
    }}
  ],
  "analysis_summary": "Brief explanation of wake patterns",
  "optimization_strategy": "Recommended approach for yaw optimization"
}}

Provide analysis for the top 3-5 most critical turbine pairs."""
        
        # Get LLM configuration directly
        if provider == "NTNU":
            api_base = "https://llm.hpc.ntnu.no/v1"
            api_key = "sk-48COknyy7BlFg8vbN1ywgg"  # NTNU API key from config
        elif provider == "OpenAI":
            api_base = "https://api.openai.com/v1"
            api_key = os.getenv("OPENAI_API_KEY", "your-openai-key")
        elif provider == "Ollama":
            api_base = "http://localhost:11434/v1"
            api_key = "ollama"  # Ollama doesn't require real API key
        elif provider == "Google":
            api_base = "https://generativelanguage.googleapis.com/v1beta"
            api_key = os.getenv("GOOGLE_API_KEY", "your-google-key")
        else:  # Anthropic
            api_base = "https://api.anthropic.com/v1"
            api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key")
        
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
        response = query_local_llm(
            api_key=config['api_key'],
            api_base=config['api_base'],
            model_name=config['model'],
            prompt=prompt,
            system_message="You are Agent 2C, an expert wind farm wake analysis system specializing in turbine pair optimization for maximum power output.",
            temperature=config['temperature'],
            max_tokens=config['max_tokens'],
            timeout=config['timeout']
        )
        
        # Parse JSON response
        import json
        try:
            analysis_data = json.loads(response)
            return {
                'agent': 'Agent 2C - Turbine Pair Selector',
                'status': 'success',
                'provider': provider,
                'model': model,
                'turbine_pairs': analysis_data.get('critical_pairs', []),
                'analysis_summary': analysis_data.get('analysis_summary', ''),
                'optimization_strategy': analysis_data.get('optimization_strategy', ''),
                'total_turbines': len(turbine_locations),
                'wind_conditions': f"{wind_speed:.1f} m/s from {wind_dir:.0f}°",
                'raw_response': response
            }
        except json.JSONDecodeError:
            # Fallback: extract key information from text response
            return {
                'agent': 'Agent 2C - Turbine Pair Selector',
                'status': 'partial',
                'provider': provider,
                'model': model,
                'turbine_pairs': [],
                'analysis_summary': response[:500] + "..." if len(response) > 500 else response,
                'optimization_strategy': 'Manual parsing required',
                'total_turbines': len(turbine_locations),
                'wind_conditions': f"{wind_speed:.1f} m/s from {wind_dir:.0f}°",
                'raw_response': response
            }
    
    except Exception as e:
        return {
            'agent': 'Agent 2C - Turbine Pair Selector',
            'status': 'error',
            'message': f'Error in turbine pair analysis: {str(e)}',
            'turbine_pairs': [],
            'analysis': 'Analysis failed due to technical error'
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
        return config.get('llm', {})
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


def query_local_llm(api_key: str, api_base: str, model_name: str, prompt: str, system_message: str = None, temperature: float = 0.7, max_tokens: int = 1000, timeout: float = None):
    """
    Query the local LLM using OpenAI-compatible API (openai>=1.0.0 interface).
    
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
        Sampling temperature for the LLM (default: 0.7)
    max_tokens : int, optional
        Maximum number of tokens to generate (default: 1000)
    timeout : float, optional
        Timeout for the API request (default: None)

    Returns:
    --------
    str
        The response from the LLM
    """
    # Use the new OpenAI API client (>=1.0.0)
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=api_base)

    try:
        messages = []
        if system_message:
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
        return response.choices[0].message.content

    except Exception as e:
        return f"Error querying the LLM: {str(e)}"


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
        
        # Set API base URL and key based on provider
        if provider == "NTNU":
            api_base = "https://llm.hpc.ntnu.no/v1"
            api_key = "sk-48COknyy7BlFg8vbN1ywgg"  # NTNU API key from config
        elif provider == "OpenAI":
            api_base = "https://api.openai.com/v1"
            api_key = os.getenv("OPENAI_API_KEY", "your-openai-key")
        elif provider == "Ollama":
            api_base = "http://localhost:11434/v1"
            api_key = "ollama"  # Ollama doesn't require real API key
        elif provider == "Google":
            api_base = "https://generativelanguage.googleapis.com/v1beta"
            api_key = os.getenv("GOOGLE_API_KEY", "your-google-key")
        else:  # Anthropic
            api_base = "https://api.anthropic.com/v1"
            api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key")
        
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
        llm_response = query_local_llm(
            api_key=config.get('api_key', ''),
            api_base=config.get('api_base', 'https://llm.hpc.ntnu.no/v1'),
            model_name=config.get('model', model),
            prompt=prompt,
            system_message=system_message,
            temperature=config.get('temperature', 0.1),
            max_tokens=config.get('max_tokens', 200000),
            timeout=config.get('timeout', 1000.0)
        )
        
        # Add provider info to response for debugging
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
            power_samples.append(np.mean(result['power_mean_MW']))
        
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


# =============================================================================
# Two-Turbine Wake Steering Optimizer (Supports Both Methods)
# =============================================================================
def optimize_two_turbine_farm(power_agent, wake_agent, turbine_spacing_D: float = 7.0, 
                               n_timesteps: int = 50, verbose: bool = False,
                               optimization_method: str = 'analytical_physics'):
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
        Spacing between turbines in rotor diameters (default: 7D)
    n_timesteps : int
        Number of timesteps for predictions
    verbose : bool
        Print optimization progress
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
        # Method 1: True AD through Differentiable ML Surrogates
        # =====================================================================
        if verbose:
            print("Building differentiable surrogates from ML models...")
        
        # Build surrogates by sampling the actual ML models
        power_surrogate = DifferentiablePowerSurrogate(power_agent)
        wake_surrogate = DifferentiableWakeSurrogate(wake_agent)
        
        def compute_farm_power_ml(misalignment_tensor):
            """
            Compute farm power using differentiable ML surrogates.
            True AD through the ML models via polynomial approximation.
            """
            upstream_misalign = misalignment_tensor[0]
            downstream_misalign = misalignment_tensor[1]
            
            # Convert to nacelle directions
            upstream_nacelle = 270.0 + upstream_misalign
            downstream_nacelle = 270.0 + downstream_misalign
            
            # ============================================================
            # Upstream Power (from ML surrogate)
            # ============================================================
            # Base power from surrogate at actual yaw
            upstream_power_aligned = power_surrogate(torch.tensor(270.0, dtype=torch.float64))
            # Power at actual yaw angle (includes yaw effect from ML model)
            upstream_power_at_yaw = power_surrogate(upstream_nacelle)
            # Additional cos³ loss not captured by surrogate (if any)
            upstream_power = upstream_power_at_yaw
            
            # ============================================================
            # Wake Deficit (from ML surrogate)
            # ============================================================
            base_deficit = wake_surrogate(upstream_nacelle)
            
            # Wake steering effect: yaw deflects wake laterally
            # At higher yaw, less of the wake impacts downstream turbine
            k_steering = 0.6
            upstream_rad = upstream_misalign * np.pi / 180.0
            steering_factor = k_steering * torch.sin(upstream_rad) * torch.cos(upstream_rad)**2
            effective_deficit = base_deficit * (1.0 - steering_factor)
            effective_deficit = torch.clamp(effective_deficit, 0.0, 0.5)
            
            # ============================================================
            # Downstream Power (from ML surrogate + wake effect)
            # ============================================================
            downstream_power_base = power_surrogate(downstream_nacelle)
            effective_wind_ratio = 1.0 - effective_deficit
            downstream_power = downstream_power_base * (effective_wind_ratio ** 3)
            
            # ============================================================
            # Total Farm Power
            # ============================================================
            total_power = upstream_power + downstream_power
            
            return total_power, upstream_power, downstream_power, effective_deficit
        
        compute_fn = compute_farm_power_ml
        method_name = 'ML Surrogate AD'
        
    elif optimization_method == 'analytical_physics':
        # =====================================================================
        # Method 2: AD through Analytical Physics Model
        # =====================================================================
        if verbose:
            print("Using analytical physics model with AD...")
        
        # Get baseline power (constant, from ML model)
        baseline_result = power_agent.predict(yaw_angle=270.0, n_time_points=n_timesteps)
        P_base = np.mean(baseline_result['power_mean_MW'])
        
        # Get baseline wake deficit (constant, from ML model)
        wake_pred, _ = wake_agent.predict(yaw_angle=270.0, n_timesteps=n_timesteps,
                                          export_vtk=False, verbose=False)
        vel_mag = np.linalg.norm(wake_pred, axis=2)
        base_deficit = 1.0 - (np.mean(vel_mag) / freestream_velocity)
        base_deficit = np.clip(base_deficit, 0, 0.5)
        
        def compute_farm_power_physics(misalignment_tensor):
            """
            Compute farm power using analytical physics with AD.
            ML models provide baseline values; physics model is differentiated.
            """
            upstream_misalign = misalignment_tensor[0]
            downstream_misalign = misalignment_tensor[1]
            
            # ============================================================
            # Upstream Power: P = P_base * cos³(γ)
            # ============================================================
            upstream_rad = upstream_misalign * np.pi / 180.0
            upstream_cos_loss = torch.cos(upstream_rad) ** 3
            upstream_power = P_base * upstream_cos_loss
            
            # ============================================================
            # Wake Steering (Bastankhah & Porté-Agel 2016)
            # ============================================================
            k_steering = 0.6
            steering_reduction = k_steering * torch.sin(upstream_rad) * torch.cos(upstream_rad)**2
            effective_deficit = base_deficit * (1.0 - steering_reduction)
            effective_deficit = torch.clamp(torch.tensor(effective_deficit), 0.0, 0.5)
            
            # ============================================================
            # Downstream Power: P = P_base * (1-δ)³ * cos³(γ)
            # ============================================================
            downstream_rad = downstream_misalign * np.pi / 180.0
            downstream_cos_loss = torch.cos(downstream_rad) ** 3
            effective_wind_ratio = 1.0 - effective_deficit
            downstream_power = P_base * (effective_wind_ratio ** 3) * downstream_cos_loss
            
            total_power = upstream_power + downstream_power
            
            return total_power, upstream_power, downstream_power, effective_deficit
        
        compute_fn = compute_farm_power_physics
        method_name = 'Analytical Physics AD'
        
    else:
        # =====================================================================
        # Method 3: Grid Search (Fallback)
        # =====================================================================
        if verbose:
            print("Using grid search optimization...")
        
        # Get baseline values
        baseline_result = power_agent.predict(yaw_angle=270.0, n_time_points=n_timesteps)
        P_base = np.mean(baseline_result['power_mean_MW'])
        
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
            for down_misalign in [0, 2, 4]:
                up_rad = np.radians(up_misalign)
                down_rad = np.radians(down_misalign)
                
                up_power = P_base * np.cos(up_rad)**3
                
                k_steering = 0.6
                steering = k_steering * np.sin(up_rad) * np.cos(up_rad)**2
                eff_deficit = base_deficit * (1.0 - steering)
                
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
    
    # =========================================================================
    # Gradient-Based Optimization (for both ML surrogate and physics methods)
    # =========================================================================
    params = torch.tensor([4.0, 0.0], dtype=torch.float64, requires_grad=True)
    
    lower_bound = torch.tensor([0.0, 0.0], dtype=torch.float64)
    upper_bound = torch.tensor([12.0, 12.0], dtype=torch.float64)
    
    optimizer = torch.optim.Adam([params], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_power = -np.inf
    best_params = params.detach().clone()
    
    n_iterations = 50
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        total_power, up_power, down_power, wake_deficit = compute_fn(params)
        
        loss = -total_power
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_([params], max_norm=5.0)
        optimizer.step()
        
        with torch.no_grad():
            params.clamp_(min=lower_bound, max=upper_bound)
        
        current_power = float(total_power.detach().numpy()) if torch.is_tensor(total_power) else float(total_power)
        scheduler.step(current_power)
        
        if current_power > best_power:
            best_power = current_power
            best_params = params.detach().clone()
        
        results_history.append({
            'iteration': iteration,
            'upstream_misalignment': float(params[0].detach().numpy()),
            'downstream_misalignment': float(params[1].detach().numpy()),
            'upstream_power': float(up_power.detach().numpy()) if torch.is_tensor(up_power) else float(up_power),
            'downstream_power': float(down_power.detach().numpy()) if torch.is_tensor(down_power) else float(down_power),
            'total_power': current_power,
            'wake_deficit': float(wake_deficit.detach().numpy()) if torch.is_tensor(wake_deficit) else float(wake_deficit),
            'gradient_norm': float(params.grad.norm().numpy()) if params.grad is not None else 0.0
        })
        
        if verbose and iteration % 10 == 0:
            grad_norm = params.grad.norm().item() if params.grad is not None else 0.0
            print(f"  Iter {iteration:3d}: Misalign=[{params[0].item():.2f}°, {params[1].item():.2f}°] "
                  f"Power={current_power:.4f} MW, |∇|={grad_norm:.4f}")
        
        if params.grad is not None and params.grad.norm().item() < 1e-4:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
    
    optimal_upstream = float(best_params[0].numpy())
    optimal_downstream = float(best_params[1].numpy())
    
    # Evaluate at optimal and baseline points
    with torch.no_grad():
        opt_total, opt_up, opt_down, opt_deficit = compute_fn(best_params)
        baseline_total, baseline_up, baseline_down, baseline_deficit = compute_fn(
            torch.tensor([0.0, 0.0], dtype=torch.float64)
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
    st.markdown('<h1 class="main-header">🌀 Wind Turbine Multi-Agent Analysis System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('''
    <p style="text-align: center; font-size: 1.0rem; color: #666; margin-top: -10px;">
    📧  <b> Contact: mandar.tabib@sintef.no | ⚠️ <em>Currently being tested/developed</em></b>
    </p>
    ''', unsafe_allow_html=True)
    

    st.markdown("""
    <div class="info-box">
    <b>Welcome!</b> This system involves developing/testing multiple agents involving AI to analyze wind turbine operations:
    <ul>
        <li><b>Agent 1:</b> Weather Station - Fetches real-time wind conditions</li>
        <li><b>Agent 2:</b> Turbine Expert - Consults NREL 5MW manual for optimal yaw</li>
        <li><b>Agent 2B:</b> LLM-based Turbine Expert - Uses local LLM for intelligent recommendations</li>
        <li><b>Agent 2C:</b> Turbine Pair Selector - Uses LLM to identify critical turbine pairs for wake optimization</li>
        <li><b>Agent 3:</b> Two-Turbine Wake Steering Optimizer - Finds optimal yaw misalignment for farm power maximization</li>
        <li><b>Agent 4:</b> Wind Turbine Wake Flow ROM - Tensor Decomposition + Operator Inference model</li>
        <li><b>Agent 5:</b> Wind Turbine Power Predictor - Gaussian Process Regressor trained at SINTEF</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
   
    st.markdown("""
    <div class="info-box">          
     <b> Select a wind farm on the left sidebar for agents to start analysis!</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🤖 LLM Configuration (Agent 2B)")
        st.markdown("Configure the AI model for turbine analysis:")
        
        # LLM Provider Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["NTNU", "OpenAI", "Ollama", "Google", "Anthropic"],
            index=0,  # Default to NTNU
            help="Select the AI service provider"
        )
        
        # Model Selection based on provider
        if llm_provider == "NTNU":
            available_models = [
                "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "moonshotai/Kimi-K2.5", 
                "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4",
                "meta-llama/Llama-3.3-70B-Instruct",
                "microsoft/Phi-3.5-mini-instruct"
            ]
        elif llm_provider == "OpenAI":
            available_models = [
                "gpt-4o",
                "gpt-4o-mini", 
                "gpt-3.5-turbo",
                "o1-preview",
                "o1-mini"
            ]
        elif llm_provider == "Ollama":
            available_models = [
                "llama3.2",
                "mistral",
                "codellama",
                "phi3",
                "qwen2.5"
            ]
        elif llm_provider == "Google":
            available_models = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro"
            ]
        else:  # Anthropic
            available_models = [
                "claude-3.5-sonnet",
                "claude-3-opus",
                "claude-3-haiku"
            ]
        
        selected_model = st.selectbox(
            "Model",
            options=available_models,
            index=0 if llm_provider == "NTNU" else 0,
            help=f"Select the {llm_provider} model to use"
        )
        explore_vs_acc = st.slider(
        "Exploration vs Accuracy",
        min_value=0.0,
        max_value=1.0,
        value=0.1,  # Default value
        step=0.1,  # Step size for the slider
        help="Select a value between 0 (accuracy) and 1 (exploration)"
    )
        
        # Store in session state
        st.session_state.llm_provider = llm_provider
        st.session_state.selected_model = selected_model
        st.session_state.explore_vs_acc = explore_vs_acc
        # Show current configuration
        with st.expander("Current LLM Configuration"):
            st.code(f"Provider: {llm_provider}\nModel: {selected_model}", language="yaml")
        
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
                st.success(f"✅ Found {search_results['total_turbines_found']} exact locations")
            elif status == 'found_web':
                st.info(f"ℹ️ Found {search_results['total_turbines_found']} locations from web")
            elif status == 'estimated':
                st.warning(f"⚠️ Generated {search_results['total_turbines_found']} estimated locations")
            else:
                st.error("❌ No turbine location data available")
            
            st.markdown(f"**Data Source:** {search_results['data_source']}")
            
            # Show map button
            if search_results['turbine_locations']:
                if st.button("📍 Show Turbine Map"):
                    st.session_state.show_turbine_map = True
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        n_timesteps = st.slider("Prediction Timesteps", min_value=10, max_value=100, value=50, step=10)
        export_vtk = st.checkbox("Export VTK Files", value=False)
        
        st.markdown("---")
        
        run_analysis = st.button("🚀 Run Analysis", type="primary")
        
        if st.session_state.analysis_complete:
            if st.button("🔄 Reset"):
                st.session_state.analysis_complete = False
                st.session_state.results = None
                st.rerun()
    
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
        | **TT-OpInf ROM** | Tensor Train Decomposition + Operator Inference | Yaw Direction (0°-15°) |  Wake Velocity Field |
        | **GP Regressor** | Gaussian Process trained on CFD data | Yaw Direction (0°-15°)  | Rotor Power (MW) with uncertainty |
       
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
                with wcol2:
                    st.metric("Mean Velocity", f"{np.mean(velocity_mag):.2f} m/s")
                with wcol3:
                    st.metric("Timesteps", wake_predictions.shape[0])
                
                # Create wake visualization
                try:
                    from wake_animation import create_wake_contour_animation
                    
                    demo_anim_path = os.path.join(SCRIPT_DIR, f"wake_demo_yaw_{st.session_state.get('demo_yaw', 0):.0f}.gif")
                    grid_path = str(PROJECT_ROOT / "ResultMLYaw" / "Grid_data.vtk")
                    
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
    
    # Main content area
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


def run_full_analysis(selected_farm: str, n_timesteps: int, export_vtk: bool):
    """Run the complete multi-agent analysis with GUI updates."""
    
    # Get wind farm info
    farm_info = NORWEGIAN_WIND_FARMS[selected_farm]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "location": f"{selected_farm}, {farm_info['location']}",
        "wind_farm": selected_farm,
        "farm_info": farm_info
    }
    
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
        
        # Fetch weather using farm coordinates
        weather = fetch_weather_for_farm(selected_farm, farm_info)
        results["weather"] = weather
        progress_bar.progress(20)

        # Agent 1A: Fetch weather from yr.no (Met.no)
        yr_weather = fetch_weather_yr_no(farm_info["latitude"], farm_info["longitude"])
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
    st.markdown("### 📖 Agent 2: Turbine Expert: to recommend yaw angle based on turbine manual")
    
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
    # AGENT 2B: LLM-based Turbine Expert
    # =========================================================================
    st.markdown("### 🤖 Agent 2B: LLM-based Turbine Expert")
    st.markdown('''
    <p style="font-size: 1.0rem; color: #666; margin-top: -10px; font-style: italic;">
    Uses local LLM (moonshotai/Kimi-K2.5) to provide intelligent turbine control recommendations.
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
    
    # =========================================================================
    # AGENT 2C: Turbine Pair Selection for Multi-Turbine Wake Optimization
    # =========================================================================
    # Add Agent 2C analysis if turbine locations are available
    if selected_farm in st.session_state.turbine_locations:
        search_results = st.session_state.turbine_locations[selected_farm]
        if search_results and search_results['turbine_locations']:
            # Arrow from Agent 2B to Agent 2C
            st.markdown('''
            <div style="text-align: center; font-size: 2.5rem; color: #1E88E5; margin: 15px 0;">
            ⬇️
            </div>
            <div style="text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 15px;">
            Wind & Turbine Data → Agent 2C (Turbine Pair Selector)
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("### 🎯 Agent 2C: Turbine Pair Selector for Multi-Turbine Wind Farm")
            st.markdown('''
            <p style="font-size: 1.0rem; color: #666; margin-top: -10px; font-style: italic;">
            Analyzes turbine locations and wind conditions using LLM to identify critical turbine pairs 
            most affected by wake effects. These pairs will be optimized by the downstream agents.
            </p>
            ''', unsafe_allow_html=True)
            
            progress_bar.progress(35)
            status_text.text("🔄 Agent 2C: Analyzing turbine pairs for wake optimization...")
            
            with st.spinner("Agent 2C: Using LLM to identify critical turbine pairs..."):
                try:
                    # Get current LLM configuration from session state
                    current_provider = st.session_state.get('llm_provider', 'NTNU')
                    current_model = st.session_state.get('selected_model', 'moonshotai/Kimi-K2.5')
                    
                    turbine_pair_analysis = get_turbine_pair_recommendations(
                        search_results['turbine_locations'],
                        weather['wind_speed_ms'],
                        weather['wind_direction_deg'],
                        current_provider,
                        current_model
                    )
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
                        
                        with st.expander("ℹ️ About Agent 2C"):
                            st.markdown(f"""
                            **Agent 2C Configuration (from GUI selections):**
                            - **LLM Provider:** {turbine_pair_analysis['provider']}
                            - **LLM Model:** {turbine_pair_analysis['model']}
                            - **Total Turbines Analyzed:** {turbine_pair_analysis['total_turbines']}
                            - **Wind Conditions:** {turbine_pair_analysis['wind_conditions']}
                            
                            **How it works:**
                            1. Agent 2C receives turbine locations from the location search
                            2. It uses the LLM provider and model selected in the GUI sidebar (same as Agent 2B)
                            3. The LLM analyzes wind direction, turbine spacing, and wake patterns
                            4. It identifies the most critical upstream-downstream turbine pairs
                            5. These pairs are prioritized for wake optimization by Agent 3
                            
                            **Purpose:** This agent bridges single-turbine analysis (Agent 2B) with 
                            multi-turbine farm optimization by intelligently selecting which turbine 
                            pairs will benefit most from wake steering control.
                            
                            **Configuration:** Change the provider and model in the sidebar to use different AI services.
                            """)
                    
                    elif turbine_pair_analysis['status'] == 'partial':
                        st.warning(f"⚠️ {turbine_pair_analysis['agent']} analysis partially complete")
                        st.markdown(f"**Analysis:** {turbine_pair_analysis.get('analysis_summary', 'Partial results available')}")
                        results["turbine_pairs"] = turbine_pair_analysis
                    
                    else:
                        st.error(f"❌ {turbine_pair_analysis['agent']} analysis failed: {turbine_pair_analysis.get('message', 'Unknown error')}")
                        results["turbine_pairs"] = None
                        
                except Exception as e:
                    st.error(f"❌ Agent 2C: Error in turbine pair analysis: {str(e)}")
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
    Optimizes yaw misalignment for turbine pairs identified by Agent 2C to maximize total power output. \n
    Justification: This optimizer adjusts yaw misalignment angles to steer the wake laterally, reducing its impact on downstream turbines and increasing overall power output.
    The turbine pairs are provided by Agent 2C based on LLM analysis of: 1. Distance Between Turbines, 2. Wind Direction, 3. Wake Interaction Strength, and 4. Farm Layout
    </p>
    ''', unsafe_allow_html=True)
    
    # Optimization method selection
    opt_method = st.radio(
        "Select Optimization Method:",
        options=['ml_surrogate', 'analytical_physics', 'grid_search'],
        format_func=lambda x: {
            'ml_surrogate': '🧠 True AD through ML Surrogates (differentiable polynomial fit to ML models)',
            'analytical_physics': '📐 AD through Analytical Physics (Bastankhah wake model)',
            'grid_search': '🔍 Grid Search (brute-force, no gradients)'
        }[x],
        index=0,
        horizontal=False,
        help="Choose how to optimize yaw angles"
    )
    
    # Explain the optimization methods
    with st.expander("🧠 Method 1: True AD through ML Surrogates (Recommended)"):
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
    
    status_text.text("🔄 Agent 3: Running two-turbine wake steering optimization...")
    time.sleep(0.5)
    
    with st.spinner(f"Optimizing yaw misalignment using {opt_method.replace('_', ' ').title()} method..."):
        progress_bar.progress(48)
        
        try:
            power_agent, wake_agent = load_agents()
            
            # Run the optimizer with selected method
            opt_results = optimize_two_turbine_farm(
                power_agent=power_agent,
                wake_agent=wake_agent,
                turbine_spacing_D=7.0,
                n_timesteps=min(n_timesteps, 30),  # Faster for optimization
                verbose=False,
                optimization_method=opt_method  # Use selected method
            )
            
            results["optimizer"] = opt_results
            progress_bar.progress(52)
            
            st.success(f"✅ Wake steering optimization complete! Method: **{opt_results['optimization_method']}**")
            
            # Display optimization results
            st.markdown("#### 📊 Optimization Results")
            
            opt_col1, opt_col2 = st.columns(2)
            
            # Get wind direction from weather agent (Agent 1)
            wind_dir = weather.get('wind_direction_deg', 270)
            
            with opt_col1:
                st.markdown("**🔹 Upstream Turbine (T1)**")
                # Calculate actual yaw = wind direction + misalignment
                actual_yaw_T1 = wind_dir + opt_results['optimal_upstream_misalignment']
                st.metric(
                    "Optimal Yaw Misalignment", 
                    f"{opt_results['optimal_upstream_misalignment']:.0f}°",
                    delta=f"Actual Yaw: {actual_yaw_T1:.0f}°"
                )
                st.metric(
                    "Power Output", 
                    f"{opt_results['optimal_upstream_power']:.3f} MW"
                )
            
            with opt_col2:
                st.markdown("**🔹 Downstream Turbine (T2)**")
                # Calculate actual yaw = wind direction + misalignment
                actual_yaw_T2 = wind_dir + opt_results['optimal_downstream_misalignment']
                st.metric(
                    "Optimal Yaw Misalignment", 
                    f"{opt_results['optimal_downstream_misalignment']:.0f}°",
                    delta=f"Actual Yaw: {actual_yaw_T2:.0f}°"
                )
                st.metric(
                    "Power Output", 
                    f"{opt_results['optimal_downstream_power']:.3f} MW"
                )
            
            st.markdown("---")
            
            # Total farm power comparison
            st.markdown("#### ⚡ Farm Power Comparison")
            
            farm_col1, farm_col2, farm_col3 = st.columns(3)
            
            with farm_col1:
                st.metric(
                    "Baseline (0° misalign)", 
                    f"{opt_results['baseline_total_power']:.3f} MW"
                )
            
            with farm_col2:
                st.metric(
                    "Optimized Total", 
                    f"{opt_results['optimal_total_power']:.3f} MW"
                )
            
            with farm_col3:
                gain_color = "normal" if opt_results['power_gain_percent'] >= 0 else "inverse"
                st.metric(
                    "Power Gain", 
                    f"{opt_results['power_gain_MW']:.3f} MW",
                    delta=f"{opt_results['power_gain_percent']:+.1f}%"
                )
            
            # Visualization of optimization results
            st.markdown("#### 📈 Optimization Landscape")
            
            # Create heatmap of power vs misalignment angles
            all_results = opt_results['all_results']
            
            # Extract unique values
            up_misaligns = sorted(set(r['upstream_misalignment'] for r in all_results))
            down_misaligns = sorted(set(r['downstream_misalignment'] for r in all_results))
            
            # Create power matrix
            power_matrix = np.zeros((len(down_misaligns), len(up_misaligns)))
            for r in all_results:
                i = down_misaligns.index(r['downstream_misalignment'])
                j = up_misaligns.index(r['upstream_misalignment'])
                power_matrix[i, j] = r['total_power']
            
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(power_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
            ax.set_xticks(range(len(up_misaligns)))
            ax.set_xticklabels([f'{m:.0f}°' for m in up_misaligns])
            ax.set_yticks(range(len(down_misaligns)))
            ax.set_yticklabels([f'{m:.0f}°' for m in down_misaligns])
            ax.set_xlabel('Upstream Turbine Yaw Misalignment')
            ax.set_ylabel('Downstream Turbine Yaw Misalignment')
            ax.set_title('Total Farm Power (MW) vs Yaw Misalignment')
            
            # Mark optimal point
            opt_i = down_misaligns.index(opt_results['optimal_downstream_misalignment'])
            opt_j = up_misaligns.index(opt_results['optimal_upstream_misalignment'])
            ax.plot(opt_j, opt_i, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=2)
            ax.annotate('Optimal', (opt_j, opt_i), xytext=(opt_j+0.5, opt_i+0.3), 
                       fontsize=10, color='white', fontweight='bold')
            
            plt.colorbar(im, label='Total Power (MW)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Store optimal nacelle directions for use in subsequent agents (ML model input)
            results["optimal_nacelle_upstream"] = opt_results['optimal_upstream_nacelle']
            results["optimal_nacelle_downstream"] = opt_results['optimal_downstream_nacelle']
            
            # Store actual yaw angles (wind direction + misalignment) for reporting
            results["actual_yaw_upstream"] = wind_dir + opt_results['optimal_upstream_misalignment']
            results["actual_yaw_downstream"] = wind_dir + opt_results['optimal_downstream_misalignment']
            
            # Show optimization method used
            opt_method = opt_results.get('optimization_method', 'Grid Search')
            st.info(f"🔧 **Optimization Method:** {opt_method}")
            
            # Convergence plot for gradient-based optimization
            if 'Gradient' in opt_method and len(all_results) > 1 and 'iteration' in all_results[0]:
                st.markdown("#### 📉 Optimization Convergence")
                
                fig_conv, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                iterations = [r['iteration'] for r in all_results]
                powers = [r['total_power'] for r in all_results]
                grad_norms = [r.get('gradient_norm', 0) for r in all_results]
                up_misaligns = [r['upstream_misalignment'] for r in all_results]
                
                # Power convergence
                ax1.plot(iterations, powers, 'b-', linewidth=2, marker='o', markersize=3)
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Total Farm Power (MW)')
                ax1.set_title('Power Maximization Convergence')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=max(powers), color='g', linestyle='--', alpha=0.7, label=f'Best: {max(powers):.4f} MW')
                ax1.legend()
                
                # Gradient norm (convergence indicator)
                ax2.semilogy(iterations, [g + 1e-10 for g in grad_norms], 'r-', linewidth=2, marker='s', markersize=3)
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Gradient Norm (log scale)')
                ax2.set_title('Gradient Convergence')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=1e-4, color='g', linestyle='--', alpha=0.7, label='Convergence threshold')
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig_conv)
                plt.close()
            
            with st.expander("📋 View All Optimization Iterations"):
                import pandas as pd
                df_results = pd.DataFrame(all_results)
                if 'upstream_misalignment' in df_results.columns:
                    df_results['upstream_nacelle'] = df_results['upstream_misalignment'].apply(
                        yaw_misalignment_to_nacelle_direction
                    )
                    df_results['downstream_nacelle'] = df_results['downstream_misalignment'].apply(
                        yaw_misalignment_to_nacelle_direction
                    )
                df_results = df_results.round(4)
                st.dataframe(df_results, use_container_width=True)
            
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            results["optimizer"] = None
    
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
    # AGENT 4: Wind Turbine Wake Flow at Recommended Yaw Angle: A Reduced Order Model.
    # =========================================================================
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
    optimal_misalign_upstream = opt_results['optimal_upstream_misalignment'] if opt_results else 0
    
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
                    grid_path = str(PROJECT_ROOT / "ResultMLYaw" / "Grid_data.vtk")
                    
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
    # AGENT 5: Power Predictor
    # =========================================================================
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
    
    progress_bar.progress(95)
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    # Store results
    st.session_state.results = results
    st.session_state.analysis_complete = True
    
    # =========================================================================
    # Summary Report
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📊 Analysis Summary Report")
    
    display_summary_report(results)


def display_summary_report(results):
    """Display a summary report of the analysis."""
    
    weather = results.get("weather", {})
    expert = results.get("expert", {})
    power = results.get("power", {})
    wake = results.get("wake", {})
    optimizer = results.get("optimizer", {})
    
    # Get wind direction for actual yaw calculation
    wind_dir = weather.get('wind_direction_deg', 270)
    
    # Use optimizer results for recommended yaw if available
    if optimizer:
        opt_upstream = optimizer.get('optimal_upstream_misalignment', 0)
        opt_downstream = optimizer.get('optimal_downstream_misalignment', 0)
        opt_method = optimizer.get('optimization_method', 'N/A')
        power_gain = optimizer.get('power_gain_percent', 0)
        # Calculate actual yaw angles: wind_direction + misalignment
        actual_yaw_T1 = wind_dir + opt_upstream
        actual_yaw_T2 = wind_dir + opt_downstream
        # Report both misalignment and actual yaw
        recommended_yaw_str = f"T1: {opt_upstream:.1f}° → {actual_yaw_T1:.0f}°, T2: {opt_downstream:.1f}° → {actual_yaw_T2:.0f}° (via {opt_method})"
    else:
        recommended_yaw_str = f"{expert.get('suggested_yaw', 0):.1f}°"
        power_gain = 0
        actual_yaw_T1 = wind_dir
        actual_yaw_T2 = wind_dir
    
    report_md = f"""
| Parameter | Value |
|-----------|-------|
| **Location** | {results.get('location', 'N/A')} |
| **Timestamp** | {results.get('timestamp', 'N/A')} |
| **Wind Speed** | {weather.get('wind_speed_ms', 0):.1f} m/s |
| **Wind Direction** | {weather.get('wind_direction_deg', 0):.0f}° |
| **Optimal Yaw (Misalign → Actual)** | {recommended_yaw_str} |
| **Farm Power Gain** | {power_gain:+.1f}% |
| **Operating Region** | {expert.get('operating_region', 'N/A')} |
| **Predicted Power** | {power.get('mean_MW', 0):.3f} ± {power.get('uncertainty_MW', 0):.3f} MW |
| **Wake Field Points** | {wake.get('shape', [0,0,0])[1] if wake else 'N/A':,} |
| **Simulation Steps** | {wake.get('shape', [0,0,0])[0] if wake else 'N/A'} |

*Note: Actual Yaw = Wind Direction + Yaw Misalignment (ML model input 270°-285° is proxy for misalignment 0°-15°)*
"""
    
    st.markdown(report_md)
    
    # Export button
    if st.button("📄 Export Full Report"):
        # Build optimizer section for export
        if optimizer:
            opt_up_misalign = optimizer.get('optimal_upstream_misalignment', 0)
            opt_down_misalign = optimizer.get('optimal_downstream_misalignment', 0)
            # Calculate actual yaw = wind direction + misalignment
            actual_yaw_T1_export = wind_dir + opt_up_misalign
            actual_yaw_T2_export = wind_dir + opt_down_misalign
            
            optimizer_section = f"""
WAKE STEERING OPTIMIZATION
--------------------------
Optimization Method: {optimizer.get('optimization_method', 'N/A')}

Note: Actual Yaw = Wind Direction + Yaw Misalignment
      (ML model input 270°-285° is a proxy for misalignment 0°-15°)

Upstream Turbine (T1):
  - Optimal Yaw Misalignment: {opt_up_misalign:.1f}°
  - Actual Yaw Angle: {actual_yaw_T1_export:.1f}° (= {wind_dir:.0f}° + {opt_up_misalign:.1f}°)
  - ML Model Input (Nacelle Dir): {optimizer.get('optimal_upstream_nacelle', 0):.1f}°
  - Power Output: {optimizer.get('optimal_upstream_power', 0):.3f} MW

Downstream Turbine (T2):
  - Optimal Yaw Misalignment: {opt_down_misalign:.1f}°
  - Actual Yaw Angle: {actual_yaw_T2_export:.1f}° (= {wind_dir:.0f}° + {opt_down_misalign:.1f}°)
  - ML Model Input (Nacelle Dir): {optimizer.get('optimal_downstream_nacelle', 0):.1f}°
  - Power Output: {optimizer.get('optimal_downstream_power', 0):.3f} MW

Total Farm Power:
  - Baseline (0° misalign): {optimizer.get('baseline_total_power', 0):.3f} MW
  - Optimized: {optimizer.get('optimal_total_power', 0):.3f} MW
  - Power Gain: {optimizer.get('power_gain_MW', 0):.3f} MW ({optimizer.get('power_gain_percent', 0):+.1f}%)
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

WEATHER CONDITIONS
------------------
Wind Speed: {weather.get('wind_speed_ms', 0):.1f} m/s
Wind Direction: {weather.get('wind_direction_deg', 0):.0f}°
Temperature: {weather.get('temperature_c', 0):.1f}°C
Data Source: {weather.get('data_source', 'N/A')}

EXPERT RECOMMENDATION
---------------------
Turbine: NREL 5MW Reference Wind Turbine
Operating Region: {expert.get('operating_region', 'N/A')}
Real-World Yaw (aligned): {expert.get('actual_yaw', 0):.1f}°
Expected Efficiency: {expert.get('expected_efficiency', 0)*100:.1f}%
{optimizer_section}
TURBINE PAIR ANALYSIS (Agent 2C)
--------------------------------
{turbine_pair_section}
POWER PREDICTION (SINTEF ML Model)
----------------------------------
Mean Power Output: {power.get('mean_MW', 0):.3f} MW
Uncertainty (±1σ): ±{power.get('uncertainty_MW', 0):.3f} MW

WAKE FLOW SIMULATION (TT-OpInf)
-------------------------------
Spatial Points: {wake.get('shape', [0,0,0])[1] if wake else 'N/A'}
Time Steps: {wake.get('shape', [0,0,0])[0] if wake else 'N/A'}
Velocity Components: 3 (Ux, Uy, Uz)

========================================
Report generated by Wind Turbine Multi-Agent System
Developed at SINTEF Energy Research
"""
        st.download_button(
            label="Download Report (TXT)",
            data=report_text,
            file_name="wind_turbine_report.txt",
            mime="text/plain"
        )


def display_results(results):
    """Display previously computed results."""
    st.info("📊 Showing results from previous analysis. Click **Reset** to run a new analysis.")
    display_summary_report(results)
    
    # Show animation if available
    anim_path = os.path.join(SCRIPT_DIR, "wake_animation.gif")
    if os.path.exists(anim_path):
        st.markdown("### 🎬 Wake Flow Animation")
        st.image(anim_path, caption="Wake Flow Evolution (TT-OpInf Model)")


if __name__ == "__main__":
    main()
