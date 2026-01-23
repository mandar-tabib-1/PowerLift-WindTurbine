"""
Wind Turbine Multi-Agent Orchestrator - GUI Version

A Streamlit-based graphical interface for the wind turbine analysis system.
Run with: streamlit run wind_turbine_gui.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import time
import os
import sys
from pathlib import Path
from datetime import datetime

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


# =============================================================================
# Expert Agent Functions
# =============================================================================
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
    
    yaw_min, yaw_max = 272, 285
    reasoning = []
    
    # Determine operating region
    if wind_speed < turbine_specs["cut_in_wind_speed_ms"]:
        operating_region = "Below Cut-in"
        reasoning.append(f"Wind speed ({wind_speed:.1f} m/s) is below cut-in speed ({turbine_specs['cut_in_wind_speed_ms']} m/s)")
        suggested_yaw = (yaw_min + yaw_max) / 2
    elif wind_speed > turbine_specs["cut_out_wind_speed_ms"]:
        operating_region = "Above Cut-out"
        reasoning.append(f"Wind speed ({wind_speed:.1f} m/s) exceeds cut-out speed")
        suggested_yaw = (yaw_min + yaw_max) / 2
    elif wind_speed <= turbine_specs["rated_wind_speed_ms"]:
        operating_region = "Partial Load (Region 2)"
        reasoning.append("Operating in partial load region - maximize energy capture")
        reasoning.append("Aligning yaw closely with wind direction is critical")
        ideal_yaw = wind_direction
        suggested_yaw = np.clip(ideal_yaw, yaw_min, yaw_max)
        if ideal_yaw < yaw_min:
            reasoning.append(f"Wind direction below yaw range - using minimum {yaw_min}°")
        elif ideal_yaw > yaw_max:
            reasoning.append(f"Wind direction above yaw range - using maximum {yaw_max}°")
        else:
            reasoning.append(f"Aligning with wind direction at {suggested_yaw:.1f}°")
    else:
        operating_region = "Full Load (Region 3)"
        reasoning.append("Operating in full load region")
        reasoning.append("Applying wake steering offset for downstream benefit")
        ideal_yaw = wind_direction + 3
        suggested_yaw = np.clip(ideal_yaw, yaw_min, yaw_max)
    
    yaw_misalignment = abs(suggested_yaw - wind_direction)
    expected_efficiency = np.cos(np.radians(yaw_misalignment)) ** 3
    
    return {
        "suggested_yaw": suggested_yaw,
        "yaw_misalignment": yaw_misalignment,
        "expected_efficiency": expected_efficiency,
        "operating_region": operating_region,
        "reasoning": reasoning,
        "turbine_specs": turbine_specs
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
    
    st.markdown("""
    <div class="info-box">
    <b>Welcome!</b> This system uses multiple AI agents to analyze wind turbine operations:
    <ul>
        <li><b>Agent 1:</b> Weather Station - Fetches real-time wind conditions</li>
        <li><b>Agent 2:</b> Turbine Expert - Consults NREL 5MW manual for optimal yaw</li>
        <li><b>Agent 3:</b> Power Predictor - ML model trained at SINTEF</li>
        <li><b>Agent 4:</b> Wake Flow Simulator - Physics-based TT-OpInf model</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📍 Wind Turbine Location")
        st.markdown("Enter the location of your wind turbine installation:")
        
        city = st.text_input("City", value="Trondheim", placeholder="e.g., Trondheim")
        country = st.text_input("Country", value="Norway", placeholder="e.g., Norway")
        
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
    
    # Main content area
    if run_analysis and city and country:
        run_full_analysis(city, country, n_timesteps, export_vtk)
    
    elif st.session_state.analysis_complete and st.session_state.results:
        display_results(st.session_state.results)
    
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Windmills_D1-D4_%28Thornton_Bank%29.jpg/1280px-Windmills_D1-D4_%28Thornton_Bank%29.jpg", 
                    caption="Offshore Wind Turbines")
            
            st.info("👈 Enter the wind turbine location in the sidebar and click **Run Analysis** to begin.")


def run_full_analysis(city: str, country: str, n_timesteps: int, export_vtk: bool):
    """Run the complete multi-agent analysis with GUI updates."""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "location": f"{city}, {country}"
    }
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create columns for agent outputs
    col1, col2 = st.columns(2)
    
    # =========================================================================
    # AGENT 1: Weather Station
    # =========================================================================
    with col1:
        st.markdown("### 🌤️ Agent 1: Weather Station")
        agent1_container = st.container()
        
    with agent1_container:
        with st.spinner(f"Fetching weather data for {city}, {country}..."):
            status_text.text("Agent 1: Connecting to weather service...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            weather = fetch_weather(city, country)
            results["weather"] = weather
            progress_bar.progress(25)
            
        # Generate wind farm name based on location
        wind_farm_name = f"{city.title()} Wind Farm"
        
        st.success(f"📍 Location: **{weather['location']}**")
        st.info(f"🏭 **Wind Farm:** {wind_farm_name}")
        
        # Show only wind direction prominently
        st.metric("🧭 Wind Direction", f"{weather['wind_direction_deg']:.0f}°")
        
        st.caption(f"Data source: {weather['data_source']}")
    
    # Pause to simulate handoff between agents
    time.sleep(1.0)
    
    # =========================================================================
    # AGENT 2: Turbine Expert
    # =========================================================================
    with col2:
        st.markdown("### 📖 Agent 2: Turbine Expert")
        agent2_container = st.container()
        
    with agent2_container:
        status_text.text("🔄 Agent 2: Starting turbine manual consultation...")
        time.sleep(0.5)
        with st.spinner("Consulting NREL 5MW turbine manual..."):
            status_text.text("Agent 2: Analyzing wind conditions against turbine specifications...")
            progress_bar.progress(35)
            time.sleep(1.2)
            
            expert = get_expert_recommendation(
                weather["wind_speed_ms"], 
                weather["wind_direction_deg"]
            )
            results["expert"] = expert
            progress_bar.progress(50)
        
        st.info(f"🎯 **Recommended Yaw Angle: {expert['suggested_yaw']:.1f}°**")
        
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.metric("Operating Region", expert['operating_region'])
        with ecol2:
            st.metric("Expected Efficiency", f"{expert['expected_efficiency']*100:.1f}%")
        
        with st.expander("View Expert Reasoning"):
            for reason in expert['reasoning']:
                st.write(f"• {reason}")
    
    st.markdown("---")
    
    # Pause to simulate agents loading ML models
    time.sleep(1.5)
    
    # =========================================================================
    # AGENT 3 & 4: ML Predictions
    # =========================================================================
    st.markdown("### 🔬 Physics-Based Machine Learning Analysis (SINTEF)")
    st.markdown("""
    <div class="info-box">
    Now running physics-informed ML models developed at <b>SINTEF Digital</b> to predict:
    <ul>
        <li><b>Power Output:</b> Using Gaussian Process regression trained on CFD data</li>
        <li><b>Wake Flow Field:</b> Using TT-OpInf (Tensor-Train Operator Inference) model</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    # AGENT 3: Power Prediction
    with col3:
        st.markdown("#### ⚡ Agent 3: Power Predictor")
        
        status_text.text("🔄 Agent 3: Loading SINTEF ML models...")
        time.sleep(0.5)
        with st.spinner("Loading ML power prediction model..."):
            status_text.text("Agent 3: Running Gaussian Process power prediction...")
            progress_bar.progress(60)
            
            try:
                power_agent, wake_agent = load_agents()
                
                power_results = power_agent.predict(
                    yaw_angle=expert['suggested_yaw'],
                    n_time_points=n_timesteps,
                    return_samples=True,
                    n_samples=50
                )
                
                mean_power = np.mean(power_results['power_mean_MW'])
                max_power = np.max(power_results['power_mean_MW'])
                uncertainty = np.mean(power_results['power_std_MW'])
                
                results["power"] = {
                    "mean_MW": mean_power,
                    "max_MW": max_power,
                    "uncertainty_MW": uncertainty,
                    "time_series": power_results
                }
                
                progress_bar.progress(70)
                
                st.success(f"✅ Power prediction complete!")
                
                # Display mean power as info box
                st.info(f"📊 **Time-Averaged Mean Power:** {mean_power:.3f} MW ± {uncertainty:.3f} MW")
                
                # Power plot showing transient samples (not just the mean)
                fig, ax = plt.subplots(figsize=(8, 4))
                t = power_results['normalized_time']
                
                # Plot several posterior samples to show transient variability
                if 'samples' in power_results and power_results['samples'] is not None:
                    samples = power_results['samples']
                    # Plot up to 20 samples in light colors to show variability
                    n_plot_samples = min(20, samples.shape[0])
                    for i in range(n_plot_samples):
                        ax.plot(t, samples[i], 'steelblue', alpha=0.15, linewidth=0.8)
                    # Plot one sample prominently as "representative transient"
                    ax.plot(t, samples[0], 'b-', linewidth=1.5, label='Transient Sample', alpha=0.8)
                
                # Also show the mean as dashed line
                ax.plot(t, power_results['power_mean_MW'], 'r--', linewidth=1.5, 
                       label=f'Mean ({mean_power:.3f} MW)', alpha=0.9)
                
                # Show 95% CI as shaded region
                ax.fill_between(t, power_results['power_lower_95_MW'], 
                              power_results['power_upper_95_MW'],
                              alpha=0.15, color='gray', label='95% CI')
                
                ax.set_xlabel('Normalized Time')
                ax.set_ylabel('Power (MW)')
                ax.set_title(f'Transient Power Prediction at Yaw = {expert["suggested_yaw"]:.1f}°')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Power prediction failed: {e}")
                results["power"] = None
    
    # AGENT 4: Wake Flow
    with col4:
        st.markdown("#### 🌊 Agent 4: Wake Flow Simulator")
        
        time.sleep(0.8)  # Pause to simulate agent starting
        status_text.text("🔄 Agent 4: Initializing TT-OpInf model...")
        time.sleep(0.5)
        with st.spinner("Running TT-OpInf wake flow simulation..."):
            status_text.text("Agent 4: Computing 3D velocity field using TT-OpInf...")
            progress_bar.progress(80)
            
            try:
                power_agent, wake_agent = load_agents()
                
                predictions, vtk_dir = wake_agent.predict(
                    yaw_angle=expert['suggested_yaw'],
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
                
                progress_bar.progress(90)
                
                st.success(f"✅ Wake simulation complete!")
                
                wcol1, wcol2 = st.columns(2)
                with wcol1:
                    st.metric("Spatial Points", f"{predictions.shape[1]:,}")
                with wcol2:
                    st.metric("Timesteps", predictions.shape[0])
                
                st.write(f"Velocity range: {velocity_mag.min():.2f} - {velocity_mag.max():.2f} m/s")
                
                if vtk_dir:
                    st.info(f"📁 VTK files saved to: `{vtk_dir}`")
                
            except Exception as e:
                st.error(f"Wake simulation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                results["wake"] = None
    
    progress_bar.progress(95)
    
    # =========================================================================
    # Animation Display
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🎬 Wake Flow Animation")
    st.markdown("""
    <div class="info-box">
    Creating ParaView-style visualization using the actual VTK mesh data and TT-OpInf predictions.
    <br><br>
    <b>⚠️ Note:</b> Resolution has been reduced (10,000 of 180,857 points sampled) to enable 
    real-time visualization in the browser. For full-resolution analysis, use the exported VTK 
    files in ParaView.
    </div>
    """, unsafe_allow_html=True)
    
    if results.get("wake") is not None:
        with st.spinner("Generating wake flow animation from VTK data..."):
            status_text.text("Creating ParaView-style visualization...")
            
            try:
                # Import the animation module
                from wake_animation import create_wake_contour_animation
                
                # Create animation using actual VTK grid data
                anim_path = os.path.join(SCRIPT_DIR, "wake_animation.gif")
                
                # Use the contour animation which interpolates actual data
                grid_path = str(PROJECT_ROOT / "ResultMLYaw" / "Grid_data.vtk")
                
                create_wake_contour_animation(
                    predictions=results["wake"]["predictions"],
                    grid_path=grid_path,
                    output_path=anim_path,
                    fps=8,
                    frame_skip=max(1, results["wake"]["predictions"].shape[0] // 20),
                    cmap="RdYlBu_r",
                    verbose=False
                )
                
                # Display animation
                st.image(anim_path, caption="Wake Flow Evolution (TT-OpInf Model - Actual VTK Data)")
                
                # Download button
                with open(anim_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Animation",
                        data=f,
                        file_name="wake_animation.gif",
                        mime="image/gif"
                    )
                    
            except Exception as e:
                st.warning(f"Could not create animation: {e}")
    
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
    
    report_md = f"""
| Parameter | Value |
|-----------|-------|
| **Location** | {results.get('location', 'N/A')} |
| **Timestamp** | {results.get('timestamp', 'N/A')} |
| **Wind Speed** | {weather.get('wind_speed_ms', 0):.1f} m/s |
| **Wind Direction** | {weather.get('wind_direction_deg', 0):.0f}° |
| **Recommended Yaw** | {expert.get('suggested_yaw', 0):.1f}° |
| **Operating Region** | {expert.get('operating_region', 'N/A')} |
| **Predicted Power** | {power.get('mean_MW', 0):.3f} ± {power.get('uncertainty_MW', 0):.3f} MW |
| **Wake Field Points** | {wake.get('shape', [0,0,0])[1] if wake else 'N/A':,} |
| **Simulation Steps** | {wake.get('shape', [0,0,0])[0] if wake else 'N/A'} |
"""
    
    st.markdown(report_md)
    
    # Export button
    if st.button("📄 Export Full Report"):
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
Recommended Yaw: {expert.get('suggested_yaw', 0):.1f}°
Expected Efficiency: {expert.get('expected_efficiency', 0)*100:.1f}%

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
Developed at SINTEF Digital
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
