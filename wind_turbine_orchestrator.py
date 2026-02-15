"""
Wind Turbine Multi-Agent Orchestrator

This orchestrator coordinates multiple specialized agents to provide comprehensive
wind turbine operation recommendations:

1. Weather Agent: Fetches local wind conditions (speed, direction)
2. NREL 5MW Expert Agent: LLM-based agent suggesting optimal yaw angle
3. Rotor Power Agent: Predicts power output for suggested yaw
4. Wake Flow Agent: Predicts wake flow dynamics

Usage:
    from wind_turbine_orchestrator import WindTurbineOrchestrator
    
    orchestrator = WindTurbineOrchestrator()
    results = orchestrator.run_analysis(location="Boulder, Colorado")
    orchestrator.generate_report(results)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Add the parent directory to path for TT-OpInf imports
sys.path.insert(0, r"C:\Users\mandart\A_MANDAR_DOCUMENTS\HAM_Wind_Energy")

# Import local rotor power agent
from rotor_power_agent import RotorPowerAgent

# Import reviewer agent
from reviewer_agent import WindTurbineReviewerAgent

# Try to import TT-OpInf based Wake Flow Agent
try:
    from tt_opinf_inference_agent import WakeFlowAgent
    WAKE_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TT-OpInf WakeFlowAgent not found: {e}. Wake flow predictions will be skipped.")
    WAKE_AGENT_AVAILABLE = False

# Try to import requests for weather API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: 'requests' library not installed. Using simulated weather data.")
    REQUESTS_AVAILABLE = False


# =============================================================================
# AGENT 1: Weather Agent
# =============================================================================
class WeatherAgent:
    """
    Agent that fetches local weather conditions from meteorological services.
    
    Uses Open-Meteo API (free, no API key required) to get:
    - Wind speed (m/s)
    - Wind direction (degrees)
    - Temperature
    - Weather conditions
    """
    
    def __init__(self):
        self.api_base = "https://api.open-meteo.com/v1/forecast"
        self.geocoding_api = "https://geocoding-api.open-meteo.com/v1/search"
        print("Weather Agent initialized")
    
    def get_coordinates(self, location: str) -> tuple:
        """Get latitude and longitude for a location name."""
        if not REQUESTS_AVAILABLE:
            # Default to Boulder, Colorado coordinates
            return (40.0150, -105.2705)
        
        try:
            response = requests.get(
                self.geocoding_api,
                params={"name": location, "count": 1},
                timeout=10
            )
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                return (result["latitude"], result["longitude"])
        except Exception as e:
            print(f"Geocoding failed: {e}. Using default coordinates.")
        
        # Default coordinates (Boulder, Colorado - near NREL)
        return (40.0150, -105.2705)
    
    def fetch_weather(self, location: str = "Boulder, Colorado") -> Dict[str, Any]:
        """
        Fetch current weather conditions for a location.
        
        Args:
            location: Location name (city, state/country)
            
        Returns:
            Dict with wind_speed_ms, wind_direction_deg, temperature_c, conditions
        """
        print(f"\n{'='*60}")
        print(f"AGENT 1: Weather Agent - Fetching conditions for {location}")
        print(f"{'='*60}")
        
        lat, lon = self.get_coordinates(location)
        
        if REQUESTS_AVAILABLE:
            try:
                response = requests.get(
                    self.api_base,
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
                
                weather_result = {
                    "location": location,
                    "latitude": lat,
                    "longitude": lon,
                    "timestamp": datetime.now().isoformat(),
                    "wind_speed_ms": current.get("wind_speed_10m", 8.5),
                    "wind_direction_deg": current.get("wind_direction_10m", 275),
                    "temperature_c": current.get("temperature_2m", 15),
                    "weather_code": current.get("weather_code", 0),
                    "data_source": "Open-Meteo API"
                }
                
            except Exception as e:
                print(f"API request failed: {e}. Using simulated data.")
                weather_result = self._simulate_weather(location, lat, lon)
        else:
            weather_result = self._simulate_weather(location, lat, lon)
        
        # Report findings
        print(f"\n📍 Location: {weather_result['location']}")
        print(f"🌡️  Temperature: {weather_result['temperature_c']:.1f}°C")
        print(f"💨 Wind Speed: {weather_result['wind_speed_ms']:.1f} m/s")
        print(f"🧭 Wind Direction: {weather_result['wind_direction_deg']:.0f}°")
        print(f"📊 Data Source: {weather_result['data_source']}")
        
        return weather_result
    
    def _simulate_weather(self, location: str, lat: float, lon: float) -> Dict[str, Any]:
        """Generate simulated weather data for testing."""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        return {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "timestamp": datetime.now().isoformat(),
            "wind_speed_ms": np.random.uniform(6, 12),
            "wind_direction_deg": np.random.uniform(265, 290),
            "temperature_c": np.random.uniform(10, 25),
            "weather_code": 0,
            "data_source": "Simulated Data"
        }


# =============================================================================
# AGENT 2: NREL 5MW Expert Agent
# =============================================================================
class NREL5MWExpertAgent:
    """
    Expert LLM-based agent for the NREL 5MW Reference Wind Turbine.
    
    This agent uses knowledge of the NREL 5MW turbine specifications and
    wind turbine aerodynamics to suggest optimal yaw angles based on
    wind conditions.
    
    NREL 5MW Reference Turbine Specifications:
    - Rated Power: 5 MW
    - Rotor Diameter: 126 m
    - Hub Height: 90 m
    - Cut-in Wind Speed: 3 m/s
    - Rated Wind Speed: 11.4 m/s
    - Cut-out Wind Speed: 25 m/s
    - Optimal Tip Speed Ratio: 7.55
    """
    
    def __init__(self):
        self.turbine_specs = {
            "name": "NREL 5MW Reference Wind Turbine",
            "rated_power_mw": 5.0,
            "rotor_diameter_m": 126,
            "hub_height_m": 90,
            "cut_in_wind_speed_ms": 3.0,
            "rated_wind_speed_ms": 11.4,
            "cut_out_wind_speed_ms": 25.0,
            "optimal_tip_speed_ratio": 7.55,
            "yaw_rate_deg_per_s": 0.3
        }
        
        # Yaw angle constraints for this analysis
        self.yaw_min = 272
        self.yaw_max = 285
        
        print("NREL 5MW Expert Agent initialized")
        print(f"  Yaw angle range: {self.yaw_min}° - {self.yaw_max}°")
    
    def analyze_and_suggest_yaw(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze weather conditions and suggest optimal yaw angle.
        
        The expert reasoning considers:
        1. Wind direction alignment
        2. Wind speed (below/above rated)
        3. Yaw misalignment effects on power (cos³ law approximation)
        4. Structural loads at high yaw misalignment
        
        Args:
            weather_data: Output from WeatherAgent.fetch_weather()
            
        Returns:
            Dict with suggested_yaw, reasoning, expected_efficiency, etc.
        """
        print(f"\n{'='*60}")
        print(f"AGENT 2: NREL 5MW Expert Agent - Analyzing conditions")
        print(f"{'='*60}")
        
        wind_speed = weather_data["wind_speed_ms"]
        wind_direction = weather_data["wind_direction_deg"]
        
        # Expert analysis
        analysis = self._perform_expert_analysis(wind_speed, wind_direction)
        
        # Print reasoning
        print(f"\n🔬 Expert Analysis:")
        print(f"   Wind Direction: {wind_direction:.0f}°")
        print(f"   Wind Speed: {wind_speed:.1f} m/s")
        print(f"   Operating Region: {analysis['operating_region']}")
        print(f"\n💡 Reasoning:")
        for line in analysis['reasoning']:
            print(f"   • {line}")
        print(f"\n✅ Suggested Yaw Angle: {analysis['suggested_yaw']:.1f}°")
        print(f"   Expected Power Efficiency: {analysis['expected_efficiency']*100:.1f}%")
        
        return analysis
    
    def _perform_expert_analysis(self, wind_speed: float, wind_direction: float) -> Dict[str, Any]:
        """
        Perform expert-level analysis to determine optimal yaw.
        
        Expert knowledge applied:
        - Yaw misalignment reduces power by approximately cos³(misalignment)
        - Small intentional misalignment can reduce wake effects on downstream turbines
        - At high wind speeds, yaw misalignment increases structural loads
        - NREL 5MW has specific performance characteristics
        """
        reasoning = []
        
        # Determine operating region
        if wind_speed < self.turbine_specs["cut_in_wind_speed_ms"]:
            operating_region = "Below Cut-in"
            reasoning.append(f"Wind speed ({wind_speed:.1f} m/s) is below cut-in speed ({self.turbine_specs['cut_in_wind_speed_ms']} m/s)")
            suggested_yaw = (self.yaw_min + self.yaw_max) / 2  # Park at neutral
        elif wind_speed > self.turbine_specs["cut_out_wind_speed_ms"]:
            operating_region = "Above Cut-out"
            reasoning.append(f"Wind speed ({wind_speed:.1f} m/s) exceeds cut-out speed ({self.turbine_specs['cut_out_wind_speed_ms']} m/s)")
            suggested_yaw = (self.yaw_min + self.yaw_max) / 2
        elif wind_speed <= self.turbine_specs["rated_wind_speed_ms"]:
            operating_region = "Partial Load (Region 2)"
            reasoning.append(f"Operating in partial load region - maximize energy capture")
            reasoning.append("Aligning yaw closely with wind direction is critical for power optimization")
            
            # Ideal yaw would match wind direction
            # But we're constrained to 272-285 range
            ideal_yaw = wind_direction
            suggested_yaw = np.clip(ideal_yaw, self.yaw_min, self.yaw_max)
            
            if ideal_yaw < self.yaw_min:
                reasoning.append(f"Wind direction ({wind_direction:.0f}°) is below yaw range - using minimum yaw {self.yaw_min}°")
            elif ideal_yaw > self.yaw_max:
                reasoning.append(f"Wind direction ({wind_direction:.0f}°) is above yaw range - using maximum yaw {self.yaw_max}°")
            else:
                reasoning.append(f"Wind direction within yaw range - aligning with wind at {suggested_yaw:.0f}°")
        else:
            operating_region = "Full Load (Region 3)"
            reasoning.append(f"Operating in full load region - power is limited to rated")
            reasoning.append("Small yaw misalignment acceptable; consider wake steering")
            
            # At rated power, slight misalignment for wake steering is beneficial
            # Target ~5° misalignment for wake deflection
            ideal_yaw = wind_direction + 3  # Small offset for wake steering
            suggested_yaw = np.clip(ideal_yaw, self.yaw_min, self.yaw_max)
            reasoning.append(f"Applying 3° wake steering offset for downstream turbine benefit")
        
        # Calculate expected efficiency based on yaw misalignment
        yaw_misalignment = abs(suggested_yaw - wind_direction)
        # Power loss approximation: cos³(misalignment)
        expected_efficiency = np.cos(np.radians(yaw_misalignment)) ** 3
        
        reasoning.append(f"Yaw misalignment: {yaw_misalignment:.1f}° → Expected efficiency: {expected_efficiency*100:.1f}%")
        
        return {
            "suggested_yaw": suggested_yaw,
            "wind_direction": wind_direction,
            "wind_speed": wind_speed,
            "yaw_misalignment": yaw_misalignment,
            "expected_efficiency": expected_efficiency,
            "operating_region": operating_region,
            "reasoning": reasoning,
            "turbine_specs": self.turbine_specs
        }


# =============================================================================
# AGENT 3: Rotor Power Agent (wrapper for existing agent)
# =============================================================================
class RotorPowerAgentWrapper:
    """
    Wrapper for the RotorPowerAgent to integrate with the orchestrator.
    """
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "rotor_power_gp_model.joblib")
        
        print("\nInitializing Rotor Power Agent...")
        self.agent = RotorPowerAgent(model_path)
    
    def predict_power(self, yaw_angle: float, n_time_points: int = 100) -> Dict[str, Any]:
        """
        Predict power output for a given yaw angle.
        
        Args:
            yaw_angle: Suggested yaw angle from Expert Agent
            n_time_points: Number of time points for trajectory
            
        Returns:
            Dict with power predictions and uncertainty
        """
        print(f"\n{'='*60}")
        print(f"AGENT 3: Rotor Power Prediction Agent")
        print(f"{'='*60}")
        print(f"Predicting power dynamics for yaw angle = {yaw_angle:.1f}°")
        
        results = self.agent.predict(
            yaw_angle=yaw_angle, 
            n_time_points=n_time_points,
            return_samples=True,
            n_samples=50
        )
        
        # Calculate summary statistics
        mean_power = np.mean(results['power_mean_MW'])
        max_power = np.max(results['power_mean_MW'])
        min_power = np.min(results['power_mean_MW'])
        mean_uncertainty = np.mean(results['power_std_MW'])
        
        print(f"\n⚡ Power Prediction Results:")
        print(f"   Mean Power: {mean_power:.3f} MW")
        print(f"   Power Range: {min_power:.3f} - {max_power:.3f} MW")
        print(f"   Mean Uncertainty (±1σ): ±{mean_uncertainty:.3f} MW")
        
        results['summary'] = {
            'mean_power_MW': mean_power,
            'max_power_MW': max_power,
            'min_power_MW': min_power,
            'mean_uncertainty_MW': mean_uncertainty
        }
        
        return results
    
    def plot_prediction(self, results: Dict, save_path: str = None):
        """Plot the power prediction."""
        self.agent.plot_prediction(results, show_samples=True, save_path=save_path)


# =============================================================================
# AGENT 4: Wake Flow Agent (TT-OpInf based)
# =============================================================================
class WakeFlowAgentWrapper:
    """
    Wrapper for the TT-OpInf based WakeFlowAgent.
    
    Uses pre-trained TT-OpInf model to predict wind turbine wake flow dynamics.
    Model and grid data are loaded from the parent ResultMLYaw folder.
    
    The agent returns:
        - predictions: Shape (n_time_steps+1, n_points, 3) - velocity field
        - velocity_magnitude: Velocity magnitude at each point
        - output_files: Dict with paths to generated files (VTK)
    """
    
    def __init__(self):
        self.agent = None
        self.available = False
        
        if WAKE_AGENT_AVAILABLE:
            try:
                print("\nInitializing TT-OpInf Wake Flow Agent...")
                # Paths are set inside WakeFlowAgent to default to parent folder
                # tt_opinf_model and Grid_data.vtk
                self.agent = WakeFlowAgent()
                self.available = True
                print("TT-OpInf Wake Flow Agent initialized successfully")
            except Exception as e:
                print(f"Failed to initialize TT-OpInf Wake Flow Agent: {e}")
                import traceback
                traceback.print_exc()
                self.available = False
        else:
            print("\nTT-OpInf Wake Flow Agent not available - skipping wake predictions")
    
    def predict_wake(self, yaw_angle: float, n_time_steps: int = 100,
                     create_animation: bool = True, export_vtk: bool = True) -> Optional[Dict[str, Any]]:
        """
        Predict wake flow dynamics for a given yaw angle using TT-OpInf model.
        
        Args:
            yaw_angle: Yaw angle for wake prediction (from Agent 2)
            n_time_steps: Number of time steps for prediction
            create_animation: Not used (kept for API compatibility)
            export_vtk: Whether to export VTK files for visualization
            
        Returns:
            Dict with wake flow predictions or None if agent unavailable:
                - predictions: Shape (n_time_steps+1, n_points, 3) - velocity field
                - velocity_magnitude: Velocity magnitude at each point  
                - output_files: Dict with paths to generated files (vtk_dir)
        """
        if not self.available:
            print(f"\n{'='*60}")
            print(f"AGENT 4: TT-OpInf Wake Flow Agent - SKIPPED (not available)")
            print(f"{'='*60}")
            return None
        
        print(f"\n{'='*60}")
        print(f"AGENT 4: Wake Flow Prediction Agent (TT-OpInf Model)")
        print(f"{'='*60}")
        print(f"Predicting wake dynamics for yaw angle = {yaw_angle:.1f}°")
        print(f"Time steps: {n_time_steps}")
        print(f"Export VTK: {export_vtk}")
        
        try:
            # Call the TT-OpInf agent's predict method
            predictions, vtk_dir = self.agent.predict(
                yaw_angle=yaw_angle,
                n_timesteps=n_time_steps,
                export_vtk=export_vtk,
                export_every=5,  # Export every 5th timestep for efficiency
                verbose=True
            )
            
            # Calculate velocity magnitude
            velocity_magnitude = np.linalg.norm(predictions, axis=2)
            
            # Build result dict compatible with orchestrator
            result = {
                'predictions': predictions,
                'velocity_magnitude': velocity_magnitude,
                'output_files': {}
            }
            
            if vtk_dir:
                result['output_files']['vtk'] = vtk_dir
            
            # Display summary
            print(f"\n🌊 Wake Flow Prediction Complete!")
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   (time_steps, spatial_points, velocity_components)")
            print(f"   Velocity magnitude range: {np.min(velocity_magnitude):.2f} - {np.max(velocity_magnitude):.2f} m/s")
            
            if vtk_dir:
                print(f"\n📁 Output Files Generated:")
                print(f"   VTK files: {vtk_dir}")
            
            return result
            
        except Exception as e:
            print(f"Wake prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_wake_statistics(self, result: Dict) -> Dict[str, Any]:
        """
        Extract summary statistics from wake prediction results.
        
        Args:
            result: Output from predict_wake()
            
        Returns:
            Dict with wake flow statistics
        """
        if result is None:
            return {}
        
        predictions = result.get('predictions')
        velocity_mag = result.get('velocity_magnitude')
        
        stats = {}
        if velocity_mag is not None:
            stats['mean_velocity_ms'] = float(np.mean(velocity_mag))
            stats['max_velocity_ms'] = float(np.max(velocity_mag))
            stats['min_velocity_ms'] = float(np.min(velocity_mag))
            stats['std_velocity_ms'] = float(np.std(velocity_mag))
        
        if predictions is not None:
            stats['n_time_steps'] = predictions.shape[0]
            stats['n_spatial_points'] = predictions.shape[1]
            stats['n_components'] = predictions.shape[2]
        
        return stats
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the loaded TT-OpInf model.
        
        Returns:
            Dict with model information or None if not available
        """
        if self.available and self.agent:
            return self.agent.get_info()
        return None


# =============================================================================
# ORCHESTRATOR
# =============================================================================
class WindTurbineOrchestrator:
    """
    Main orchestrator that coordinates all agents for wind turbine analysis.
    
    Workflow:
    1. Weather Agent fetches current conditions
    2. NREL 5MW Expert suggests optimal yaw angle
    3. Rotor Power Agent predicts power output
    4. Wake Flow Agent predicts wake dynamics
    5. Generate comprehensive report
    """
    
    def __init__(self, rotor_power_model_path: str = None, config: Dict[str, Any] = None, 
                 reviewer_mode: str = "advisory", reviewer_enabled: bool = True):
        print("\n" + "="*70)
        print("  WIND TURBINE MULTI-AGENT ORCHESTRATOR")
        print("="*70)
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        # Initialize all agents
        self.weather_agent = WeatherAgent()
        self.expert_agent = NREL5MWExpertAgent()
        self.power_agent = RotorPowerAgentWrapper(rotor_power_model_path)
        self.wake_agent = WakeFlowAgentWrapper()
        
        # Initialize reviewer agent
        self.config = config or {}
        self.reviewer_agent = WindTurbineReviewerAgent(
            config=self.config,
            mode=reviewer_mode,
            enabled=reviewer_enabled
        )
        
        print("\n" + "-"*70)
        print("All agents initialized. Ready for analysis.")
        if reviewer_enabled:
            print(f"Expert Reviewer: ENABLED (mode={reviewer_mode})")
        else:
            print("Expert Reviewer: DISABLED")
        print("-"*70)
    
    async def run_analysis(self, location: str = "Boulder, Colorado", 
                     n_time_points: int = 100,
                     create_wake_animation: bool = True,
                     export_wake_vtk: bool = False,
                     agent2b_result: Optional[Dict[str, Any]] = None,
                     turbine_pairs: Optional[Dict[str, Any]] = None,
                     optimization_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run full multi-agent analysis.
        
        Args:
            location: Location for weather data
            n_time_points: Number of time points for predictions
            create_wake_animation: Whether to generate wake animation
            export_wake_vtk: Whether to export VTK files for wake visualization
            agent2b_result: Optional Agent 2B LLM expert result
            turbine_pairs: Optional Agent 2C/2D turbine pair result
            optimization_result: Optional optimization results from Agent 3
            
        Returns:
            Dict with results from all agents and expert reviews
        """
        print("\n" + "🚀"*35)
        print("  STARTING MULTI-AGENT ANALYSIS")
        print("🚀"*35 + "\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "reviews": {}  # Store all expert reviews
        }
        
        # AGENT 1: Get weather conditions
        weather_data = self.weather_agent.fetch_weather(location)
        results["weather"] = weather_data
        
        # AGENT 2: Get expert yaw recommendation
        expert_analysis = self.expert_agent.analyze_and_suggest_yaw(weather_data)
        results["expert_analysis"] = expert_analysis
        suggested_yaw = expert_analysis["suggested_yaw"]
        
        # CHECKPOINT 1: Review Agent 2 outputs
        review1 = await self.reviewer_agent.review_agent2(
            weather_data=weather_data,
            expert_analysis=expert_analysis,
            agent2b_result=agent2b_result,
            turbine_pairs=turbine_pairs
        )
        results["reviews"]["checkpoint1_agent2"] = review1
        
        # Check if workflow should continue (blocking mode with critical issues)
        if not review1.get("allow_continue", True):
            print("\n" + "🛑"*35)
            print("  WORKFLOW HALTED: Critical issues detected in Agent 2")
            print("🛑"*35 + "\n")
            results["status"] = "halted_at_checkpoint1"
            return results
        
        # AGENT 3: Predict power output
        power_prediction = self.power_agent.predict_power(suggested_yaw, n_time_points)
        results["power_prediction"] = power_prediction
        
        # CHECKPOINT 2: Review Agent 3 outputs
        review2 = await self.reviewer_agent.review_agent3(
            weather_data=weather_data,
            expert_analysis=expert_analysis,
            power_prediction=power_prediction,
            optimization_result=optimization_result
        )
        results["reviews"]["checkpoint2_agent3"] = review2
        
        # Check if workflow should continue
        if not review2.get("allow_continue", True):
            print("\n" + "🛑"*35)
            print("  WORKFLOW HALTED: Critical issues detected in Agent 3")
            print("🛑"*35 + "\n")
            results["status"] = "halted_at_checkpoint2"
            return results
        
        # AGENT 4: Predict wake flow (agent is callable, animation/vtk handled internally)
        wake_prediction = self.wake_agent.predict_wake(
            yaw_angle=suggested_yaw, 
            n_time_steps=n_time_points,
            create_animation=create_wake_animation,
            export_vtk=export_wake_vtk
        )
        results["wake_prediction"] = wake_prediction
        
        # Get wake statistics if available
        if wake_prediction is not None:
            results["wake_statistics"] = self.wake_agent.get_wake_statistics(wake_prediction)
        
        # CHECKPOINT 3: Review Agent 4 outputs
        review3 = await self.reviewer_agent.review_agent4(
            weather_data=weather_data,
            expert_analysis=expert_analysis,
            wake_prediction=wake_prediction or {}
        )
        results["reviews"]["checkpoint3_agent4"] = review3
        
        # Check if workflow should continue
        if not review3.get("allow_continue", True):
            print("\n" + "🛑"*35)
            print("  WORKFLOW HALTED: Critical issues detected in Agent 4")
            print("🛑"*35 + "\n")
            results["status"] = "halted_at_checkpoint3"
            return results
        
        # Generate final comprehensive review
        final_review = await self.reviewer_agent.generate_final_review(results)
        results["reviews"]["final_review"] = final_review
        
        print("\n" + "✅"*35)
        print("  ANALYSIS COMPLETE")
        if final_review.get("overall_status") == "FAILED":
            print("  ⚠️ WARNING: Analysis contains critical issues")
        elif final_review.get("overall_status") == "WARNING":
            print("  ⚠️ Analysis completed with warnings")
        else:
            print("  ✅ All validation checks passed")
        print("✅"*35)
        
        results["status"] = "completed"
        return results
    
    def generate_report(self, results: Dict[str, Any], save_path: str = None) -> str:
        """
        Generate a comprehensive report from all agent results.
        
        Args:
            results: Output from run_analysis()
            save_path: Path to save report (optional)
            
        Returns:
            Report string
        """
        weather = results.get("weather", {})
        expert = results.get("expert_analysis", {})
        power = results.get("power_prediction", {})
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            WIND TURBINE MULTI-AGENT ANALYSIS REPORT                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Generated: {results.get('timestamp', 'N/A'):<54} ║
║  Location: {results.get('location', 'N/A'):<55} ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. WEATHER CONDITIONS (Agent 1)                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  Wind Speed:       {weather.get('wind_speed_ms', 0):<8.1f} m/s                                     │
│  Wind Direction:   {weather.get('wind_direction_deg', 0):<8.0f} degrees                                 │
│  Temperature:      {weather.get('temperature_c', 0):<8.1f} °C                                      │
│  Data Source:      {weather.get('data_source', 'N/A'):<40}   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. NREL 5MW EXPERT ANALYSIS (Agent 2)                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│  Operating Region: {expert.get('operating_region', 'N/A'):<40}   │
│  Suggested Yaw:    {expert.get('suggested_yaw', 0):<8.1f} degrees                                 │
│  Yaw Misalignment: {expert.get('yaw_misalignment', 0):<8.1f} degrees                                 │
│  Expected Eff.:    {expert.get('expected_efficiency', 0)*100:<8.1f} %                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  Expert Reasoning:                                                           │
"""
        for reason in expert.get('reasoning', []):
            report += f"│    • {reason:<72} │\n"
        
        report += f"""└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. POWER PREDICTION (Agent 3)                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  Predicted Yaw:    {power.get('yaw_angle', 0):<8.1f} degrees                                 │
│  Mean Power:       {power.get('summary', {}).get('mean_power_MW', 0):<8.3f} MW                                      │
│  Power Range:      {power.get('summary', {}).get('min_power_MW', 0):.3f} - {power.get('summary', {}).get('max_power_MW', 0):.3f} MW                               │
│  Uncertainty (1σ): ±{power.get('summary', {}).get('mean_uncertainty_MW', 0):<7.3f} MW                                      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. WAKE FLOW PREDICTION (Agent 4 - TT-OpInf Model)                           │
├──────────────────────────────────────────────────────────────────────────────┤
"""
        wake_stats = results.get('wake_statistics', {})
        wake_pred = results.get('wake_prediction')
        
        if wake_pred is not None and wake_stats:
            report += f"""│  Status:           Available                                                 │
│  Time Steps:       {wake_stats.get('n_time_steps', 0):<8}                                          │
│  Spatial Points:   {wake_stats.get('n_spatial_points', 0):<8}                                          │
│  Mean Velocity:    {wake_stats.get('mean_velocity_ms', 0):<8.2f} m/s                                   │
│  Velocity Range:   {wake_stats.get('min_velocity_ms', 0):.2f} - {wake_stats.get('max_velocity_ms', 0):.2f} m/s                               │
"""
            output_files = wake_pred.get('output_files', {})
            if output_files:
                if 'animation' in output_files:
                    report += f"""│  Animation:        {str(output_files.get('animation', 'N/A')):<40}   │
"""
        else:
            report += f"""│  Status:           Not Available                                             │
"""
        
        report += f"""└──────────────────────────────────────────────────────────────────────────────┘
"""
        
        # Add Expert Review section if available
        final_review = results.get("reviews", {}).get("final_review", {})
        if final_review and final_review.get("overall_status"):
            status = final_review.get("overall_status", "UNKNOWN")
            status_message = final_review.get('status_message', 'N/A')[:56]  # Truncate if needed
            
            report += f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. EXPERT REVIEW ASSESSMENT                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  Overall Status:   {status:<56}  │
│  Status Message:   {status_message:<56}  │
│                                                                              │
│  Critical Issues:  {final_review.get('critical_count', 0):<56}  │
│  Warnings:         {final_review.get('warning_count', 0):<56}  │
│  Info Messages:    {final_review.get('info_count', 0):<56}  │
│  Checkpoints:      {final_review.get('checkpoint_count', 0):<56}  │
"""
            
            # Add key findings if any
            key_findings = final_review.get("key_findings", [])
            if key_findings:
                report += "├──────────────────────────────────────────────────────────────────────────────┤\n"
                report += "│  Key Findings:                                                               │\n"
                for finding in key_findings[:5]:  # Limit to 5 findings
                    finding_type = finding.get("type", "info")
                    msg = finding.get("message", "")[:70]  # Truncate long messages
                    prefix = "[CRITICAL]" if finding_type == "critical" else "[WARNING]" if finding_type == "warning" else "[INFO]"
                    line_text = f"{prefix} {msg}"[:74]  # Total width 74
                    report += f"│  {line_text:<76}│\n"
            
            # Add recommendations
            recommendations = final_review.get("recommendations", [])
            if recommendations:
                report += "├──────────────────────────────────────────────────────────────────────────────┤\n"
                report += "│  Recommendations:                                                            │\n"
                for rec in recommendations[:3]:  # Limit to 3 recommendations
                    # Strip emojis and truncate
                    clean_rec = rec.replace("✅", "").replace("⚠️", "").replace("🔴", "").strip()[:74]
                    report += f"│  {clean_rec:<76}│\n"
            
            report += "└──────────────────────────────────────────────────────────────────────────────┘\n"
        
        report += f"""
═══════════════════════════════════════════════════════════════════════════════
                              END OF REPORT
═══════════════════════════════════════════════════════════════════════════════
"""
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {save_path}")
        
        return report
    
    def visualize_results(self, results: Dict[str, Any], save_dir: str = None):
        """
        Create visualization of all results.
        
        Args:
            results: Output from run_analysis()
            save_dir: Directory to save figures
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Weather conditions summary
        ax1 = axes[0, 0]
        weather = results.get("weather", {})
        categories = ['Wind Speed\n(m/s)', 'Wind Dir\n(°/10)', 'Temperature\n(°C)']
        values = [
            weather.get('wind_speed_ms', 0),
            weather.get('wind_direction_deg', 0) / 10,  # Scale for visualization
            weather.get('temperature_c', 0)
        ]
        bars = ax1.bar(categories, values, color=['steelblue', 'coral', 'green'])
        ax1.set_ylabel('Value')
        ax1.set_title('Weather Conditions\n(Wind Direction scaled by 1/10)')
        for bar, val in zip(bars, [weather.get('wind_speed_ms', 0), 
                                    weather.get('wind_direction_deg', 0),
                                    weather.get('temperature_c', 0)]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=10)
        
        # Plot 2: Yaw analysis
        ax2 = axes[0, 1]
        expert = results.get("expert_analysis", {})
        wind_dir = expert.get('wind_direction', 275)
        suggested_yaw = expert.get('suggested_yaw', 278)
        
        # Polar plot for yaw visualization
        ax2.remove()
        ax2 = fig.add_subplot(2, 2, 2, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        
        # Plot wind direction
        ax2.annotate('', xy=(np.radians(wind_dir), 1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=3))
        # Plot suggested yaw
        ax2.annotate('', xy=(np.radians(suggested_yaw), 0.8), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
        
        ax2.set_ylim(0, 1.2)
        ax2.set_title(f'Wind Direction (blue): {wind_dir:.0f}°\nSuggested Yaw (red): {suggested_yaw:.0f}°')
        
        # Plot 3: Power prediction with uncertainty
        ax3 = axes[1, 0]
        power = results.get("power_prediction", {})
        if 'normalized_time' in power:
            t = power['normalized_time']
            ax3.fill_between(t, power['power_lower_95_MW'], power['power_upper_95_MW'],
                           alpha=0.3, color='steelblue', label='95% CI')
            ax3.fill_between(t, 
                           power['power_mean_MW'] - power['power_std_MW'],
                           power['power_mean_MW'] + power['power_std_MW'],
                           alpha=0.4, color='steelblue', label='±1σ')
            ax3.plot(t, power['power_mean_MW'], 'b-', linewidth=2, label='Mean')
            ax3.set_xlabel('Normalized Time')
            ax3.set_ylabel('Power (MW)')
            ax3.set_title(f'Power Prediction at Yaw = {power.get("yaw_angle", 0):.1f}°')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        summary = power.get('summary', {})
        metrics = ['Mean\nPower', 'Max\nPower', 'Min\nPower', 'Mean\nUncertainty']
        values = [
            summary.get('mean_power_MW', 0),
            summary.get('max_power_MW', 0),
            summary.get('min_power_MW', 0),
            summary.get('mean_uncertainty_MW', 0)
        ]
        colors = ['steelblue', 'forestgreen', 'coral', 'gray']
        bars = ax4.bar(metrics, values, color=colors)
        ax4.set_ylabel('Power (MW)')
        ax4.set_title('Power Statistics Summary')
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'analysis_results.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Initialize the orchestrator
    orchestrator = WindTurbineOrchestrator()
    
    # Run the multi-agent analysis
    results = orchestrator.run_analysis(
        location="Boulder, Colorado",  # Near NREL
        n_time_points=100,
        create_wake_animation=True
    )
    
    # Generate and save report
    orchestrator.generate_report(results, save_path="analysis_report.txt")
    
    # Create visualizations
    orchestrator.visualize_results(results, save_dir=".")
    
    # Save results to JSON (excluding numpy arrays for serialization)
    results_json = {
        "timestamp": results["timestamp"],
        "location": results["location"],
        "weather": results["weather"],
        "expert_analysis": {
            k: v for k, v in results["expert_analysis"].items() 
            if not isinstance(v, np.ndarray)
        },
        "power_summary": results["power_prediction"].get("summary", {}),
        "wake_available": results["wake_prediction"] is not None
    }
    
    with open("analysis_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print("\nResults saved to: analysis_results.json")
