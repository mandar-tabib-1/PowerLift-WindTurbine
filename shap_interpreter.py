"""
SHAP Interpretability Module for Wind Turbine Multi-Agent Framework

Provides SHAP-based explanations for:
1. TT-OpInf Wake Flow ROM
2. Gaussian Process Power Predictor
3. Wake Steering Optimization decisions

Contact: mandar.tabib@sintef.no
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, fallback to simplified explanations if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("⚠️ SHAP not installed. Install with: `pip install shap` for full interpretability features.")

class WindTurbineInterpreter:
    """
    Interpretability agent for wind turbine ML models.
    
    Provides explanations for model predictions to build engineer trust.
    Falls back to physics-based explanations if SHAP unavailable.
    """
    
    def __init__(self, power_agent=None, wake_agent=None):
        self.power_agent = power_agent
        self.wake_agent = wake_agent
        self.background_data = None
        
    def prepare_background_data(self, n_samples: int = 50):
        """Create background dataset for SHAP explanations."""
        yaw_angles = np.linspace(270, 285, n_samples)
        self.background_data = yaw_angles.reshape(-1, 1)
        return self.background_data
        
    def explain_yaw_impact_simplified(self, yaw_misalignment: float) -> Dict:
        """
        Physics-based explanation when SHAP is unavailable.
        Args:
            yaw_misalignment: Yaw misalignment in degrees (0-15), where 0 = 275° yaw angle
        """
        # Convert misalignment to actual yaw angle
        # Formula: yaw_angle = 275 + (misalignment * 2/3)
        # At misalignment=0: yaw=275°, at misalignment=15: yaw=285°
        yaw_angle = 275.0 + (yaw_misalignment * 2.0 / 3.0)
        
        # Physics-based impact assessment
        # Calculate misalignment from baseline (270°)
        actual_misalignment = yaw_angle - 270.0
        power_efficiency = np.cos(np.radians(actual_misalignment)) ** 1.88  # Simplified power law
        wake_deflection = 0.3 * actual_misalignment  # Approximate wake deflection
        
        explanation = {
            'yaw_angle': yaw_angle,
            'yaw_misalignment': yaw_misalignment,
            'misalignment': actual_misalignment,
            'power_efficiency': power_efficiency,
            'power_loss_percent': (1 - power_efficiency) * 100,
            'wake_deflection': wake_deflection,
            'explanation_type': 'Physics-Based Analysis',
            'key_factors': {
                'Misalignment Effect': f"{(1-power_efficiency)*100:.1f}% power loss",
                'Wake Steering': f"{wake_deflection:.1f}° deflection expected",
                'Operational Impact': "Reduced power but potential wake steering benefits"
            }
        }
        
        return explanation
    
    def create_power_surrogate_simple(self):
        """Simplified power surrogate when GP model unavailable."""
        def power_func(yaw_angles):
            results = []
            for yaw in yaw_angles.flatten():
                misalignment = yaw - 270.0
                # NREL 5MW power curve approximation at 8.5 m/s
                base_power = 4.8  # MW at optimal alignment
                efficiency = np.cos(np.radians(misalignment)) ** 1.88
                power = base_power * efficiency
                results.append(power)
            return np.array(results)
        return power_func
    
    def create_wake_surrogate_simple(self):
        """Simplified wake surrogate when TT-OpInf unavailable."""
        def wake_func(yaw_angles):
            results = []
            for yaw in yaw_angles.flatten():
                misalignment = yaw - 270.0
                # Empirical wake deflection model
                deflection = 0.3 * misalignment
                # Wake deficit (0 = no deficit, 1 = complete blockage)
                base_deficit = 0.4  # 40% deficit at 7D downstream
                # Deflection reduces deficit for downstream turbines
                effective_deficit = base_deficit * np.exp(-0.02 * deflection**2)
                results.append(effective_deficit)
            return np.array(results)
        return wake_func
    
    def explain_interactive_yaw(self, yaw_misalignment: float) -> Dict:
        """
        Provide real-time explanation of yaw misalignment impact.
        Uses SHAP if available, otherwise physics-based explanations.
        Args:
            yaw_misalignment: Yaw misalignment in degrees (0-15), where 0 = 275° yaw angle
        """
        # Convert misalignment to actual yaw angle
        yaw_angle = 275.0 + (yaw_misalignment * 2.0 / 3.0)
        
        if not SHAP_AVAILABLE or self.power_agent is None:
            return self.explain_yaw_impact_simplified(yaw_misalignment)
        
        try:
            # SHAP-based explanation
            if self.background_data is None:
                self.prepare_background_data()
            
            power_surrogate = self.create_power_surrogate_simple()
            wake_surrogate = self.create_wake_surrogate_simple()
            
            # Try to use actual models if available
            if self.power_agent:
                def actual_power_surrogate(yaw_vals):
                    results = []
                    for yaw in yaw_vals.flatten():
                        try:
                            pred = self.power_agent.predict(yaw_angle=yaw, n_time_points=10)
                            results.append(np.mean(pred['power_mean_MW']))
                        except:
                            results.append(power_surrogate([yaw])[0])
                    return np.array(results)
                power_surrogate = actual_power_surrogate
            
            # Create SHAP explainers
            power_explainer = shap.Explainer(power_surrogate, self.background_data[:20])
            wake_explainer = shap.Explainer(wake_surrogate, self.background_data[:20])
            
            # Compute explanations
            target = np.array([[yaw_angle]])
            power_shap = power_explainer(target)
            wake_shap = wake_explainer(target)
            
            # Get predictions
            power_pred = power_surrogate(target)[0]
            wake_pred = wake_surrogate(target)[0]
            
            # Baseline values
            power_baseline = power_surrogate(np.array([[270.0]]))[0]
            wake_baseline = wake_surrogate(np.array([[270.0]]))[0]
            
            explanation = {
                'yaw_angle': yaw_angle,
                'yaw_misalignment': yaw_misalignment,
                'misalignment': yaw_angle - 270.0,
                'power_prediction': power_pred,
                'power_baseline': power_baseline,
                'power_impact': power_pred - power_baseline,
                'wake_deficit': wake_pred,
                'wake_baseline': wake_baseline,
                'wake_impact': wake_pred - wake_baseline,
                'power_shap_value': power_shap.values[0][0] if len(power_shap.values[0]) > 0 else 0,
                'wake_shap_value': wake_shap.values[0][0] if len(wake_shap.values[0]) > 0 else 0,
                'explanation_type': 'SHAP Analysis',
                'confidence': 'High' if abs(yaw_angle - 270) <= 10 else 'Medium'
            }
            
        except Exception as e:
            # Fallback to simplified explanation
            explanation = self.explain_yaw_impact_simplified(yaw_misalignment)
            explanation['shap_error'] = str(e)
        
        return explanation
    
    def create_interpretation_plot(self, explanation: Dict) -> plt.Figure:
        """Create visualization of model interpretation."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        yaw_angle = explanation['yaw_angle']
        misalignment = yaw_angle - 270.0
        
        # Plot 1: Power Impact
        if 'power_prediction' in explanation:
            power_loss = (explanation['power_baseline'] - explanation['power_prediction'])
            power_loss_pct = power_loss / explanation['power_baseline'] * 100
        else:
            power_loss_pct = explanation['power_loss_percent']
        
        colors = ['green', 'orange', 'red']
        power_color = colors[min(2, int(abs(power_loss_pct) / 5))]
        
        axes[0].bar(['Power Loss'], [abs(power_loss_pct)], color=power_color, alpha=0.7)
        axes[0].set_ylabel('Power Loss (%)')
        yaw_misalign_display = explanation.get('yaw_misalignment', misalignment)
        axes[0].set_title(f'Power Impact\nMisalign: {yaw_misalign_display:.1f}° (Yaw: {yaw_angle:.1f}°)')
        axes[0].set_ylim(0, 20)
        axes[0].text(0, abs(power_loss_pct) + 0.5, f'{abs(power_loss_pct):.1f}%', 
                    ha='center', fontweight='bold')
        
        # Plot 2: Wake Deflection
        if 'wake_impact' in explanation:
            wake_deflection = explanation['wake_impact'] * 100  # Convert to percentage
        else:
            wake_deflection = explanation['wake_deflection']
        
        wake_color = 'blue' if wake_deflection > 0 else 'purple'
        axes[1].bar(['Wake Deflection'], [abs(wake_deflection)], color=wake_color, alpha=0.7)
        axes[1].set_ylabel('Deflection/Impact')
        axes[1].set_title(f'Wake Steering Effect\n{misalignment:.1f}° Misalignment')
        axes[1].text(0, abs(wake_deflection) + 0.1, f'{wake_deflection:.1f}', 
                    ha='center', fontweight='bold')
        
        # Plot 3: Confidence/Interpretation
        confidence_score = 100
        if 'confidence' in explanation:
            conf_map = {'High': 90, 'Medium': 70, 'Low': 50}
            confidence_score = conf_map.get(explanation['confidence'], 70)
        
        conf_color = 'green' if confidence_score > 80 else 'orange' if confidence_score > 60 else 'red'
        axes[2].bar(['Model Confidence'], [confidence_score], color=conf_color, alpha=0.7)
        axes[2].set_ylabel('Confidence (%)')
        axes[2].set_title('Prediction Confidence')
        axes[2].set_ylim(0, 100)
        axes[2].text(0, confidence_score + 2, f'{confidence_score}%', 
                    ha='center', fontweight='bold')
        
        plt.tight_layout()
        return fig


def display_interpretive_agent_gui():
    """
    Display the interpretive agent interface in the GUI.
    Should be called near the yaw slider in the main GUI.
    """
    st.markdown("""
    <div class="agent-box">
    <h4>🧠 AI Interpretive Agent</h4>
    <p style="margin-top: -10px; color: #666;">
    Real-time explanations of ML model predictions to build engineering trust
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ask user if they want to use the interpretive agent
    use_agent = st.radio(
        "**Do you want to explore the AI Interpretive Agent?**",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
        help="Select 'Yes' to test different yaw misalignment angles and see real-time ML model explanations"
    )
    
    if use_agent == "No":
        st.info("💡 Select 'Yes' above to explore how ML models interpret different yaw misalignment angles.")
        return
    
    # Interactive yaw misalignment slider for interpretation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Test Yaw Misalignment:**")
        test_misalignment = st.slider(
            "Yaw Misalignment (°)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="0° = 275° yaw angle (baseline), 15° = 285° yaw angle. Adjust to see real-time ML model explanations"
        )
        
        # Calculate actual yaw angle from misalignment
        actual_yaw = 275.0 + (test_misalignment * 2.0 / 3.0)
        
        # Display current settings
        st.metric("Yaw Misalignment", f"{test_misalignment:.1f}°")
        st.metric("Actual Yaw Angle", f"{actual_yaw:.1f}°")
        st.metric("Deviation from Wind (270°)", f"{actual_yaw - 270.0:.1f}°")
    
    with col2:
        # Create interpreter (without agents for now - can be enhanced later)
        interpreter = WindTurbineInterpreter()
        
        # Get explanation for current yaw misalignment
        explanation = interpreter.explain_interactive_yaw(test_misalignment)
        
        # Display interpretation plot
        fig = interpreter.create_interpretation_plot(explanation)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    # Detailed explanation
    st.markdown("**AI Model Interpretation:**")
    
    explanation_tabs = st.tabs(["📊 Impact Summary", "🔍 Technical Details", "⚙️ Engineering Insights"])
    
    with explanation_tabs[0]:
        if 'power_loss_percent' in explanation:
            power_loss = explanation['power_loss_percent']
        else:
            power_loss = (explanation.get('power_baseline', 5.0) - explanation.get('power_prediction', 4.8)) / explanation.get('power_baseline', 5.0) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Power Impact", 
                f"{power_loss:.1f}% loss",
                delta=f"{-power_loss:.1f}%",
                help="Power reduction due to yaw misalignment"
            )
        
        with col2:
            wake_effect = explanation.get('wake_deflection', explanation.get('wake_impact', 0))
            st.metric(
                "Wake Steering", 
                f"{abs(wake_effect):.1f}° deflection",
                delta=f"+{wake_effect:.1f}°" if wake_effect > 0 else f"{wake_effect:.1f}°",
                help="Expected wake deflection for downstream benefits"
            )
        
        with col3:
            # Use actual deviation from wind direction for benefit assessment
            display_misalign = explanation.get('yaw_misalignment', 0)
            actual_deviation = explanation.get('misalignment', display_misalign)
            net_benefit = "Positive" if actual_deviation > 0 and actual_deviation < 8 else "Questionable"
            st.metric(
                "Net Farm Benefit", 
                net_benefit,
                help="Overall impact considering upstream loss vs downstream gain"
            )
    
    with explanation_tabs[1]:
        if explanation['explanation_type'] == 'SHAP Analysis':
            st.markdown("**SHAP Analysis Results:**")
            if 'power_shap_value' in explanation:
                st.write(f"• Power SHAP Value: {explanation['power_shap_value']:.3f} MW")
                st.write(f"• Wake SHAP Value: {explanation['wake_shap_value']:.3f}")
                st.write(f"• Model Confidence: {explanation.get('confidence', 'Medium')}")
            
        else:
            st.markdown("**Physics-Based Analysis:**")
            if 'key_factors' in explanation:
                for factor, impact in explanation['key_factors'].items():
                    st.write(f"• **{factor}**: {impact}")
        
        st.markdown(f"**Baseline Comparison:**")
        display_misalign = explanation.get('yaw_misalignment', 0)
        actual_yaw = explanation.get('yaw_angle', 275.0)
        actual_deviation = explanation.get('misalignment', display_misalign)
        st.write(f"• Test Misalignment: {display_misalign:.1f}° → Yaw Angle: {actual_yaw:.1f}°")
        st.write(f"• Baseline (optimal): 0° misalignment → 275° yaw angle")
        st.write(f"• Deviation from wind direction (270°): {actual_deviation:.1f}°")
        
    with explanation_tabs[2]:
        st.markdown("**Engineering Recommendations:**")
        
        # Use yaw_misalignment from explanation if available, otherwise calculate from yaw_angle
        display_misalign = explanation.get('yaw_misalignment', explanation.get('misalignment', 0))
        actual_deviation = explanation.get('misalignment', display_misalign)
        
        if abs(actual_deviation) < 2:
            st.success("✅ **Optimal Operation**: Minimal power loss, good for individual turbine optimization")
        elif abs(actual_deviation) < 8:
            st.warning("⚠️ **Wake Steering Mode**: Acceptable power loss for potential downstream benefits")
        else:
            st.error("❌ **Excessive Misalignment**: High power loss, limited wake steering benefit")
        
        # Operational guidance
        st.markdown("**Operational Guidance:**")
        recommendations = []
        
        display_misalign = explanation.get('yaw_misalignment', explanation.get('misalignment', 0))
        actual_deviation = explanation.get('misalignment', display_misalign)
        
        if actual_deviation > 0:
            recommendations.extend([
                f"🔄 Upstream turbine sacrifices {power_loss:.1f}% power for wake steering",
                f"📍 Wake deflected ~{wake_effect:.1f}° - check downstream turbine positions",
                f"⚖️ Net farm benefit depends on downstream turbine layout"
            ])
        else:
            recommendations.extend([
                "🎯 Near-optimal alignment for individual turbine",
                "📈 Maximizes single-turbine power output",
                "🌊 Minimal wake steering - standard operation"
            ])
        
        for rec in recommendations:
            st.write(rec)
        
        # Model trust indicators
        st.markdown("**Model Trust Indicators:**")
        trust_score = 85 if abs(actual_deviation) <= 10 else 65
        trust_color = "green" if trust_score > 80 else "orange"
        
        st.markdown(f"""
        <div style="background-color: {trust_color}20; padding: 10px; border-radius: 5px; border-left: 3px solid {trust_color};">
        <strong>Trust Score: {trust_score}%</strong><br>
        • Model trained on CFD data in this range: {'✅' if abs(actual_deviation) <= 12 else '⚠️'}<br>
        • Physics alignment with expectations: {'✅' if power_loss > 0 else '⚠️'}<br>
        • Prediction consistency: {'✅' if explanation.get('confidence', 'Medium') == 'High' else '⚠️'}
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Demo the interpretive agent
    st.set_page_config(page_title="AI Interpretive Agent Demo", layout="wide")
    st.title("🧠 AI Interpretive Agent for Wind Turbine Control")
    display_interpretive_agent_gui()