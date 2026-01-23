"""
Rotor Power Prediction Agent with Uncertainty Quantification

This module provides an agent for predicting wind turbine rotor power 
transient dynamics based on yaw angle using a trained Gaussian Process model.

Usage:
    from rotor_power_agent import RotorPowerAgent
    
    # Initialize the agent (loads the trained model)
    agent = RotorPowerAgent()
    
    # Predict power dynamics for a new yaw angle
    results = agent.predict(yaw_angle=278)
    
    # Visualize the prediction
    agent.plot_prediction(results)
    
    # Compare multiple yaw angles
    agent.compare_yaw_angles([270, 275, 280, 285])
    
    # Get summary statistics
    stats = agent.get_statistics(yaw_angle=278)
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


class RotorPowerAgent:
    """
    Agent for predicting rotor power transient dynamics with uncertainty quantification.
    
    This agent loads a trained Gaussian Process model and provides methods to:
    - Predict power trajectories for any yaw angle
    - Quantify prediction uncertainty
    - Visualize transient dynamics
    
    Attributes:
        gp_model: Trained Gaussian Process Regressor
        scaler_X: StandardScaler for input features
        scaler_y: StandardScaler for target variable
        yaw_angles_trained: List of yaw angles used in training
        metadata: Dictionary with model performance metrics
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the agent by loading the trained model.
        
        Args:
            model_path: Path to the saved model file (.joblib).
                        If None, looks in the default location.
        """
        if model_path is None:
            # Default path - adjust as needed
            model_path = Path(__file__).parent / "rotor_power_gp_model.joblib"
        
        print(f"Loading model from: {model_path}")
        artifacts = joblib.load(model_path)
        
        self.gp_model = artifacts['gp_model']
        self.scaler_X = artifacts['scaler_X']
        self.scaler_y = artifacts['scaler_y']
        self.yaw_angles_trained = artifacts['yaw_angles_trained']
        self.metadata = artifacts['metadata']
        
        print(f"Model loaded successfully!")
        print(f"Trained on yaw angles: {self.yaw_angles_trained}")
        print(f"Model performance: RMSE={self.metadata['rmse_mw']:.4f} MW, R²={self.metadata['r2_score']:.4f}")
    
    def predict(self, yaw_angle: float, n_time_points: int = 100, 
                return_samples: bool = False, n_samples: int = 100) -> dict:
        """
        Predict rotor power transient dynamics for a given yaw angle.
        
        The model uses normalized time (0 to 1) to represent the transient
        trajectory, allowing predictions without knowing the absolute time.
        
        Args:
            yaw_angle: The yaw angle (in degrees) for prediction
            n_time_points: Number of time points to predict (default: 100)
            return_samples: Whether to return posterior samples (default: False)
            n_samples: Number of posterior samples if return_samples=True
        
        Returns:
            dict with keys:
                - 'normalized_time': Array of normalized time points (0 to 1)
                - 'power_mean_MW': Predicted mean power (MW)
                - 'power_std_MW': Predicted standard deviation (MW)
                - 'power_lower_95_MW': Lower 95% confidence bound (MW)
                - 'power_upper_95_MW': Upper 95% confidence bound (MW)
                - 'yaw_angle': Input yaw angle
                - 'samples' (optional): Posterior samples if requested
        """
        # Check if yaw angle is within training range
        min_yaw = min(self.yaw_angles_trained)
        max_yaw = max(self.yaw_angles_trained)
        
        if yaw_angle < min_yaw or yaw_angle > max_yaw:
            print(f"⚠️ Warning: Yaw angle {yaw_angle}° is outside training range [{min_yaw}°, {max_yaw}°]")
            print("  Predictions may have higher uncertainty (extrapolation)")
        
        # Create prediction grid
        normalized_time = np.linspace(0, 1, n_time_points)
        X_pred = np.column_stack([
            np.full(n_time_points, yaw_angle),
            normalized_time
        ])
        
        # Scale features
        X_pred_scaled = self.scaler_X.transform(X_pred)
        
        # Predict with uncertainty
        y_pred_scaled, y_std_scaled = self.gp_model.predict(X_pred_scaled, return_std=True)
        
        # Transform back to original scale
        power_mean = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        power_std = y_std_scaled * self.scaler_y.scale_[0]
        
        # Compute confidence intervals (95% = ±1.96 std)
        power_lower_95 = power_mean - 1.96 * power_std
        power_upper_95 = power_mean + 1.96 * power_std
        
        results = {
            'normalized_time': normalized_time,
            'power_mean_MW': power_mean,
            'power_std_MW': power_std,
            'power_lower_95_MW': power_lower_95,
            'power_upper_95_MW': power_upper_95,
            'yaw_angle': yaw_angle
        }
        
        # Optional: return posterior samples
        if return_samples:
            y_samples_scaled = self.gp_model.sample_y(X_pred_scaled, n_samples=n_samples, random_state=42)
            samples = self.scaler_y.inverse_transform(y_samples_scaled).T
            results['samples'] = samples
        
        return results
    
    def plot_prediction(self, results: dict, ax=None, show_samples: bool = False,
                        title: str = None, save_path: str = None):
        """
        Visualize the predicted power transient dynamics.
        
        Args:
            results: Output from predict() method
            ax: Matplotlib axis (creates new figure if None)
            show_samples: Whether to show posterior samples
            title: Custom plot title
            save_path: Path to save the figure
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        t = results['normalized_time']
        yaw = results['yaw_angle']
        
        # Plot confidence interval
        ax.fill_between(t, results['power_lower_95_MW'], results['power_upper_95_MW'],
                        alpha=0.3, color='steelblue', label='95% Confidence Interval')
        
        # Plot ±1 std interval
        ax.fill_between(t, 
                        results['power_mean_MW'] - results['power_std_MW'],
                        results['power_mean_MW'] + results['power_std_MW'],
                        alpha=0.4, color='steelblue', label='±1σ Uncertainty')
        
        # Plot mean prediction
        ax.plot(t, results['power_mean_MW'], 'b-', linewidth=2, label='Mean Prediction')
        
        # Optionally plot samples
        if show_samples and 'samples' in results:
            for sample in results['samples'][:20]:
                ax.plot(t, sample, 'gray', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Normalized Time (0-1)', fontsize=12)
        ax.set_ylabel('Rotor Power (MW)', fontsize=12)
        
        if title is None:
            title = f'Predicted Rotor Power Transient Dynamics\nYaw Angle = {yaw}°'
        ax.set_title(title, fontsize=14)
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return ax
    
    def compare_yaw_angles(self, yaw_angles: list, n_time_points: int = 100,
                           save_path: str = None):
        """
        Compare power dynamics across multiple yaw angles.
        
        Args:
            yaw_angles: List of yaw angles to compare
            n_time_points: Number of time points for prediction
            save_path: Path to save the figure
            
        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(yaw_angles)))
        
        for i, yaw in enumerate(yaw_angles):
            results = self.predict(yaw_angle=yaw, n_time_points=n_time_points)
            
            ax.fill_between(results['normalized_time'], 
                           results['power_lower_95_MW'], 
                           results['power_upper_95_MW'],
                           alpha=0.2, color=colors[i])
            ax.plot(results['normalized_time'], results['power_mean_MW'], 
                   color=colors[i], linewidth=2, label=f'Yaw = {yaw}°')
        
        ax.set_xlabel('Normalized Time (0-1)', fontsize=12)
        ax.set_ylabel('Rotor Power (MW)', fontsize=12)
        ax.set_title('Comparison of Rotor Power Dynamics Across Yaw Angles', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        return fig
    
    def get_statistics(self, yaw_angle: float, n_time_points: int = 100) -> dict:
        """
        Get summary statistics for the predicted power dynamics.
        
        Args:
            yaw_angle: The yaw angle for prediction
            n_time_points: Number of time points
            
        Returns:
            dict with summary statistics
        """
        results = self.predict(yaw_angle, n_time_points)
        
        return {
            'yaw_angle': yaw_angle,
            'mean_power_MW': np.mean(results['power_mean_MW']),
            'max_power_MW': np.max(results['power_mean_MW']),
            'min_power_MW': np.min(results['power_mean_MW']),
            'power_range_MW': np.max(results['power_mean_MW']) - np.min(results['power_mean_MW']),
            'mean_uncertainty_MW': np.mean(results['power_std_MW']),
            'max_uncertainty_MW': np.max(results['power_std_MW'])
        }


# Example usage when run as script
if __name__ == "__main__":
    # Initialize agent
    agent = RotorPowerAgent()
    
    # Predict for a new yaw angle
    results = agent.predict(yaw_angle=278, return_samples=True)
    
    # Plot prediction
    agent.plot_prediction(results, show_samples=True)
    plt.show()
    
    # Print statistics
    stats = agent.get_statistics(278)
    print("\nStatistics for yaw=278°:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
