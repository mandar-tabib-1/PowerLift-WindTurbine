"""
Wake Flow Prediction Agent

This agent uses a pre-trained LF-TTOI (GCA-ROM + TT Decomposition + OpInf) model
to predict wind turbine wake flow dynamics at specified yaw angles.

Can be called by an orchestrator agent to generate wake flow predictions and animations.

Usage:
    from agents.wake_flow_prediction_agent import WakeFlowPredictionAgent
    
    agent = WakeFlowPredictionAgent()
    result = agent.predict(yaw_angle=276, n_time_steps=100)
    agent.create_animation(result, output_path='wake_animation.gif')
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

# Add LF_TTOI library to path
# File location: ResultMLYaw/PowerLift/wake_flow_prediction_agent.py
SCRIPT_DIR = Path(__file__).parent  # ResultMLYaw/PowerLift
RESULT_PATH = SCRIPT_DIR.parent     # ResultMLYaw
ROOT_PATH = RESULT_PATH.parent      # HAM_Wind_Energy
LF_TTOI_PATH = ROOT_PATH / 'LF_TTOI'

sys.path.insert(0, str(LF_TTOI_PATH))
sys.path.insert(0, str(ROOT_PATH))

# Import LF-TTOI components
try:
    from lf_ttoi import LF_TTOI
    from utils import (
        compute_error_metrics,
        compute_velocity_magnitude_error,
    )
except ImportError as e:
    print(f"Error importing LF_TTOI library: {e}")
    print(f"Ensure LF_TTOI is available at: {LF_TTOI_PATH}")
    raise


class WakeFlowPredictionAgent:
    """
    Agent for predicting wind turbine wake flow dynamics using LF-TTOI model.
    
    This agent loads a pre-trained model and provides methods for:
    - Predicting flow fields at new yaw angles
    - Generating animations of wake dynamics
    - Exporting predictions to VTK for ParaView visualization
    
    Attributes:
        model: Loaded LF-TTOI model
        device: Computation device (cuda/cpu)
        grid: Reference PyVista grid for visualization
        coordinates: Spatial coordinates of mesh nodes
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        grid_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Wake Flow Prediction Agent.
        
        Args:
            model_path: Path to saved LF-TTOI model directory.
                       Defaults to 'ResultMLYaw/lf_ttoi_model_opinf_yaw_interpolation'
            grid_path: Path to VTK grid file for visualization.
                       Defaults to 'ResultMLYaw/Clipped_Grid_data.vtk'
            device: Computation device ('cuda' or 'cpu'). Auto-detected if None.
        """
        # Set default paths relative to ResultMLYaw folder
        if model_path is None:
            model_path = str(RESULT_PATH / 'lf_ttoi_model_opinf_yaw_interpolation')
        if grid_path is None:
            grid_path = str(RESULT_PATH / 'Clipped_Grid_data.vtk')
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"=" * 60)
        print("WAKE FLOW PREDICTION AGENT")
        print(f"=" * 60)
        print(f"Device: {self.device}")
        print(f"Model path: {model_path}")
        print(f"Grid path: {grid_path}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load reference grid for visualization
        self.grid = self._load_grid(grid_path)
        self.coordinates = np.array(self.grid.points) if self.grid is not None else None
        
        # Store metadata
        self.model_path = model_path
        self.grid_path = grid_path
        
        print(f"\nAgent initialized successfully!")
        print(f"=" * 60)
    
    def _load_model(self, model_path: str) -> LF_TTOI:
        """Load the pre-trained LF-TTOI model."""
        print(f"\nLoading LF-TTOI model...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        model = LF_TTOI.load(model_path, device=self.device)
        
        print(f"  Model type: {model.dynamics_type}")
        print(f"  GCA latent dim: {model.gca_latent_dim}")
        if hasattr(model, 'tt_decomp') and model.tt_decomp is not None:
            print(f"  TT ranks: {model.tt_decomp.ranks}")
        
        return model
    
    def _load_grid(self, grid_path: str):
        """Load the reference VTK grid for visualization."""
        try:
            import pyvista as pv
            
            if os.path.exists(grid_path):
                grid = pv.read(grid_path)
                print(f"\nLoaded reference grid:")
                print(f"  Number of points: {grid.n_points}")
                print(f"  Number of cells: {grid.n_cells}")
                return grid
            else:
                print(f"\nWarning: Grid file not found at {grid_path}")
                print("  VTK export will not be available.")
                return None
        except ImportError:
            print("\nWarning: PyVista not available. VTK export disabled.")
            return None
    
    def predict(
        self,
        yaw_angle: float,
        n_time_steps: int = 100,
        dt: float = 0.1,
        return_latent: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict wake flow field at a specified yaw angle.
        
        Args:
            yaw_angle: Yaw angle in degrees for prediction
            n_time_steps: Number of time steps to predict
            dt: Time step size (default: 0.1)
            return_latent: Whether to return latent trajectories
        
        Returns:
            Dictionary containing:
                - 'predictions': Velocity field (n_time_steps, n_nodes, 3)
                - 'yaw_angle': Input yaw angle
                - 'n_time_steps': Number of time steps
                - 'time': Time array
                - 'velocity_magnitude': Magnitude of velocity (n_time_steps, n_nodes)
                - 'latent_trajectory': (optional) Latent space trajectory
        """
        print(f"\n{'=' * 60}")
        print(f"PREDICTING WAKE FLOW AT YAW = {yaw_angle}°")
        print(f"{'=' * 60}")
        print(f"  Time steps: {n_time_steps}")
        print(f"  dt: {dt}")
        
        # Run prediction
        if return_latent:
            predictions, latent_trajectory = self.model.predict(
                param_value=yaw_angle,
                n_time_steps=n_time_steps,
                return_latent=True
            )
        else:
            predictions = self.model.predict(
                param_value=yaw_angle,
                n_time_steps=n_time_steps,
                return_latent=False
            )
            latent_trajectory = None
        
        # Convert to numpy if tensor
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # Ensure float64 for compatibility
        predictions = predictions.astype(np.float64)
        
        # Compute velocity magnitude
        velocity_magnitude = np.linalg.norm(predictions, axis=-1)
        
        # Create time array
        time_array = np.arange(predictions.shape[0]) * dt
        
        print(f"\nPrediction complete!")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Velocity range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  Magnitude range: [{velocity_magnitude.min():.4f}, {velocity_magnitude.max():.4f}]")
        
        result = {
            'predictions': predictions,
            'yaw_angle': yaw_angle,
            'n_time_steps': n_time_steps,
            'dt': dt,
            'time': time_array,
            'velocity_magnitude': velocity_magnitude,
            'Ux': predictions[:, :, 0],
            'Uy': predictions[:, :, 1],
            'Uz': predictions[:, :, 2],
        }
        
        if return_latent and latent_trajectory is not None:
            if isinstance(latent_trajectory, torch.Tensor):
                latent_trajectory = latent_trajectory.detach().cpu().numpy()
            result['latent_trajectory'] = latent_trajectory
        
        return result
    
    def create_animation(
        self,
        result: Dict[str, np.ndarray],
        output_path: str = 'wake_animation.gif',
        field: str = 'velocity_magnitude',
        slice_axis: str = 'z',
        slice_index: Optional[int] = None,
        fps: int = 10,
        dpi: int = 100,
        cmap: str = 'jet',
        figsize: Tuple[int, int] = (12, 8),
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> str:
        """
        Create an animation of the wake flow dynamics.
        
        Args:
            result: Prediction result dictionary from predict()
            output_path: Output file path (.gif or .mp4)
            field: Field to visualize ('velocity_magnitude', 'Ux', 'Uy', 'Uz')
            slice_axis: Axis for 2D slice ('x', 'y', 'z')
            slice_index: Index for slice. If None, uses middle slice.
            fps: Frames per second
            dpi: DPI for output
            cmap: Colormap
            figsize: Figure size
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
        
        Returns:
            Path to saved animation file
        """
        print(f"\n{'=' * 60}")
        print(f"CREATING WAKE FLOW ANIMATION")
        print(f"{'=' * 60}")
        
        if self.coordinates is None:
            raise ValueError("Grid coordinates not available. Cannot create animation.")
        
        # Get field data
        if field in result:
            data = result[field]
        elif field == 'velocity_magnitude':
            data = result['velocity_magnitude']
        else:
            raise ValueError(f"Unknown field: {field}")
        
        n_time, n_nodes = data.shape[:2] if data.ndim > 1 else (data.shape[0], 1)
        
        # Get unique coordinates for the slice
        coords = self.coordinates
        x_unique = np.unique(coords[:, 0])
        y_unique = np.unique(coords[:, 1])
        z_unique = np.unique(coords[:, 2])
        
        print(f"  Field: {field}")
        print(f"  Time steps: {n_time}")
        print(f"  Grid dimensions: X={len(x_unique)}, Y={len(y_unique)}, Z={len(z_unique)}")
        
        # Determine slice
        if slice_axis == 'z':
            if slice_index is None:
                slice_index = len(z_unique) // 2
            slice_val = z_unique[min(slice_index, len(z_unique)-1)]
            mask = np.abs(coords[:, 2] - slice_val) < (z_unique[1] - z_unique[0]) / 2
            x_plot = coords[mask, 0]
            y_plot = coords[mask, 1]
            xlabel, ylabel = 'X', 'Y'
        elif slice_axis == 'y':
            if slice_index is None:
                slice_index = len(y_unique) // 2
            slice_val = y_unique[min(slice_index, len(y_unique)-1)]
            mask = np.abs(coords[:, 1] - slice_val) < (y_unique[1] - y_unique[0]) / 2
            x_plot = coords[mask, 0]
            y_plot = coords[mask, 2]
            xlabel, ylabel = 'X', 'Z'
        else:  # x
            if slice_index is None:
                slice_index = len(x_unique) // 2
            slice_val = x_unique[min(slice_index, len(x_unique)-1)]
            mask = np.abs(coords[:, 0] - slice_val) < (x_unique[1] - x_unique[0]) / 2
            x_plot = coords[mask, 1]
            y_plot = coords[mask, 2]
            xlabel, ylabel = 'Y', 'Z'
        
        print(f"  Slice: {slice_axis}={slice_val:.2f} (index={slice_index})")
        print(f"  Points in slice: {mask.sum()}")
        
        # Set color limits
        if vmin is None:
            vmin = data[:, mask].min()
        if vmax is None:
            vmax = data[:, mask].max()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initial plot
        scatter = ax.scatter(
            x_plot, y_plot, 
            c=data[0, mask], 
            cmap=cmap, 
            vmin=vmin, vmax=vmax,
            s=1
        )
        
        cbar = plt.colorbar(scatter, ax=ax, label=field)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        title = ax.set_title(f"Yaw={result['yaw_angle']}°, t={result['time'][0]:.2f}s")
        
        def update(frame):
            scatter.set_array(data[frame, mask])
            title.set_text(f"Yaw={result['yaw_angle']}°, t={result['time'][frame]:.2f}s")
            return scatter, title
        
        print(f"\nGenerating animation with {n_time} frames...")
        
        anim = animation.FuncAnimation(
            fig, update, frames=n_time, 
            interval=1000/fps, blit=True
        )
        
        # Save animation
        output_path = str(output_path)
        if output_path.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi)
        else:
            anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        
        plt.close(fig)
        
        print(f"Animation saved to: {output_path}")
        return output_path
    
    def export_to_vtk(
        self,
        result: Dict[str, np.ndarray],
        output_dir: str,
        base_name: str = 'wake_prediction',
        time_stride: int = 1
    ) -> str:
        """
        Export predictions to VTK files for ParaView visualization.
        
        Args:
            result: Prediction result dictionary from predict()
            output_dir: Output directory for VTK files
            base_name: Base name for output files
            time_stride: Export every N time steps
        
        Returns:
            Output directory path
        """
        print(f"\n{'=' * 60}")
        print(f"EXPORTING TO VTK")
        print(f"{'=' * 60}")
        
        if self.grid is None:
            raise ValueError("Reference grid not available. Cannot export to VTK.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = result['predictions']
        n_time = predictions.shape[0]
        time_indices = list(range(0, n_time, time_stride))
        
        print(f"  Output directory: {output_dir}")
        print(f"  Exporting {len(time_indices)} VTK files...")
        
        for i, t in enumerate(time_indices):
            filepath = os.path.join(output_dir, f'{base_name}_{t:04d}.vtk')
            
            # Create a copy of reference grid
            output_grid = self.grid.copy()
            
            # Clear existing point data
            for key in list(output_grid.point_data.keys()):
                del output_grid.point_data[key]
            
            # Add velocity data
            velocity = predictions[t]
            output_grid.point_data['Velocity'] = np.ascontiguousarray(velocity)
            output_grid.point_data['Velocity_Magnitude'] = result['velocity_magnitude'][t]
            output_grid.point_data['Ux'] = np.ascontiguousarray(velocity[:, 0])
            output_grid.point_data['Uy'] = np.ascontiguousarray(velocity[:, 1])
            output_grid.point_data['Uz'] = np.ascontiguousarray(velocity[:, 2])
            
            # Add metadata
            output_grid.field_data['TimeIndex'] = np.array([t])
            output_grid.field_data['YawAngle'] = np.array([result['yaw_angle']])
            output_grid.field_data['Time'] = np.array([result['time'][t]])
            
            output_grid.save(filepath)
            
            if (i + 1) % 10 == 0:
                print(f"    Exported {i + 1}/{len(time_indices)} files")
        
        print(f"\nExported {len(time_indices)} VTK files to: {output_dir}")
        return output_dir
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            'model_path': self.model_path,
            'device': self.device,
            'dynamics_type': self.model.dynamics_type,
            'gca_latent_dim': self.model.gca_latent_dim,
        }
        
        if hasattr(self.model, 'tt_decomp') and self.model.tt_decomp is not None:
            info['tt_ranks'] = self.model.tt_decomp.ranks
        
        if self.grid is not None:
            info['n_grid_points'] = self.grid.n_points
            info['n_grid_cells'] = self.grid.n_cells
        
        return info
    
    def __call__(
        self,
        yaw_angle: float,
        n_time_steps: int = 100,
        create_animation: bool = True,
        export_vtk: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Main entry point for orchestrator calls.
        
        Args:
            yaw_angle: Yaw angle in degrees
            n_time_steps: Number of time steps to predict
            create_animation: Whether to create animation
            export_vtk: Whether to export VTK files
            output_dir: Output directory for files
        
        Returns:
            Dictionary with predictions and output file paths
        """
        print(f"\n{'#' * 60}")
        print(f"# WAKE FLOW PREDICTION AGENT CALLED")
        print(f"# Yaw Angle: {yaw_angle}°")
        print(f"# Time Steps: {n_time_steps}")
        print(f"{'#' * 60}")
        
        # Set default output directory
        if output_dir is None:
            output_dir = str(RESULT_PATH / f'agent_output_yaw{yaw_angle}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Predict
        result = self.predict(yaw_angle, n_time_steps, return_latent=True)
        
        # Create outputs
        output_files = {}
        
        if create_animation:
            anim_path = os.path.join(output_dir, f'wake_yaw{yaw_angle}.gif')
            self.create_animation(result, output_path=anim_path)
            output_files['animation'] = anim_path
        
        if export_vtk and self.grid is not None:
            vtk_dir = os.path.join(output_dir, 'vtk')
            self.export_to_vtk(result, vtk_dir, time_stride=5)
            output_files['vtk_directory'] = vtk_dir
        
        # Summary
        result['output_dir'] = output_dir
        result['output_files'] = output_files
        
        print(f"\n{'#' * 60}")
        print(f"# AGENT TASK COMPLETE")
        print(f"# Output directory: {output_dir}")
        print(f"{'#' * 60}")
        
        return result


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Wake Flow Prediction Agent - Predict wind turbine wake dynamics'
    )
    parser.add_argument(
        '--yaw', type=float, required=True,
        help='Yaw angle in degrees'
    )
    parser.add_argument(
        '--time-steps', type=int, default=100,
        help='Number of time steps to predict (default: 100)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-animation', action='store_true',
        help='Skip animation generation'
    )
    parser.add_argument(
        '--export-vtk', action='store_true',
        help='Export predictions to VTK files'
    )
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='Path to saved LF-TTOI model'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        choices=['cuda', 'cpu'],
        help='Computation device'
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = WakeFlowPredictionAgent(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run prediction
    result = agent(
        yaw_angle=args.yaw,
        n_time_steps=args.time_steps,
        create_animation=not args.no_animation,
        export_vtk=args.export_vtk,
        output_dir=args.output_dir
    )
    
    print(f"\nPrediction complete! Results saved to: {result['output_dir']}")
    return result


if __name__ == '__main__':
    main()
