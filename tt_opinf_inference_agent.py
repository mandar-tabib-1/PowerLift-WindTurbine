"""
TT-OpInf Inference Agent

Standalone inference module for wind turbine wake prediction.
Can be called by an orchestrator with new yaw angle and time period.

Usage:
    from agents.tt_opinf_inference_agent import WakeFlowAgent
    
    agent = WakeFlowAgent(model_dir="path/to/tt_opinf_model", grid_path="path/to/Grid_data.vtk")
    
    # Export all timesteps
    predictions, vtk_dir = agent.predict(yaw_angle=275, n_timesteps=100)
    
    # Export every 5th timestep (faster, smaller output)
    predictions, vtk_dir = agent.predict(yaw_angle=275, n_timesteps=100, export_every=5)
    
    # Export every 10th timestep
    predictions, vtk_dir = agent.predict(yaw_angle=275, n_timesteps=100, export_every=10)
    
    # Skip VTK export entirely (fastest)
    predictions, _ = agent.predict(yaw_angle=275, n_timesteps=100, export_vtk=False)

Or from command line:
    python tt_opinf_inference_agent.py --yaw 275 --timesteps 100 --output ./results
    python tt_opinf_inference_agent.py --yaw 275 --timesteps 100 --export-every 5  # Every 5th step
"""

import os
import sys
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple, Union

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULT_ML_YAW = SCRIPT_DIR.parent  # ResultMLYaw folder
PROJECT_ROOT = RESULT_ML_YAW.parent  # HAM_Wind_Energy folder (where TT_OpInf is)
sys.path.insert(0, str(PROJECT_ROOT))

# Import TT-OpInf module
from TT_OpInf.tt_opinf import TT_OpInf


class WakeFlowAgent:
    """
    Agent for wind turbine wake flow prediction using TT-OpInf model.
    
    This agent can be instantiated once and called multiple times for
    different yaw angles, making it efficient for orchestrator use.
    """
    
    def __init__(
        self,
        model_dir: str = None,
        grid_path: str = None,
        default_output_dir: str = None
    ):
        """
        Initialize the Wake Flow Agent.
        
        Args:
            model_dir: Path to saved TT-OpInf model directory
            grid_path: Path to VTK grid file for mesh structure
            default_output_dir: Default directory for VTK output
        """
        # Set default paths relative to project root
        if model_dir is None:
            model_dir = str(PROJECT_ROOT / "ResultMLYaw" / "tt_opinf_model")
        if grid_path is None:
            grid_path = str(PROJECT_ROOT / "ResultMLYaw" / "Grid_data.vtk")
        if default_output_dir is None:
            default_output_dir = str(PROJECT_ROOT / "ResultMLYaw" / "agent_predictions")
            
        self.model_dir = model_dir
        self.grid_path = grid_path
        self.default_output_dir = default_output_dir
        
        # Load model and grid
        self._load_model()
        self._load_grid()
        
        print(f"WakeFlowAgent initialized:")
        print(f"  Model: {self.model_dir}")
        print(f"  Grid: {self.grid_path}")
        print(f"  Training yaw angles: {self.model.param_values}")
        
    def _load_model(self):
        """Load the TT-OpInf model."""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model not found: {self.model_dir}")
        self.model = TT_OpInf.load(self.model_dir)
        
    def _load_grid(self):
        """Load the VTK grid for mesh structure."""
        try:
            import pyvista as pv
            if not os.path.exists(self.grid_path):
                raise FileNotFoundError(f"Grid not found: {self.grid_path}")
            self.grid = pv.read(self.grid_path)
            self.n_points = self.grid.n_points
        except ImportError:
            print("Warning: pyvista not available. VTK export will be disabled.")
            self.grid = None
            self.n_points = self.model.n_space
            
    def predict(
        self,
        yaw_angle: float,
        n_timesteps: int = 100,
        output_dir: str = None,
        export_vtk: bool = True,
        export_every: int = 1,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Predict velocity field for a new yaw angle and export to VTK.
        
        Args:
            yaw_angle: New yaw angle in degrees (e.g., 275.5)
            n_timesteps: Number of timesteps to predict
            output_dir: Output directory for VTK files (None = auto-generate)
            export_vtk: Whether to export VTK files (set False to skip VTK export)
            export_every: Export every N timesteps to reduce file count and speed up export:
                          1 = all timesteps (default)
                          5 = every 5th timestep (20% of files)
                          10 = every 10th timestep (10% of files)
            verbose: Print progress information
            
        Returns:
            predictions: numpy array of shape (n_timesteps+1, n_space, 3)
            vtk_dir: Path to VTK output directory (None if export_vtk=False)
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"TT-OpInf Inference Agent")
            print(f"{'='*60}")
            print(f"  Input yaw angle: {yaw_angle}°")
            print(f"  Timesteps: {n_timesteps}")
        
        # Step 1: Predict velocity field
        t1 = time.time()
        predictions = self.model.predict(
            param_value=yaw_angle,
            n_time_steps=n_timesteps
        )
        pred_time = time.time() - t1
        
        if verbose:
            print(f"\n  ✓ Prediction: {pred_time:.2f}s")
            print(f"    Output shape: {predictions.shape}")
            print(f"    Velocity range: [{predictions.min():.2f}, {predictions.max():.2f}] m/s")
        
        # Step 2: Export to VTK (if enabled)
        vtk_dir = None
        if export_vtk and self.grid is not None:
            t2 = time.time()
            vtk_dir = self._export_vtk(
                predictions, yaw_angle, output_dir, export_every, verbose
            )
            export_time = time.time() - t2
            if verbose:
                print(f"  ✓ VTK export: {export_time:.2f}s")
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\n  Total time: {total_time:.2f}s")
            print(f"{'='*60}")
        
        return predictions, vtk_dir
    
    def _export_vtk(
        self,
        predictions: np.ndarray,
        yaw_angle: float,
        output_dir: str,
        export_every: int,
        verbose: bool
    ) -> str:
        """Export predictions to VTK files with proper time-series naming."""
        # Create clean yaw angle string (no decimals) for folder name
        yaw_clean = f"{int(round(yaw_angle))}"
        
        if output_dir is None:
            output_dir = os.path.join(self.default_output_dir, f"yaw_{yaw_clean}")
        os.makedirs(output_dir, exist_ok=True)
        
        total_timesteps = predictions.shape[0]
        time_indices = list(range(0, total_timesteps, export_every))
        n_files = len(time_indices)
        
        if verbose:
            print(f"    Exporting {n_files} of {total_timesteps} timesteps (every {export_every})")
        
        # Use VTK/ParaView-friendly naming: basename.0000.vtk format
        # This format is recognized as a time series by ParaView
        basename = f"wake_yaw{yaw_clean}"
        
        for file_idx, t in enumerate(time_indices):
            vtk_grid = self.grid.copy()
            velocity = predictions[t]
            
            vtk_grid['Velocity'] = velocity
            vtk_grid['Ux'] = velocity[:, 0]
            vtk_grid['Uy'] = velocity[:, 1]
            vtk_grid['Uz'] = velocity[:, 2]
            vtk_grid['Velocity_Magnitude'] = np.linalg.norm(velocity, axis=1)
            
            # ParaView time-series format: name.0000.vtk, name.0001.vtk, etc.
            filename = f'{basename}.{file_idx:04d}.vtk'
            vtk_grid.save(os.path.join(output_dir, filename))
        
        if verbose:
            print(f"    Saved to: {output_dir}")
            print(f"    File pattern: {basename}.####.vtk (ParaView time-series format)")
        
        return output_dir
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            "model_dir": self.model_dir,
            "grid_path": self.grid_path,
            "training_yaw_angles": list(self.model.param_values),
            "n_spatial_points": self.model.n_space,
            "n_training_timesteps": self.model.n_time_train,
            "dt": self.model.dt,
            "tt_ranks": self.model.tt_decomp.ranks
        }


def run_inference(
    yaw_angle: float,
    n_timesteps: int = 100,
    model_dir: str = None,
    grid_path: str = None,
    output_dir: str = None,
    export_every: int = 1
) -> Tuple[np.ndarray, str]:
    """
    Standalone function for single inference call.
    
    This is a convenience function for orchestrators that prefer
    functional calls over object-oriented approach.
    
    Args:
        yaw_angle: Yaw angle in degrees
        n_timesteps: Number of timesteps to predict
        model_dir: Path to saved TT-OpInf model
        grid_path: Path to VTK grid file
        output_dir: Output directory for VTK files
        export_every: Export every N timesteps
        
    Returns:
        predictions: numpy array of velocity field
        vtk_dir: Path to VTK output directory
    """
    agent = WakeFlowAgent(
        model_dir=model_dir,
        grid_path=grid_path
    )
    return agent.predict(
        yaw_angle=yaw_angle,
        n_timesteps=n_timesteps,
        output_dir=output_dir,
        export_every=export_every
    )


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command line interface for the agent."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TT-OpInf Wake Flow Prediction Agent"
    )
    parser.add_argument(
        "--yaw", type=float, required=True,
        help="Yaw angle in degrees (e.g., 275)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100,
        help="Number of timesteps to predict (default: 100)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to TT-OpInf model directory"
    )
    parser.add_argument(
        "--grid", type=str, default=None,
        help="Path to VTK grid file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for VTK files"
    )
    parser.add_argument(
        "--export-every", type=int, default=1,
        help="Export every N timesteps (default: 1)"
    )
    parser.add_argument(
        "--no-vtk", action="store_true",
        help="Skip VTK export, return predictions only"
    )
    
    args = parser.parse_args()
    
    # Run inference
    agent = WakeFlowAgent(
        model_dir=args.model,
        grid_path=args.grid
    )
    
    predictions, vtk_dir = agent.predict(
        yaw_angle=args.yaw,
        n_timesteps=args.timesteps,
        output_dir=args.output,
        export_vtk=not args.no_vtk,
        export_every=args.export_every
    )
    
    print(f"\nResults:")
    print(f"  Predictions shape: {predictions.shape}")
    if vtk_dir:
        print(f"  VTK files: {vtk_dir}")


if __name__ == "__main__":
    main()
