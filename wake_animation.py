"""
Wake Flow Animation Module

Creates ParaView-style 3D animations of wind turbine wake flow predictions
using PyVista for rendering. Can be integrated with the GUI or used standalone.

Usage:
    from wake_animation import create_wake_animation_3d
    
    # Create animation from predictions array
    anim_path = create_wake_animation_3d(
        predictions=predictions,  # Shape: (n_timesteps, n_points, 3)
        grid_path="path/to/Grid_data.vtk",
        output_path="wake_animation.gif",
        fps=10
    )
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import tempfile
import shutil

# Try to import visualization libraries
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: pyvista not available. 3D animations will be disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. GIF creation will be disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Default paths
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULT_ML_YAW = SCRIPT_DIR.parent
PROJECT_ROOT = RESULT_ML_YAW.parent
DEFAULT_GRID_PATH = str(SCRIPT_DIR / "data" / "Grid_data.vtk")


def create_wake_animation_3d(
    predictions: np.ndarray,
    grid_path: str = None,
    output_path: str = "wake_animation.gif",
    fps: int = 10,
    frame_skip: int = 2,
    view_angle: Tuple[float, float, float] = (45, -45, 0),
    scalar_bar: bool = True,
    show_turbine: bool = False,
    cmap: str = "coolwarm",
    background_color: str = "white",
    window_size: Tuple[int, int] = (1200, 600),
    clim: Optional[Tuple[float, float]] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    Create a ParaView-style 3D animation of wake flow.
    
    Args:
        predictions: Velocity field array of shape (n_timesteps, n_points, 3)
        grid_path: Path to VTK grid file (defaults to project Grid_data.vtk)
        output_path: Output path for the animation GIF
        fps: Frames per second for the animation
        frame_skip: Skip every N frames (for faster rendering)
        view_angle: Camera view angle (azimuth, elevation, roll)
        scalar_bar: Show color bar
        show_turbine: Show turbine representation
        cmap: Colormap for velocity magnitude
        background_color: Background color
        window_size: Window size (width, height)
        clim: Color limits (min, max). If None, auto-calculated.
        verbose: Print progress information
        
    Returns:
        Path to the created animation file, or None if failed
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Cannot create 3D animation.")
        return None
    
    if not PIL_AVAILABLE:
        print("PIL not available. Cannot create GIF animation.")
        return None
    
    if grid_path is None:
        grid_path = DEFAULT_GRID_PATH
    
    if not os.path.exists(grid_path):
        print(f"Grid file not found: {grid_path}")
        return None
    
    if verbose:
        print(f"Creating 3D wake flow animation...")
        print(f"  Grid: {grid_path}")
        print(f"  Predictions shape: {predictions.shape}")
    
    # Load grid
    grid = pv.read(grid_path)
    
    # Calculate velocity magnitude for all timesteps
    velocity_mag = np.linalg.norm(predictions, axis=2)
    
    # Determine color limits
    if clim is None:
        clim = (velocity_mag.min(), velocity_mag.max())
    
    n_timesteps = predictions.shape[0]
    frames_to_render = list(range(0, n_timesteps, frame_skip))
    
    if verbose:
        print(f"  Rendering {len(frames_to_render)} frames (skip={frame_skip})...")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    try:
        # Use off-screen rendering
        pv.start_xvfb()  # For headless environments
    except:
        pass
    
    # Set up the plotter
    pv.global_theme.background = background_color
    pv.global_theme.font.color = 'black'
    
    for i, t in enumerate(frames_to_render):
        if verbose and i % 5 == 0:
            print(f"    Frame {i+1}/{len(frames_to_render)} (timestep {t})")
        
        # Create plotter for this frame
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        
        # Update grid with current timestep data
        frame_grid = grid.copy()
        frame_grid['Velocity_Magnitude'] = velocity_mag[t]
        frame_grid['Velocity'] = predictions[t]
        
        # Add the wake flow mesh
        plotter.add_mesh(
            frame_grid,
            scalars='Velocity_Magnitude',
            cmap=cmap,
            clim=clim,
            scalar_bar_args={
                'title': 'Velocity (m/s)',
                'vertical': True,
                'position_x': 0.85,
                'position_y': 0.1,
                'width': 0.1,
                'height': 0.8,
                'fmt': '%.1f'
            } if scalar_bar else None,
            show_scalar_bar=scalar_bar,
            opacity=0.9
        )
        
        # Add turbine representation
        if show_turbine:
            # Create a simple turbine representation (cylinder + disk)
            # Get grid bounds to position turbine
            bounds = grid.bounds
            x_min, x_max = bounds[0], bounds[1]
            y_min, y_max = bounds[2], bounds[3]
            z_min, z_max = bounds[4], bounds[5]
            
            # Turbine position (assumed at inlet, centered)
            turbine_x = x_min + (x_max - x_min) * 0.05
            turbine_y = (y_min + y_max) / 2
            turbine_z = (z_min + z_max) / 2
            
            # Rotor disk (126m diameter for NREL 5MW, scale to domain)
            domain_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
            rotor_radius = domain_size * 0.05  # Scale to domain
            
            rotor_disk = pv.Disc(
                center=(turbine_x, turbine_y, turbine_z),
                inner=0,
                outer=rotor_radius,
                normal=(1, 0, 0),
                r_res=1,
                c_res=36
            )
            plotter.add_mesh(rotor_disk, color='black', opacity=0.8)
            
            # Tower (cylinder)
            tower = pv.Cylinder(
                center=(turbine_x, turbine_y, (turbine_z + z_min) / 2),
                direction=(0, 0, 1),
                radius=rotor_radius * 0.1,
                height=turbine_z - z_min
            )
            plotter.add_mesh(tower, color='gray', opacity=0.8)
        
        # Set camera view
        plotter.camera_position = 'xz'
        plotter.camera.azimuth = view_angle[0]
        plotter.camera.elevation = view_angle[1]
        plotter.camera.roll = view_angle[2]
        plotter.camera.zoom(1.2)
        
        # Add title with timestep
        plotter.add_text(
            f"Wake Flow - Timestep: {t}",
            position='upper_left',
            font_size=14,
            color='black'
        )
        
        # Add axes
        plotter.add_axes(line_width=2, labels_off=False)
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plotter.screenshot(frame_path)
        frame_files.append(frame_path)
        
        plotter.close()
    
    # Create GIF from frames
    if verbose:
        print(f"  Combining frames into GIF...")
    
    images = [Image.open(f) for f in frame_files]
    
    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    # Clean up temp files
    shutil.rmtree(temp_dir)
    
    if verbose:
        print(f"  ✓ Animation saved to: {output_path}")
    
    return output_path


def create_wake_slice_animation(
    predictions: np.ndarray,
    grid_path: str = None,
    output_path: str = "wake_slice_animation.gif",
    slice_axis: str = 'z',
    slice_position: float = 0.5,
    fps: int = 8,
    frame_skip: int = 2,
    cmap: str = "RdYlBu_r",
    figsize: Tuple[int, int] = (14, 6),
    clim: Optional[Tuple[float, float]] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    Create a 2D slice animation through the wake flow field.
    
    This is similar to taking a horizontal or vertical slice in ParaView.
    
    Args:
        predictions: Velocity field array of shape (n_timesteps, n_points, 3)
        grid_path: Path to VTK grid file
        output_path: Output path for the animation GIF
        slice_axis: Axis to slice along ('x', 'y', or 'z')
        slice_position: Position along axis as fraction (0-1)
        fps: Frames per second
        frame_skip: Skip frames for faster rendering
        cmap: Colormap
        figsize: Figure size
        clim: Color limits
        verbose: Print progress
        
    Returns:
        Path to animation file or None
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available.")
        return None
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available.")
        return None
    
    if grid_path is None:
        grid_path = DEFAULT_GRID_PATH
    
    if not os.path.exists(grid_path):
        print(f"Grid file not found: {grid_path}")
        return None
    
    if verbose:
        print(f"Creating 2D slice animation...")
    
    # Load grid
    grid = pv.read(grid_path)
    bounds = grid.bounds
    
    # Calculate slice position
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(slice_axis.lower(), 2)
    
    min_val = bounds[axis_idx * 2]
    max_val = bounds[axis_idx * 2 + 1]
    slice_val = min_val + (max_val - min_val) * slice_position
    
    # Get point coordinates
    points = np.array(grid.points)
    
    # Calculate velocity magnitude
    velocity_mag = np.linalg.norm(predictions, axis=2)
    
    if clim is None:
        clim = (velocity_mag.min(), velocity_mag.max())
    
    n_timesteps = predictions.shape[0]
    frames_to_render = list(range(0, n_timesteps, frame_skip))
    
    # Find points near the slice
    slice_tolerance = (max_val - min_val) * 0.05
    slice_mask = np.abs(points[:, axis_idx] - slice_val) < slice_tolerance
    
    if slice_mask.sum() == 0:
        print("No points found in slice. Adjusting tolerance...")
        slice_tolerance = (max_val - min_val) * 0.1
        slice_mask = np.abs(points[:, axis_idx] - slice_val) < slice_tolerance
    
    slice_points = points[slice_mask]
    
    # Determine the 2D coordinates based on slice axis
    if slice_axis.lower() == 'z':
        x_coords = slice_points[:, 0]
        y_coords = slice_points[:, 1]
        xlabel = 'X (Downstream)'
        ylabel = 'Y (Lateral)'
    elif slice_axis.lower() == 'y':
        x_coords = slice_points[:, 0]
        y_coords = slice_points[:, 2]
        xlabel = 'X (Downstream)'
        ylabel = 'Z (Vertical)'
    else:
        x_coords = slice_points[:, 1]
        y_coords = slice_points[:, 2]
        xlabel = 'Y (Lateral)'
        ylabel = 'Z (Vertical)'
    
    if verbose:
        print(f"  Slice at {slice_axis}={slice_val:.1f}, {slice_mask.sum()} points")
        print(f"  Rendering {len(frames_to_render)} frames...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initial scatter plot
    scatter = ax.scatter(
        x_coords, y_coords,
        c=velocity_mag[0, slice_mask],
        cmap=cmap,
        s=2,
        vmin=clim[0],
        vmax=clim[1]
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Velocity Magnitude (m/s)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Wake Flow Slice - Timestep: 0')
    ax.set_aspect('equal', adjustable='box')
    
    # Add turbine marker
    turbine_x = x_coords.min()
    turbine_y = (y_coords.min() + y_coords.max()) / 2
    ax.axvline(x=turbine_x, color='black', linewidth=3, label='Turbine')
    
    plt.tight_layout()
    
    def update(frame_idx):
        t = frames_to_render[frame_idx]
        scatter.set_array(velocity_mag[t, slice_mask])
        ax.set_title(f'Wake Flow Slice - Timestep: {t}')
        return scatter,
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_to_render),
        interval=int(1000/fps), blit=True
    )
    
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    if verbose:
        print(f"  ✓ Animation saved to: {output_path}")
    
    return output_path


def get_rotated_turbine_marker(turbine_x: float, turbine_y: float, rotor_half: float, 
                               yaw_misalignment: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate rotated turbine rotor marker coordinates based on yaw misalignment.
    
    The rotor is initially vertical (aligned with Y-axis). It rotates clockwise 
    as yaw_misalignment increases.
    
    Args:
        turbine_x: Center X position of turbine
        turbine_y: Center Y position of turbine
        rotor_half: Half-length of rotor (distance from center to blade tip)
        yaw_misalignment: Yaw angle in degrees (0-15 typical). Positive = clockwise rotation
    
    Returns:
        Tuple of (x_coords, y_coords) for the rotor line endpoints and center
    """
    # Convert yaw angle from degrees to radians (clockwise = positive)
    angle_rad = np.radians(yaw_misalignment)
    
    # Initial rotor endpoints (vertical line): top and bottom
    # Top point: (turbine_x, turbine_y + rotor_half)
    # Bottom point: (turbine_x, turbine_y - rotor_half)
    
    # Rotate both endpoints around the turbine center
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotate top point (0, rotor_half) around origin
    x_top_rot = 0 * cos_a + rotor_half * sin_a
    y_top_rot = -0 * sin_a + rotor_half * cos_a
    
    # Rotate bottom point (0, -rotor_half) around origin
    x_bottom_rot = 0 * cos_a - rotor_half * sin_a
    y_bottom_rot = -0 * sin_a - rotor_half * cos_a
    
    # Translate back to turbine position
    x_top = turbine_x + x_top_rot
    y_top = turbine_y + y_top_rot
    x_bottom = turbine_x + x_bottom_rot
    y_bottom = turbine_y + y_bottom_rot
    
    # Return line endpoints as arrays
    x_coords = np.array([x_bottom, x_top])
    y_coords = np.array([y_bottom, y_top])
    
    return x_coords, y_coords


def create_wake_contour_animation(
    predictions: np.ndarray,
    grid_path: str = None,
    output_path: str = "wake_contour_animation.gif",
    fps: int = 8,
    frame_skip: int = 2,
    cmap: str = "RdYlBu_r",
    figsize: Tuple[int, int] = (14, 5),
    n_levels: int = 30,
    clim: Optional[Tuple[float, float]] = None,
    verbose: bool = True,
    max_points: int = 10000,  # Limit points for fast rendering
    yaw_misalignment: float = 0.0  # Yaw misalignment angle in degrees
) -> Optional[str]:
    """
    Create a fast contour-style animation similar to ParaView visualization.
    
    Uses sampling and fast triangulation for quick rendering.
    
    Args:
        predictions: Velocity field (n_timesteps, n_points, 3)
        grid_path: Path to VTK grid
        output_path: Output animation path
        fps: Frames per second
        frame_skip: Skip frames
        cmap: Colormap
        figsize: Figure size
        n_levels: Number of contour levels
        clim: Color limits
        verbose: Print progress
        max_points: Maximum points to use (sampling for speed)
        yaw_misalignment: Yaw misalignment angle in degrees (rotates turbine marker clockwise)
        
    Returns:
        Animation file path or None
    """
    if not PYVISTA_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("Required libraries not available.")
        return None
    
    if grid_path is None:
        grid_path = DEFAULT_GRID_PATH
    
    if not os.path.exists(grid_path):
        print(f"Grid file not found: {grid_path}")
        return None
    
    if verbose:
        print(f"Creating contour animation (fast mode)...")
    
    # Load grid and get bounds
    grid = pv.read(grid_path)
    points = np.array(grid.points)
    bounds = grid.bounds
    n_total_points = len(points)
    
    # Calculate velocity magnitude
    velocity_mag = np.linalg.norm(predictions, axis=2)
    
    if clim is None:
        clim = (velocity_mag.min(), velocity_mag.max())
    
    n_timesteps = predictions.shape[0]
    frames_to_render = list(range(0, n_timesteps, frame_skip))
    
    # Sample points if too many (for speed)
    if n_total_points > max_points:
        np.random.seed(42)  # Reproducible sampling
        sample_idx = np.random.choice(n_total_points, max_points, replace=False)
        sample_idx = np.sort(sample_idx)  # Sort for better cache performance
    else:
        sample_idx = np.arange(n_total_points)
    
    x_pts = points[sample_idx, 0]
    y_pts = points[sample_idx, 1]
    
    if verbose:
        print(f"  Using {len(sample_idx)} of {n_total_points} points")
        print(f"  Rendering {len(frames_to_render)} frames...")
    
    # Pre-compute triangulation once (major speedup!)
    from matplotlib.tri import Triangulation
    try:
        triang = Triangulation(x_pts, y_pts)
    except Exception as e:
        print(f"  Triangulation failed: {e}")
        print(f"  Falling back to scatter plot...")
        return create_wake_scatter_animation(
            predictions, grid_path, output_path, fps, frame_skip, cmap, figsize, clim, verbose, yaw_misalignment=yaw_misalignment
        )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    levels = np.linspace(clim[0], clim[1], n_levels)
    
    # Turbine position
    turbine_x = bounds[0] + (bounds[1] - bounds[0]) * 0.02
    turbine_y = (bounds[2] + bounds[3]) / 2
    rotor_half = (bounds[3] - bounds[2]) * 0.1
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    for i, t in enumerate(frames_to_render):
        if verbose:
            print(f"    Frame {i+1}/{len(frames_to_render)}", end='\r')
        
        ax.clear()
        
        # Get velocity for this timestep (sampled points)
        vel = velocity_mag[t, sample_idx]
        
        # Fast tricontourf using pre-computed triangulation
        contour = ax.tricontourf(triang, vel, levels=levels, cmap=cmap, extend='both')
        
        # Add rotated turbine marker
        x_rotor, y_rotor = get_rotated_turbine_marker(turbine_x, turbine_y, rotor_half, yaw_misalignment)
        ax.plot(x_rotor, y_rotor, 'k-', linewidth=5, label='Rotor')
        ax.plot(turbine_x, turbine_y, 'ko', markersize=10, label='Hub')
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(f'Wake Flow (TT-OpInf) - Step {t}/{n_timesteps-1}', fontsize=12)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=80, bbox_inches='tight')
        frame_files.append(frame_path)
    
    plt.close(fig)
    
    if verbose:
        print(f"\n  Combining {len(frame_files)} frames into GIF...")
    
    # Combine into GIF
    images = [Image.open(f) for f in frame_files]
    duration = int(1000 / fps)
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    if verbose:
        print(f"  ✓ Animation saved to: {output_path}")
    
    return output_path


def create_wake_scatter_animation(
    predictions: np.ndarray,
    grid_path: str = None,
    output_path: str = "wake_scatter_animation.gif",
    fps: int = 8,
    frame_skip: int = 2,
    cmap: str = "RdYlBu_r",
    figsize: Tuple[int, int] = (14, 5),
    clim: Optional[Tuple[float, float]] = None,
    verbose: bool = True,
    max_points: int = 8000,
    yaw_misalignment: float = 0.0  # Yaw misalignment angle in degrees
) -> Optional[str]:
    """
    Create a fast scatter-based animation (fallback for triangulation issues).
    
    Args:
        predictions: Velocity field (n_timesteps, n_points, 3)
        grid_path: Path to VTK grid
        output_path: Output animation path
        fps: Frames per second
        frame_skip: Skip frames
        cmap: Colormap
        figsize: Figure size
        clim: Color limits
        verbose: Print progress
        max_points: Maximum points to use for scatter
        yaw_misalignment: Yaw misalignment angle in degrees (rotates turbine marker clockwise)
    
    Returns:
        Animation file path or None
    """
    if not PYVISTA_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return None
    
    if grid_path is None:
        grid_path = DEFAULT_GRID_PATH
    
    if not os.path.exists(grid_path):
        return None
    
    if verbose:
        print(f"Creating scatter animation...")
    
    grid = pv.read(grid_path)
    points = np.array(grid.points)
    bounds = grid.bounds
    n_total = len(points)
    
    velocity_mag = np.linalg.norm(predictions, axis=2)
    
    if clim is None:
        clim = (velocity_mag.min(), velocity_mag.max())
    
    n_timesteps = predictions.shape[0]
    frames_to_render = list(range(0, n_timesteps, frame_skip))
    
    # Sample points
    if n_total > max_points:
        np.random.seed(42)
        sample_idx = np.random.choice(n_total, max_points, replace=False)
    else:
        sample_idx = np.arange(n_total)
    
    x_pts = points[sample_idx, 0]
    y_pts = points[sample_idx, 1]
    
    if verbose:
        print(f"  Using {len(sample_idx)} points, {len(frames_to_render)} frames")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    turbine_x = bounds[0] + (bounds[1] - bounds[0]) * 0.02
    turbine_y = (bounds[2] + bounds[3]) / 2
    rotor_half = (bounds[3] - bounds[2]) * 0.1
    
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    for i, t in enumerate(frames_to_render):
        if verbose:
            print(f"    Frame {i+1}/{len(frames_to_render)}", end='\r')
        
        ax.clear()
        vel = velocity_mag[t, sample_idx]
        
        scatter = ax.scatter(x_pts, y_pts, c=vel, cmap=cmap, s=1, 
                            vmin=clim[0], vmax=clim[1])
        
        # Add rotated turbine marker
        x_rotor, y_rotor = get_rotated_turbine_marker(turbine_x, turbine_y, rotor_half, yaw_misalignment)
        ax.plot(x_rotor, y_rotor, 'k-', linewidth=5, label='Rotor')
        ax.plot(turbine_x, turbine_y, 'ko', markersize=10, label='Hub')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Wake Flow - Step {t}/{n_timesteps-1}')
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=80, bbox_inches='tight')
        frame_files.append(frame_path)
    
    plt.close(fig)
    
    if verbose:
        print(f"\n  Combining frames...")
    
    images = [Image.open(f) for f in frame_files]
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    
    shutil.rmtree(temp_dir)
    
    if verbose:
        print(f"  ✓ Saved: {output_path}")
    
    return output_path


# =============================================================================
# Main function for standalone use
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create wake flow animations")
    parser.add_argument("--vtk-dir", type=str, help="Directory with VTK time series")
    parser.add_argument("--output", type=str, default="wake_animation.gif", help="Output file")
    parser.add_argument("--type", type=str, choices=['3d', 'slice', 'contour'], default='contour')
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    
    args = parser.parse_args()
    
    print("Wake Animation Creator")
    print("=" * 50)
    
    if args.vtk_dir and os.path.exists(args.vtk_dir):
        # Load predictions from VTK files
        vtk_files = sorted([f for f in os.listdir(args.vtk_dir) if f.endswith('.vtk')])
        print(f"Found {len(vtk_files)} VTK files")
        
        predictions_list = []
        for vf in vtk_files:
            mesh = pv.read(os.path.join(args.vtk_dir, vf))
            vel = mesh['Velocity']
            predictions_list.append(vel)
        
        predictions = np.array(predictions_list)
        print(f"Predictions shape: {predictions.shape}")
        
        if args.type == '3d':
            create_wake_animation_3d(predictions, output_path=args.output, fps=args.fps)
        elif args.type == 'slice':
            create_wake_slice_animation(predictions, output_path=args.output, fps=args.fps)
        else:
            create_wake_contour_animation(predictions, output_path=args.output, fps=args.fps)
    else:
        print("Please provide --vtk-dir with VTK time series directory")
