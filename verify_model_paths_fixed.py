#!/usr/bin/env python3
"""
Quick verification script to check if model paths have been fixed.
"""

import os
from pathlib import Path

# Set up paths just like in the agents
SCRIPT_DIR = Path(__file__).parent.resolve()

def check_paths():
    """Check if the required model and data files exist at the expected paths."""
    print("Model Path Verification")
    print("=" * 50)
    
    # Check model path (should be relative to SCRIPT_DIR)
    model_path = SCRIPT_DIR / "models" / "tt_opinf_model"
    print(f"Model directory: {model_path}")
    if model_path.exists():
        print("✓ Model directory exists")
        
        # Check model files
        model_files = ['metadata.npz', 'opinf.npz', 'tt_decomp.npz']
        for file in model_files:
            file_path = model_path / file
            if file_path.exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ❌ {file} missing")
    else:
        print("❌ Model directory not found")
    
    print()
    
    # Check grid path (should be relative to SCRIPT_DIR)
    grid_path = SCRIPT_DIR / "data" / "Grid_data.vtk"
    print(f"Grid file: {grid_path}")
    if grid_path.exists():
        print("✓ Grid file exists")
        file_size = grid_path.stat().st_size / (1024*1024)  # MB
        print(f"  Size: {file_size:.1f} MB")
    else:
        print("❌ Grid file not found")
    
    print()
    
    # Test import and initialization
    try:
        print("Testing WakeFlowAgent initialization...")
        from tt_opinf_inference_agent import WakeFlowAgent
        
        # This should use the default paths we just fixed
        agent = WakeFlowAgent()
        print("✓ WakeFlowAgent initialized successfully with default paths")
        print(f"  Model loaded from: {agent.model_dir}")
        print(f"  Grid loaded from: {agent.grid_path}")
        
    except Exception as e:
        print(f"❌ Error initializing WakeFlowAgent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_paths()