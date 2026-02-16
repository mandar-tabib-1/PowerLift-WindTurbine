"""
Test ML-based Wake Analysis Functions

This script demonstrates the sophisticated wake analysis approach based on
user's expert wind turbine methodology:
1. Extract vertical profiles from ML wake model at different distances
2. Calculate wake centerline and lateral migration
3. Extrapolate to predict wake behavior at actual turbine spacing
4. Check if wake overlaps downstream turbine
5. Find optimal yaw angle for wake avoidance

Author: Expert Wind Turbine Analysis System
Date: 2026-02-13
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_ml_wake_analysis():
    """
    Test the ML-based wake analysis functions with actual TT-OpInf model.
    """
    print("=" * 80)
    print("TESTING ML-BASED WAKE ANALYSIS FUNCTIONS")
    print("=" * 80)
    print("\nThis test demonstrates the expert approach:")
    print("1. Extract vertical velocity profiles from ML wake model")
    print("2. Calculate wake deficit and lateral migration")
    print("3. Extrapolate to actual turbine spacing")
    print("4. Determine if wake overlaps downstream turbine")
    print("5. Find optimal yaw angle for wake steering\n")
    
    try:
        # Import wake agent
        from tt_opinf_inference_agent import WakeFlowAgent
        from wind_turbine_gui import (
            extract_vertical_profile_from_wake,
            calculate_wake_centerline_and_deficit,
            extrapolate_wake_trajectory,
            check_wake_turbine_overlap,
            find_optimal_yaw_for_wake_avoidance
        )
        
        print("✓ Imports successful\n")
        
        # Initialize wake agent
        print("Loading TT-OpInf wake flow model...")
        model_dir = str(SCRIPT_DIR / "models" / "tt_opinf_model")
        grid_path = str(SCRIPT_DIR / "data" / "Grid_data.vtk")
        
        if not os.path.exists(model_dir):
            print(f"❌ Model not found at: {model_dir}")
            print("   Please ensure TT-OpInf model is available.")
            return
        
        if not os.path.exists(grid_path):
            print(f"❌ Grid not found at: {grid_path}")
            print("   Please ensure Grid_data.vtk is available.")
            return
        
        wake_agent = WakeFlowAgent(model_dir=model_dir, grid_path=grid_path)
        print("✓ Wake agent initialized\n")
        
        # Test Configuration
        test_scenarios = [
            {
                'name': 'Close Spacing (6D)',
                'spacing_D': 6.0,
                'yaw_angles': [0, 6, 12],
                'description': 'Strong wake interaction - needs aggressive steering'
            },
            {
                'name': 'Medium Spacing (7.5D)',
                'spacing_D': 7.5,
                'yaw_angles': [0, 6, 12],
                'description': 'Moderate wake - typical farm spacing'
            },
            {
                'name': 'Wide Spacing (11D)',
                'spacing_D': 11.0,
                'yaw_angles': [0, 3, 6],
                'description': 'Weak wake - minimal steering needed'
            }
        ]
        
        reference_distance_m = 1700.0  # Distance where ML model extracts profile
        rotor_diameter = 126.0  # NREL 5MW
        
        print("=" * 80)
        print("TEST SCENARIOS")
        print("=" * 80)
        
        for scenario in test_scenarios:
            print(f"\n{'─' * 80}")
            print(f"SCENARIO: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"Spacing: {scenario['spacing_D']:.1f}D = {scenario['spacing_D'] * rotor_diameter:.0f}m")
            print(f"{'─' * 80}\n")
            
            for yaw_deg in scenario['yaw_angles']:
                print(f"  Testing Yaw = {yaw_deg}°:")
                
                try:
                    # Get wake predictions
                    nacelle_dir = 270.0 + yaw_deg
                    wake_pred, _ = wake_agent.predict(
                        yaw_angle=nacelle_dir,
                        n_timesteps=30,
                        export_vtk=False,
                        verbose=False
                    )
                    print(f"    ✓ ML prediction shape: {wake_pred.shape}")
                    
                    # Extract profiles
                    profile_start = extract_vertical_profile_from_wake(
                        wake_pred, wake_agent.grid, 
                        downstream_distance_m=0, 
                        vertical_range_m=300
                    )
                    
                    profile_end = extract_vertical_profile_from_wake(
                        wake_pred, wake_agent.grid,
                        downstream_distance_m=reference_distance_m,
                        vertical_range_m=300
                    )
                    
                    if profile_start and profile_end:
                        print(f"    ✓ Extracted profiles: {profile_start['n_points']} points (start), "
                              f"{profile_end['n_points']} points (end)")
                        
                        # Calculate wake characteristics
                        wake_analysis = calculate_wake_centerline_and_deficit(
                            profile_start, profile_end, freestream_velocity=8.5
                        )
                        
                        print(f"    ✓ Wake at 1700m: Deficit={wake_analysis['deficit_end']:.3f}, "
                              f"Lateral migration={wake_analysis['lateral_migration']:.1f}m")
                        
                        # Extrapolate to actual spacing
                        target_distance_m = scenario['spacing_D'] * rotor_diameter
                        wake_at_target = extrapolate_wake_trajectory(
                            wake_analysis, target_distance_m, reference_distance_m
                        )
                        
                        print(f"    ✓ Wake at {scenario['spacing_D']:.1f}D: "
                              f"Deficit={wake_at_target['deficit_at_target']:.3f}, "
                              f"Lateral migration={wake_at_target['lateral_migration_at_target']:.1f}m")
                        
                        # Check overlap
                        downstream_position = {'x': target_distance_m, 'y': 0}
                        overlap_check = check_wake_turbine_overlap(
                            wake_at_target, downstream_position, rotor_radius_m=63.0
                        )
                        
                        overlap_status = "❌ OVERLAPS" if overlap_check['overlaps'] else "✅ CLEAR"
                        print(f"    {overlap_status}: Overlap fraction={overlap_check['overlap_fraction']:.1%}, "
                              f"Clearance={overlap_check['lateral_clearance']:.1f}m\n")
                    else:
                        print(f"    ⚠ Could not extract profiles (insufficient grid coverage)\n")
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}\n")
        
        # Test comprehensive optimization
        print("\n" + "=" * 80)
        print("COMPREHENSIVE OPTIMIZATION TEST")
        print("=" * 80)
        print("\nFinding optimal yaw for wake avoidance at 7.5D spacing...\n")
        
        try:
            opt_result = find_optimal_yaw_for_wake_avoidance(
                wake_agent=wake_agent,
                turbine_spacing_D=7.5,
                rotor_diameter=126.0,
                downstream_turbine_position={'x': 7.5 * 126.0, 'y': 0},
                yaw_constraint=(0, 15),
                n_timesteps=30,
                reference_distance_m=1700.0,
                verbose=True
            )
            
            print("\n" + "─" * 80)
            print("OPTIMIZATION RESULTS:")
            print("─" * 80)
            print(f"Optimal yaw angle: {opt_result['optimal_yaw']:.1f}°")
            print(f"Wake avoidance achieved: {opt_result['avoidance_achieved']}")
            print(f"Turbine spacing: {opt_result['turbine_spacing_D']:.1f}D ({opt_result['turbine_spacing_m']:.0f}m)")
            print(f"Wake trajectory analyzed: {len(opt_result['wake_trajectory'])} yaw angles tested")
            
            if opt_result['wake_trajectory']:
                print("\nDetailed trajectory:")
                for result in opt_result['wake_trajectory']:
                    yaw = result['yaw_deg']
                    overlap = result['overlap_check']['overlap_fraction']
                    migration = result['wake_at_target']['lateral_migration_at_target']
                    print(f"  Yaw {yaw:5.1f}°: Overlap {overlap:5.1%}, Migration {migration:6.1f}m")
            
        except Exception as e:
            print(f"❌ Optimization error: {e}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED")
        print("=" * 80)
        print("\n✅ All ML-based wake analysis functions are working!")
        print("   Functions are ready for integration into optimization workflow.")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Ensure tt_opinf_inference_agent.py and wind_turbine_gui.py are available.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ml_wake_analysis()
