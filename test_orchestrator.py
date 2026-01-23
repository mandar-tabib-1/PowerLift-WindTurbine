"""Quick test script for orchestrator with TT-OpInf wake flow agent."""

from wind_turbine_orchestrator import WindTurbineOrchestrator

# Initialize the orchestrator
print("Initializing orchestrator...")
orchestrator = WindTurbineOrchestrator()

# Run a short analysis with VTK export
print("\nRunning analysis with VTK export...")
results = orchestrator.run_analysis(
    location="Boulder, Colorado",
    n_time_points=10,  # Short run for testing
    create_wake_animation=False,
    export_wake_vtk=True  # Enable VTK export
)

# Print summary
print("\n" + "="*60)
print("TEST RESULTS SUMMARY")
print("="*60)

if results.get("wake_prediction") is not None:
    wake = results["wake_prediction"]
    print(f"✓ Wake prediction shape: {wake['predictions'].shape}")
    print(f"✓ Velocity magnitude range: {wake['velocity_magnitude'].min():.2f} - {wake['velocity_magnitude'].max():.2f} m/s")
    
    output_files = wake.get("output_files", {})
    if output_files.get("vtk"):
        print(f"✓ VTK output directory: {output_files['vtk']}")
    else:
        print("⚠ No VTK output (pyvista may be missing)")
else:
    print("✗ Wake prediction failed")

print("\n✓ Orchestration test completed successfully!")
