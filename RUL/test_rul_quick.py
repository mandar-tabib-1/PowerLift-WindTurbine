"""
Quick Test of Wind Turbine RUL Prediction (Reduced Dataset)
"""
import sys
import os

# Temporarily patch the main script to use fewer samples
import wind_turbine_pm_sklearn as wtpm

# Override the data generation to use fewer turbines and hours
original_generate = wtpm.generate_wind_turbine_scada_data

def quick_generate(n_turbines=2, hours_per_turbine=2000, **kwargs):
    """Generate minimal dataset for quick testing"""
    return original_generate(n_turbines=n_turbines, 
                            hours_per_turbine=hours_per_turbine,
                            **kwargs)

# Monkey patch
wtpm.generate_wind_turbine_scada_data = quick_generate

print("="*80)
print("QUICK RUL TEST - Reduced Dataset (2 turbines, 2000 hours)")
print("="*80)
print()

# Run main function
results = wtpm.main()

print("\n" + "="*80)
print("QUICK TEST COMPLETE!")
print("="*80)
print(f"\nKey Results:")
print(f"  - Data points processed: {len(results['health_indicator']):,}")
print(f"  - Health states found: {len(set(results['states']))}")
print(f"  - LSTM models trained: {len(results['lstm_models'])}")
print(f"  - RUL predictions: {sum(~results['rul_predictions'].isnan())}")
print(f"  - Average RUL: {results['rul_predictions'].mean():.1f} hours")
