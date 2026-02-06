"""
Test script for Agent 2D column mapping
"""
import pandas as pd
import sys
from pathlib import Path

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# Test data with 'latitude' and 'longitude' columns (as used in your code)
test_turbines = [
    {'turbine_id': 1, 'latitude': 64.230229, 'longitude': 10.380853, 'type': 'known'},
    {'turbine_id': 2, 'latitude': 64.230062, 'longitude': 10.376347, 'type': 'known'},
    {'turbine_id': 3, 'latitude': 64.230828, 'longitude': 10.371845, 'type': 'known'},
    {'turbine_id': 4, 'latitude': 64.232406, 'longitude': 10.380896, 'type': 'known'},
    {'turbine_id': 5, 'latitude': 64.235126, 'longitude': 10.371195, 'type': 'known'},
]

print("=" * 60)
print("Testing Agent 2D Column Mapping")
print("=" * 60)

print("\n1. Original Data Structure:")
df = pd.DataFrame(test_turbines)
print(df.head())
print(f"\nColumns: {list(df.columns)}")

print("\n2. Testing Column Normalization:")
# This is the same logic used in Agent 2D
if 'latitude' in df.columns and 'lat' not in df.columns:
    df['lat'] = df['latitude']
    print("✅ Created 'lat' column from 'latitude'")

if 'longitude' in df.columns and 'lon' not in df.columns:
    df['lon'] = df['longitude']
    print("✅ Created 'lon' column from 'longitude'")

print(f"\nFinal columns: {list(df.columns)}")
print(f"\nCan access 'lat': {'lat' in df.columns}")
print(f"Can access 'lon': {'lon' in df.columns}")

print("\n3. Testing Agent 2D Function:")
try:
    from wind_turbine_gui import get_wake_influenced_turbine_pairs
    
    result = get_wake_influenced_turbine_pairs(
        turbine_locations=test_turbines,
        wind_speed=8.5,
        wind_direction=270.0,
        rotor_diameter=126.0,
        hub_height=90.0,
        wake_expansion_factor=0.1,
        min_influence_threshold=0.05
    )
    
    print(f"\n✅ Agent 2D executed successfully!")
    print(f"   Status: {result.get('status', 'unknown')}")
    print(f"   Agent Type: {result.get('agent_type', 'unknown')}")
    print(f"   Total Pairs Found: {result.get('total_pairs_found', 0)}")
    print(f"   Method: {result.get('method', 'unknown')}")
    
    if result.get('turbine_pairs'):
        print(f"\n   Sample Turbine Pair:")
        pair = result['turbine_pairs'][0]
        print(f"   - Upstream: T{pair.get('upstream_id', 'N/A')}")
        print(f"   - Downstream: T{pair.get('downstream_id', 'N/A')}")
        print(f"   - Distance: {pair.get('distance_rotor_diameters', 'N/A'):.2f}D")
    
except Exception as e:
    print(f"\n❌ Error running Agent 2D: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
