#!/usr/bin/env python3
"""
Test script for Agent 2C: Turbine Pair Selector functionality.
This script validates the turbine pair analysis function.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_turbine_pair_analysis():
    """Test the turbine pair analysis functionality."""
    
    # Import the function from the main GUI module
    try:
        from wind_turbine_gui import get_turbine_pair_recommendations
        print("✅ Successfully imported get_turbine_pair_recommendations")
    except ImportError as e:
        print(f"❌ Failed to import function: {e}")
        return False
    
    # Sample turbine locations (using Bessaker Wind Farm data)
    sample_turbines = [
        {'turbine_id': 1, 'latitude': 64.230229, 'longitude': 10.380853, 'type': 'known'},
        {'turbine_id': 2, 'latitude': 64.230062, 'longitude': 10.376347, 'type': 'known'},
        {'turbine_id': 3, 'latitude': 64.230828, 'longitude': 10.371845, 'type': 'known'},
        {'turbine_id': 4, 'latitude': 64.232406, 'longitude': 10.380896, 'type': 'known'},
        {'turbine_id': 5, 'latitude': 64.235126, 'longitude': 10.371195, 'type': 'known'},
    ]
    
    # Sample wind conditions
    wind_speed = 10.5
    wind_direction = 270  # West wind
    
    print(f"🌬️  Testing with {len(sample_turbines)} turbines")
    print(f"🎯 Wind conditions: {wind_speed} m/s from {wind_direction}°")
    
    # Test the function
    try:
        result = get_turbine_pair_recommendations(
            turbine_locations=sample_turbines,
            wind_speed=wind_speed,
            wind_dir=wind_direction,
            provider='NTNU',
            model='moonshotai/Kimi-K2.5'
        )
        
        print(f"🔍 Analysis status: {result['status']}")
        print(f"🤖 Agent: {result['agent']}")
        
        if result['status'] == 'success':
            print(f"✅ Analysis completed successfully!")
            print(f"📊 Total turbines: {result['total_turbines']}")
            print(f"🎯 Critical pairs found: {len(result['turbine_pairs'])}")
            print(f"📝 Analysis summary: {result.get('analysis_summary', 'N/A')[:100]}...")
            
            if result['turbine_pairs']:
                print("\n🔗 Critical turbine pairs:")
                for i, pair in enumerate(result['turbine_pairs'][:3]):  # Show first 3
                    upstream = pair.get('upstream_turbine', 'N/A')
                    downstream = pair.get('downstream_turbine', 'N/A')
                    strength = pair.get('wake_strength', 'unknown')
                    print(f"   {i+1}. T{upstream} → T{downstream} ({strength} wake)")
            
            return True
            
        elif result['status'] == 'error':
            print(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
            return False
            
        else:
            print(f"⚠️  Analysis partially completed: {result['status']}")
            return True
            
    except Exception as e:
        print(f"❌ Exception during analysis: {e}")
        return False

def test_turbine_database():
    """Test the known turbine locations database."""
    
    try:
        from wind_turbine_gui import get_known_turbine_locations
        print("✅ Successfully imported get_known_turbine_locations")
    except ImportError as e:
        print(f"❌ Failed to import function: {e}")
        return False
    
    # Test known wind farms
    test_farms = ["Bessaker Wind Farm", "Smøla Wind Farm", "Unknown Farm"]
    
    for farm in test_farms:
        locations = get_known_turbine_locations(farm)
        if locations:
            print(f"✅ {farm}: Found {len(locations)} turbine locations")
        else:
            print(f"ℹ️  {farm}: No turbine locations found")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Agent 2C: Turbine Pair Selector")
    print("=" * 50)
    
    # Test 1: Database functionality
    print("\n📊 Test 1: Turbine Database")
    test_turbine_database()
    
    # Test 2: Turbine pair analysis (this will likely fail without LLM connection)
    print("\n🎯 Test 2: Turbine Pair Analysis")
    print("Note: This test may fail if LLM is not accessible")
    test_turbine_pair_analysis()
    
    print("\n✅ Testing completed!")
    print("\n💡 To run full analysis, start the Streamlit GUI:")
    print("   streamlit run wind_turbine_gui.py")