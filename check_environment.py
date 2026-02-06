"""
Environment Diagnostic Script for Wind Turbine GUI

Run this to check which Python environment is being used and what packages are installed.
"""

import sys
import subprocess

print("=" * 60)
print("Python Environment Diagnostic")
print("=" * 60)

print(f"\n1. Python Executable:")
print(f"   {sys.executable}")

print(f"\n2. Python Version:")
print(f"   {sys.version}")

print(f"\n3. Checking for required packages:")
packages = ['streamlit', 'geopy', 'numpy', 'pandas', 'torch', 'matplotlib', 'pydeck', 'openai']

for package in packages:
    try:
        __import__(package)
        # Get version if available
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {package:15s} - version {version}")
        except:
            print(f"   ✅ {package:15s} - installed")
    except ImportError:
        print(f"   ❌ {package:15s} - NOT INSTALLED")

print(f"\n4. System Path:")
for i, path in enumerate(sys.path[:5]):
    print(f"   [{i}] {path}")

print("\n" + "=" * 60)
print("To install missing packages, run:")
print("pip install geopy streamlit numpy pandas torch matplotlib pydeck openai")
print("=" * 60)

# Check if geopy specifically works
print("\n5. Testing geopy import:")
try:
    from geopy.distance import geodesic
    print("   ✅ geopy.distance.geodesic imported successfully")
    
    # Test calculation
    dist = geodesic((63.4305, 10.3951), (63.4405, 10.4051)).meters
    print(f"   ✅ Distance calculation test: {dist:.2f} meters")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
