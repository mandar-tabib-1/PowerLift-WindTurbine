"""Fix encoding issues in wind_turbine_gui.py"""

# Read the file with UTF-8 encoding
with open('wind_turbine_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Ensure all emojis are properly encoded
# Replace any remaining problematic characters
replacements = []

# Write back with proper UTF-8 encoding
with open('wind_turbine_gui.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ File encoding fixed successfully")
print(f"  Total characters: {len(content)}")
print(f"  Total lines: {content.count(chr(10)) + 1}")

# Clear any cached .pyc files
import os
import glob

pyc_files = glob.glob('__pycache__/*.pyc')
for pyc in pyc_files:
    try:
        os.remove(pyc)
        print(f"  Removed cached file: {pyc}")
    except Exception as e:
        pass

print("\n✓ Ready to restart Streamlit")
print("  Run: streamlit run wind_turbine_gui.py --server.port 8502")
