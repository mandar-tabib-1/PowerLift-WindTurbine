"""
Quick validation test for RUL code dependencies and basic functionality
"""
import sys
print("Testing RUL code dependencies...\n")

# Test all imports
try:
    import numpy as np
    print("✓ numpy imported successfully")
except Exception as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except Exception as e:
    print(f"✗ pandas import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except Exception as e:
    print(f"✗ matplotlib import failed: {e}")
    sys.exit(1)

try:
    import seaborn as sns
    print("✓ seaborn imported successfully")
except Exception as e:
    print(f"✗ seaborn import failed: {e}")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print("✓ scikit-learn imported successfully")
except Exception as e:
    print(f"✗ scikit-learn import failed: {e}")
    sys.exit(1)

# Test basic functionality
print("\nTesting basic functionality...")
try:
    # Create simple test data
    test_data = np.random.rand(100, 5)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(test_data)
    print(f"✓ Data scaling works (shape: {scaled_data.shape})")
    
    # Test GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    labels = gmm.fit_predict(scaled_data)
    print(f"✓ GMM clustering works (found {len(np.unique(labels))} clusters)")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED - RUL CODE DEPENDENCIES ARE WORKING!")
    print("="*60)
    
except Exception as e:
    print(f"✗ Functionality test failed: {e}")
    sys.exit(1)
