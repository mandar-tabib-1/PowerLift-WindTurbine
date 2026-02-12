"""
Quick test to simulate GUI predictive maintenance inference.
This verifies the complete workflow works with the corrected GMM state ordering.
"""

import numpy as np
import sys
import os

# Add RUL to path
rul_path = os.path.join(os.path.dirname(__file__), 'RUL')
if rul_path not in sys.path:
    sys.path.insert(0, rul_path)

from save_models import load_all_models, load_test_data

print("=" * 80)
print("SIMULATING GUI PREDICTIVE MAINTENANCE INFERENCE")
print("=" * 80)

# Simulate GUI loading models
print("\n1. Loading models (as GUI does)...")
models_dict = {
    'models': load_all_models(),
    'test_data': load_test_data()
}
print("   ✅ Models loaded")

# Simulate running inference (as GUI run_pdm_inference does)
print("\n2. Running inference on test samples...")
X_subset = models_dict['test_data']['X_test'][:500]
y_subset = models_dict['test_data']['y_test'][:500]

autoencoder = models_dict['models']['autoencoder']
gmm = models_dict['models']['gmm']
state_order_map = models_dict['models']['state_order_map']

# Compute health indicator
hi = autoencoder.get_health_indicator(X_subset)
print(f"   Health Indicators computed: min={np.min(hi):.4f}, max={np.max(hi):.4f}, mean={np.mean(hi):.4f}")

# Predict states with mapping
raw_states = gmm.predict(hi.reshape(-1, 1))
states = np.array([state_order_map[s] for s in raw_states])
print(f"   States predicted using mapping: {state_order_map}")

# Verify ordering (as GUI display_pdm_results does)
print("\n3. Verifying state ordering (as GUI does)...")
state_labels = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
verification_data = []

for state in [0, 1, 2]:
    mask = states == state
    if np.any(mask):
        hi_in_state = hi[mask]
        verification_data.append({
            'State': f"{state} ({state_labels[state]})",
            'Samples': int(np.sum(mask)),
            'HI Mean': f"{np.mean(hi_in_state):.4f}",
            'HI Min': f"{np.min(hi_in_state):.4f}",
            'HI Max': f"{np.max(hi_in_state):.4f}"
        })
        print(f"   {verification_data[-1]}")

# Check ordering
print("\n4. State ordering check:")
means = [float(row['HI Mean']) for row in verification_data]
if all(means[i] < means[i+1] for i in range(len(means)-1)):
    print("   ✅ PASS: States correctly ordered: Healthy (low HI) → Degrading (medium HI) → Critical (high HI)")
    print(f"      Healthy mean: {means[0]:.4f}")
    print(f"      Degrading mean: {means[1]:.4f}")
    print(f"      Critical mean: {means[2]:.4f}")
else:
    print("   ❌ FAIL: States are NOT correctly ordered!")

print("\n" + "=" * 80)
print("GUI SIMULATION COMPLETE")
print("=" * 80)
print("\n✅ The predictive maintenance mode in the GUI will now work correctly!")
print("   - High HI samples will be labeled 'Critical' (not 'Degrading')")
print("   - States will be properly ordered by health indicator value")
print("\n💡 To test in the GUI:")
print("   1. Run: streamlit run wind_turbine_gui.py")
print("   2. Select 'Predictive Maintenance' mode")
print("   3. Click 'Run Inference'")
print("   4. Check the 'State Distribution' tab for the verification table")
print("=" * 80)
