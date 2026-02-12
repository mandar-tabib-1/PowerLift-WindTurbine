"""Test corrected GMM state ordering with inference."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from save_models import load_all_models, load_test_data

print("=" * 80)
print("TESTING CORRECTED GMM STATE ORDERING")
print("=" * 80)

# Load models and test data
models = load_all_models()
test = load_test_data()

print(f"\n1. Corrected state_order_map: {models['state_order_map']}")

# Run inference on a sample
X_sample = test['X_test'][:200]
print(f"\n2. Running inference on {len(X_sample)} test samples...")

# Compute health indicators
hi = models['autoencoder'].get_health_indicator(X_sample)
print(f"   Health Indicators: min={np.min(hi):.4f}, max={np.max(hi):.4f}, mean={np.mean(hi):.4f}")

# Predict states using GMM and apply mapping
raw_states = models['gmm'].predict(hi.reshape(-1, 1))
states = np.array([models['state_order_map'][s] for s in raw_states])

# Analyze state distribution
print(f"\n3. State distribution and HI ranges:")
state_labels = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
state_hi_means = []

for state in [0, 1, 2]:
    mask = states == state
    count = np.sum(mask)
    if count > 0:
        hi_in_state = hi[mask]
        mean_hi = np.mean(hi_in_state)
        state_hi_means.append(mean_hi)
        print(f"   State {state} ({state_labels[state]:9s}): {count:3d} samples | "
              f"HI range=[{np.min(hi_in_state):.4f}, {np.max(hi_in_state):.4f}] | "
              f"mean={mean_hi:.4f}")
    else:
        state_hi_means.append(np.nan)

# Verify ordering
print(f"\n4. State ordering verification:")
if len([m for m in state_hi_means if not np.isnan(m)]) >= 2:
    valid_means = [(i, m) for i, m in enumerate(state_hi_means) if not np.isnan(m)]
    is_ordered = all(valid_means[i][1] < valid_means[i+1][1] for i in range(len(valid_means)-1))
    
    if is_ordered:
        print("   ✅ PASS: States are correctly ordered!")
        print(f"   Mean HI progression: ", end="")
        print(" < ".join([f"State {i} ({state_hi_means[i]:.4f})" for i, m in valid_means]))
    else:
        print("   ❌ FAIL: States are NOT correctly ordered!")
        print(f"   Mean HI values: {[f'{m:.4f}' if not np.isnan(m) else 'N/A' for m in state_hi_means]}")
else:
    print("   ⚠️ WARNING: Not enough states to verify ordering")

print("\n" + "=" * 80)
print("Now the GMM will correctly assign:")
print("  - Low HI samples → State 0 (Healthy)")
print("  - Medium HI samples → State 1 (Degrading)")  
print("  - High HI samples → State 2 (Critical)")
print("=" * 80)
