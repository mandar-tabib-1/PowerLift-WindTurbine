# GMM State Ordering Fix - Summary

## Problem Identified

The GMM (Gaussian Mixture Model) was assigning health states incorrectly:
- **High HI samples** (indicating degradation) were labeled as "Degrading" instead of "Critical"
- **Root Cause**: The saved model had an identity mapping `{0:0, 1:1, 2:2}` instead of the correct mapping based on cluster centers

## Root Cause Analysis

GMM clustering is **unsupervised** and doesn't know which cluster corresponds to which state. The clusters need to be **ordered by mean health indicator (HI) value**:

1. **Cluster with lowest mean HI** → State 0 (Healthy)
2. **Cluster with medium mean HI** → State 1 (Degrading)
3. **Cluster with highest mean HI** → State 2 (Critical)

### What Was Wrong

The saved model had:
```python
GMM cluster centers: [0.2054, 0.6981, 0.3505]
state_order_map: {0: 0, 1: 1, 2: 2}  # WRONG - identity mapping
```

This caused:
- Cluster 1 (HI=0.6981, **highest**) → mapped to State 1 (Degrading) ❌ Should be State 2!
- Cluster 2 (HI=0.3505, medium) → mapped to State 2 (Critical) ❌ Should be State 1!

## Solution Implemented

### 1. Fixed Training Code (`wind_turbine_pm_fuhrlander.py`)

Modified `classify_health_states()` to return the `state_order_map`:

```python
def classify_health_states(health_indicator, n_states=3):
    """GMM-based unsupervised health state classification."""
    HI = health_indicator.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_states, random_state=42, n_init=10)
    gmm.fit(HI)
    states = gmm.predict(HI)
    
    # Create mapping from raw GMM cluster IDs to ordered state IDs
    state_means = np.array([HI[states == i].mean() for i in range(n_states)])
    state_order = np.argsort(state_means)
    state_order_map = {int(old): int(new) for new, old in enumerate(state_order)}
    
    # Apply mapping to get sorted states
    sorted_states = np.array([state_order_map[s] for s in states])
    
    print(f"  GMM Cluster Centers (raw): {gmm.means_.flatten()}")
    print(f"  Mean HI per raw cluster: {state_means}")
    print(f"  State ordering (raw→sorted): {state_order_map}")
    
    return sorted_states, gmm, probs, state_order_map  # Now returns mapping!
```

### 2. Fixed Main Training Loop

Uses the returned mapping instead of recreating it incorrectly:

```python
# Get sorted states and the ordering map from training data
train_states, gmm, _, state_order_map = classify_health_states(train_hi)

# Apply the same mapping to test data
test_raw_states = gmm.predict(test_hi.reshape(-1, 1))
test_states = np.array([state_order_map[s] for s in test_raw_states])
```

### 3. Fixed Existing Saved Model (`fix_state_order_map.py`)

Created a script to correct the saved model without retraining:

```python
# Compute correct mapping based on cluster centers
cluster_means = gmm.means_.flatten()  # [0.2054, 0.6981, 0.3505]
state_order = np.argsort(cluster_means)  # [0, 2, 1]
correct_map = {int(old): int(new) for new, old in enumerate(state_order)}
# Result: {0: 0, 2: 1, 1: 2}
```

### 4. Added Diagnostics to GUI (`wind_turbine_gui.py`)

Added verification in the inference display:

```python
st.markdown("**🔍 GMM State Ordering Verification:**")
for state in [0, 1, 2]:
    mask = pdm_results['health_states'] == state
    if np.any(mask):
        hi_in_state = pdm_results['health_indicator'][mask]
        st.write(f"State {state}: Samples={np.sum(mask)}, HI Mean={np.mean(hi_in_state):.4f}")

# Check ordering
if all(means[i] < means[i+1] for i in range(len(means)-1)):
    st.success("✅ States correctly ordered")
else:
    st.error("❌ States NOT correctly ordered!")
```

## Correct Mapping Now

```python
GMM cluster centers: [0.2054, 0.6981, 0.3505]
state_order_map: {0: 0, 2: 1, 1: 2}  # ✓ CORRECT
```

This correctly maps:
- **Raw cluster 0** (HI=0.2054, lowest) → **State 0 (Healthy)** ✅
- **Raw cluster 2** (HI=0.3505, medium) → **State 1 (Degrading)** ✅
- **Raw cluster 1** (HI=0.6981, highest) → **State 2 (Critical)** ✅

## Verification Test Results

```
State 0 (Healthy):   160 samples | HI range=[0.1436, 0.3053] | mean=0.2338
State 1 (Degrading):  39 samples | HI range=[0.3078, 0.4516] | mean=0.3547
State 2 (Critical):    1 sample  | HI range=[0.6595, 0.6595] | mean=0.6595

✅ PASS: States are correctly ordered!
Mean HI progression: State 0 (0.2338) < State 1 (0.3547) < State 2 (0.6595)
```

## Files Modified

1. **Training**: `RUL/wind_turbine_pm_fuhrlander.py`
   - Modified `classify_health_states()` to return mapping
   - Updated main training loop to use returned mapping

2. **Inference**: `wind_turbine_gui.py`
   - Added diagnostic verification in state distribution tab
   - Added console logging of state ordering

3. **Model Fix**: `RUL/fix_state_order_map.py` (new)
   - Script to correct existing saved models

4. **Testing**: `RUL/test_state_ordering.py` (new)
   - Verification script for state ordering

## Usage

### For New Training
Just run the training script - it will automatically create the correct mapping:
```bash
cd RUL
python wind_turbine_pm_fuhrlander.py
```

### For Existing Saved Models
Run the fix script to correct the mapping:
```bash
cd RUL
python fix_state_order_map.py
```

### Verification
Test the corrected mapping:
```bash
cd RUL
python test_state_ordering.py
```

### In GUI
The predictive maintenance mode will now:
1. Display state ordering verification table
2. Show ✅ or ❌ indicator for correct ordering
3. Print diagnostic info to console during inference

## Expected Behavior

After the fix, the GUI's predictive maintenance mode will correctly show:
- **State 0 (Healthy)**: Samples with **low HI** values (< 0.3)
- **State 1 (Degrading)**: Samples with **medium HI** values (0.3 - 0.5)
- **State 2 (Critical)**: Samples with **high HI** values (> 0.5)

High HI samples will now correctly receive "Critical" labels instead of "Degrading".

---

**Date**: February 12, 2026  
**Fixed By**: GitHub Copilot (Claude Sonnet 4.5)  
**Status**: ✅ Verified and Working
