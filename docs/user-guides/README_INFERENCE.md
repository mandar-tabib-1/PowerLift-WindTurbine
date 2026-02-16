# Fuhrlander FL2500 Predictive Maintenance - Inference Setup Guide

## Overview

This folder contains the complete trained ML pipeline for predictive maintenance
of Fuhrlander FL2500 (2.5MW) wind turbines. The system predicts:

- **Health Indicator (HI)**: Autoencoder reconstruction error as a health score
- **Health State**: GMM classification (Healthy / Degrading / Critical)
- **Fault Probability**: Binary (GradientBoosting) and 3-class (RandomForest)
- **Remaining Useful Life (RUL)**: State-specific trend prediction (hours to failure)
- **Feature Explanations**: SHAP-based per-sample and global explanations

---

## Saved Models

### File Locations

```
RUL/saved_models/
    fuhrlander_fl2500_pm_models.joblib   # All model artifacts
    test_data.npz                         # Test data arrays (X_test, y_test)
    test_df.parquet                       # Full test DataFrame with metadata
    metadata.json                         # Human-readable manifest
```

### Model Inventory

| Model | Type | Purpose | Input Shape | Output Shape |
|-------|------|---------|-------------|--------------|
| **SimpleAutoencoder** | Custom numpy (27->64->32->8->32->64->27) | Health Indicator | `(N, 27)` float64 | `(N,)` float64 HI values |
| **GaussianMixture** | sklearn (3 components) | Health state classification | `(N, 1)` HI values | `(N,)` states {0=Healthy, 1=Degrading, 2=Critical} |
| **GradientBoostingClassifier** | sklearn (200 trees, depth 5) | Binary fault prediction | `(N, 27)` float64 | `(N,)` {0=Healthy, 1=Anomalous} + `(N,2)` probabilities |
| **RandomForestClassifier** | sklearn (300 trees, depth 10) | 3-class fault prediction | `(N, 27)` float64 | `(N,)` {0=Healthy, 1=Pre-Fault, 2=Fault} + `(N,3)` probabilities |
| **SimpleLSTM** (per state) | Custom numpy (seq_length=24) | RUL prediction | `(24,)` HI sequence | `float` next HI value |
| **failure_threshold** | float (0.3916) | Failure cutoff | - | 90th percentile of training HI |
| **state_order_map** | dict | GMM state reordering | raw GMM label | ordered label (0=healthy, 2=critical) |

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Training Turbines | 80, 81, 82 (53,810 hourly samples) |
| Test Turbines | 83, 84 (35,411 hourly samples) |
| Feature Dimension | 27 (18 raw + 9 engineered) |
| Autoencoder Encoding Dim | 8 |
| GMM Components | 3 |
| Pre-Fault Window | 48 hours |
| LSTM Sequence Length | 24 hours |
| Data Frequency | 1 hour (aggregated from 5-minute SCADA) |

---

## Input Features (27 total)

### 18 Raw SCADA Features

| # | Feature Name | System | Description |
|---|---|---|---|
| 1 | `wtrm_avg_TrmTmp_Gbx` | Transmission | Gearbox temperature (avg) |
| 2 | `wtrm_avg_TrmTmp_GbxOil` | Transmission | Gearbox oil sump temperature |
| 3 | `wtrm_avg_TrmTmp_GbxBrg151` | Transmission | Gearbox bearing 151 temperature |
| 4 | `wtrm_avg_TrmTmp_GbxBrg152` | Transmission | Gearbox bearing 152 temperature |
| 5 | `wtrm_avg_TrmTmp_GbxBrg450` | Transmission | Gearbox bearing 450 temperature |
| 6 | `wtrm_avg_TrmTmp_GnBrgDE` | Transmission | Generator bearing (drive end) temp |
| 7 | `wtrm_avg_TrmTmp_GnBrgNDE` | Transmission | Generator bearing (non-drive end) temp |
| 8 | `wtrm_avg_Gbx_OilPres` | Transmission | Gearbox oil pressure |
| 9 | `wtrm_avg_Brg_OilPres` | Transmission | Main bearing oil pressure |
| 10 | `wgen_avg_GnTmp_phsA` | Generator | Generator winding temperature (phase A) |
| 11 | `wgen_avg_Spd` | Generator | Generator speed (RPM) |
| 12 | `wnac_avg_WSpd1` | Nacelle | Wind speed (m/s) |
| 13 | `wnac_avg_NacTmp` | Nacelle | Nacelle temperature |
| 14 | `wgdc_avg_TriGri_PwrAt` | Grid | Active power output (kW) |
| 15 | `wgdc_avg_TriGri_A` | Grid | Grid current (A) |
| 16 | `wtrm_sdv_TrmTmp_Gbx` | Transmission | Gearbox temp std deviation (variability) |
| 17 | `wtrm_sdv_TrmTmp_GbxOil` | Transmission | Oil temp std deviation |
| 18 | `wgen_sdv_Spd` | Generator | Generator speed std deviation |

### 9 Engineered Features

| # | Feature Name | Formula / Description |
|---|---|---|
| 19 | `thermal_stress_idx` | Weighted sum of 5 gearbox temps / 100 |
| 20 | `bearing_stress_idx` | Weighted sum of 5 bearing temps / 100 |
| 21 | `power_efficiency` | Active_Power / (0.5 * WindSpeed^3 + 50) |
| 22 | `gbx_temp_trend` | 24h rolling mean diff of gearbox temp |
| 23 | `oil_pressure_ratio` | Gbx_OilPres / (Brg_OilPres + 0.01) |
| 24 | `bearing_temp_spread` | max(BearingTemps) - min(BearingTemps) |
| 25 | `gen_thermal_load` | GenTemp - NacelleTemp |
| 26 | `oil_temp_trend` | 24h rolling mean of oil temp |
| 27 | `variability_trend` | 24h rolling mean of gearbox temp std dev |

---

## Step-by-Step Inference Pipeline

### Step 1: Load Models

```python
import sys
sys.path.insert(0, r'C:\Users\mandart\A_MANDAR_DOCUMENTS\HAM_Wind_Energy\ResultMLYaw\PowerLift\RUL')

from save_models import load_all_models, load_test_data

# Load all trained models
models = load_all_models()

autoencoder     = models['autoencoder']         # SimpleAutoencoder
gmm             = models['gmm']                 # GaussianMixture
binary_clf      = models['binary_clf']          # GradientBoostingClassifier
multi_clf       = models['multi_clf']           # RandomForestClassifier
lstm_models     = models['lstm_models']         # dict {state: SimpleLSTM}
failure_threshold = models['failure_threshold']  # float (0.3916)
state_order_map = models['state_order_map']     # dict {raw_gmm_idx: ordered_idx}
feature_names   = models['feature_names']       # list of 27 strings
```

### Step 2: Prepare Input Data

**Option A: Load saved test data (for demo/inference)**

```python
test = load_test_data()
X_test  = test['X_test']    # shape (35411, 27)
y_test  = test['y_test']    # shape (35411,)
test_df = test['test_df']   # DataFrame with date_time, turbine_id, features, labels
```

**Option B: Load new SCADA data**

```python
import numpy as np
import pandas as pd
from wind_turbine_pm_fuhrlander import engineer_features, ALL_FEATURES

# Data must contain: date_time, turbine_id, and all 18 RAW_FEATURES columns
# Data should be at hourly frequency (aggregate 5-min data if needed)
new_df = pd.read_csv('new_scada_data.csv', parse_dates=['date_time'])

# Engineer the 9 derived features
eng_df = engineer_features(new_df, window_size=24)

# Extract feature matrix
X_new = eng_df[ALL_FEATURES].values
X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
```

### Step 3: Run Inference (all 4 models)

```python
import numpy as np
from wind_turbine_pm_fuhrlander import predict_rul

# 3a. Autoencoder -> Health Indicator
hi = autoencoder.get_health_indicator(X_test)          # (N,) float

# 3b. GMM -> Health States
raw_states = gmm.predict(hi.reshape(-1, 1))
states = np.array([state_order_map[s] for s in raw_states])  # (N,) {0,1,2}

# 3c. Binary fault prediction
binary_pred = binary_clf.predict(X_test)               # (N,) {0,1}
binary_prob = binary_clf.predict_proba(X_test)          # (N,2)

# 3d. Multi-class fault prediction
multi_pred = multi_clf.predict(X_test)                  # (N,) {0,1,2}
multi_prob = multi_clf.predict_proba(X_test)            # (N,3)

# 3e. RUL prediction
rul = predict_rul(hi, states, lstm_models,
                  failure_threshold, seq_length=24)      # (N,) float, may have NaN
```

### Step 4: Generate Visualizations

```python
from inference_viz import generate_inference_report

# Filter for a single turbine
mask = test_df['turbine_id'] == 83
timestamps = test_df.loc[mask, 'date_time'].values

report = generate_inference_report(
    turbine_id=83,
    timestamps=timestamps,
    health_indicator=hi[mask.values],
    gmm_states=states[mask.values],
    fault_probabilities=binary_prob[mask.values, 1],
    multi_class_probs=multi_prob[mask.values],
    rul_predictions=rul[mask.values],
    failure_threshold=failure_threshold,
    feature_names=feature_names,
    true_labels=y_test[mask.values],          # optional (for validation)
    output_dir='./inference_output/',
)

print(report['text_summary'])
# Plots saved to:
#   inference_output/inference_turbine_83_dashboard.png
#   inference_output/inference_turbine_83_confusion.png
#   inference_output/inference_turbine_83_summary.png
#   inference_output/inference_turbine_83_rul_trend.png
```

### Step 5: SHAP Explanation (per-sample)

```python
from shap_explainer import SHAPExplainer

explainer = SHAPExplainer(
    binary_clf=binary_clf,
    multi_clf=multi_clf,
    feature_names=feature_names,
    output_dir='./inference_output/'
)

# Explain a specific sample
sample_idx = -1  # latest observation
result = explainer.explain_single_sample(
    X_test[mask.values][sample_idx:],
    sample_id=f"turbine_83_latest",
    save_plot=True,
    verbose=True
)

# Structured results
print(result['explanation_text'])                        # Human-readable
print(f"Binary: P(anomalous) = {result['binary_probability'][1]:.3f}")
print(f"Multi:  P(fault)     = {result['multi_probability'][2]:.3f}")
print(result['top_features_pushing_fault'])              # [(feat, shap, val), ...]
print(result['top_features_pushing_healthy'])            # [(feat, shap, val), ...]

# Use SHAP values in the waterfall plot
from inference_viz import plot_feature_waterfall
plot_feature_waterfall(
    feature_names=feature_names,
    feature_values=X_test[mask.values][sample_idx],
    shap_values=np.array(result['binary_shap_values']),
    prediction_label='HEALTHY' if result['binary_prediction'] == 0 else 'ANOMALOUS',
    prediction_prob=result['binary_probability'][1],
    save_path='./inference_output/shap_waterfall.png',
)
```

---

## Module Reference

### save_models.py

| Function | Purpose | Returns |
|----------|---------|---------|
| `save_all_models(...)` | Save all model artifacts after training | dict of file paths |
| `load_all_models(model_dir=None)` | Load and reconstruct all models | dict with all model objects |
| `load_test_data(model_dir=None)` | Load saved test data | dict: X_test, y_test, test_df |

### inference_viz.py

| Function | Purpose | Output |
|----------|---------|--------|
| `plot_turbine_health_dashboard(...)` | 4-panel turbine health overview | Figure (+ PNG) |
| `plot_confusion_report(...)` | Confusion matrix + classification report | Figure (+ PNG) |
| `plot_feature_waterfall(...)` | SHAP waterfall for one sample | Figure (+ PNG) |
| `plot_multi_model_summary(...)` | All-model-outputs-at-a-glance panel | Figure (+ PNG) |
| `plot_rul_trend(...)` | RUL over time with optional ground truth | Figure (+ PNG) |
| `generate_inference_report(...)` | **Primary function** - generates all plots | dict of paths + text summary |

### shap_explainer.py

| Function/Method | Purpose | Returns |
|----------|---------|---------|
| `SHAPExplainer(binary_clf, multi_clf, feature_names)` | Initialize SHAP explainer | SHAPExplainer instance |
| `.explain_single_sample(X_single, sample_id)` | Per-sample SHAP explanation | dict with predictions, SHAP values, text |
| `.run_full_analysis(X_test, y_test)` | Full batch SHAP analysis | Markdown report string |

---

## Test Data

| File | Shape | Description |
|------|-------|-------------|
| `test_data.npz['X_test']` | (35411, 27) | Feature matrix for turbines 83 and 84 |
| `test_data.npz['y_test']` | (35411,) | Labels: 0=Healthy, 1=Pre-Fault, 2=Fault |
| `test_df.parquet` | DataFrame (35411 rows) | Full data with `date_time`, `turbine_id`, all 27 features, `label`, `hours_to_fault` |

**Label distribution in test data:**
- Healthy: 28,193 (79.6%)
- Pre-Fault: 7,049 (19.9%)
- Fault: 169 (0.5%)

---

## Inference Data Flow Diagram

```
                  Input: X (N, 27) - 27 SCADA features
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
        Autoencoder    Binary Clf    Multi-Class Clf
        (N,27)->(N,)   (N,27)->(N,2) (N,27)->(N,3)
        Health Indicator  P(anomalous)  P(H,PF,F)
              |
              v
           GMM (3)
        (N,1)->(N,)
        Health State
        {0,1,2}
              |
              v
         LSTM RUL
        (24,) -> float
        per-state trend
              |
              v
        All outputs --> generate_inference_report()
                              |
                    +---------+---------+
                    |    |    |    |    |
                    v    v    v    v    v
                 Dashboard  Confusion  Waterfall  Summary  RUL Trend
```

---

## Agentic Usage Pattern

An LLM agent can use this pipeline as follows:

```python
# Agent receives: "Assess health of turbine 83"

import sys, numpy as np
sys.path.insert(0, r'C:\Users\mandart\A_MANDAR_DOCUMENTS\HAM_Wind_Energy\ResultMLYaw\PowerLift\RUL')

# 1. Load models (once, cache in agent memory)
from save_models import load_all_models, load_test_data
models = load_all_models()
test = load_test_data()

# 2. Filter for requested turbine
test_df = test['test_df']
mask = test_df['turbine_id'] == 83
X = test['X_test'][mask.values]
y = test['y_test'][mask.values]

# 3. Run all models
from wind_turbine_pm_fuhrlander import predict_rul
hi = models['autoencoder'].get_health_indicator(X)
raw_states = models['gmm'].predict(hi.reshape(-1, 1))
states = np.array([models['state_order_map'][s] for s in raw_states])
binary_prob = models['binary_clf'].predict_proba(X)
multi_prob = models['multi_clf'].predict_proba(X)
rul = predict_rul(hi, states, models['lstm_models'],
                  models['failure_threshold'], seq_length=24)

# 4. Generate visualizations
from inference_viz import generate_inference_report
report = generate_inference_report(
    turbine_id=83,
    timestamps=test_df.loc[mask, 'date_time'].values,
    health_indicator=hi, gmm_states=states,
    fault_probabilities=binary_prob[:, 1],
    multi_class_probs=multi_prob,
    rul_predictions=rul,
    failure_threshold=models['failure_threshold'],
    feature_names=models['feature_names'],
    true_labels=y,
    output_dir='./inference_output/',
)

# 5. Get SHAP explanation for latest observation
from shap_explainer import SHAPExplainer
explainer = SHAPExplainer(models['binary_clf'], models['multi_clf'],
                          models['feature_names'])
explanation = explainer.explain_single_sample(X[-1:], sample_id='latest')

# 6. Return to user
return {
    'summary': report['text_summary'],
    'explanation': explanation['explanation_text'],
    'plots': [report['dashboard_path'], report['summary_path']],
}
```

---

## File Structure Summary

```
RUL/
    wind_turbine_pm_fuhrlander.py    # Training pipeline (run once to train + save)
    save_models.py                    # Model save/load module
    inference_viz.py                  # Inference visualization module
    shap_explainer.py                 # SHAP explainability module
    README_INFERENCE.md               # This file

    saved_models/                     # Created by training pipeline
        fuhrlander_fl2500_pm_models.joblib
        test_data.npz
        test_df.parquet
        metadata.json

    fuhrlander-master/                # Raw dataset (not needed for inference)
        dataset/
            turbine_80.json.bz2 ... turbine_84.json.bz2
            wind_plant_data.json
```
