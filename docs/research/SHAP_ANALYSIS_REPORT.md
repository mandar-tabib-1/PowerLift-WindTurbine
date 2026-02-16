# SHAP Explainability Analysis Report
## Fuhrlander FL2500 Predictive Maintenance

### Method

This analysis uses **SHAP (SHapley Additive exPlanations)** to explain model
predictions. Unlike the MDI (Mean Decrease in Impurity / Gini importance) used
in the base model, SHAP provides:

1. **Consistent feature attribution** - based on game-theoretic Shapley values
2. **Direction of effect** - shows whether a feature pushes prediction toward
   fault or healthy
3. **Per-sample explanations** - not just global ranking, but why each
   individual prediction was made
4. **Interaction detection** - captures how features work together

SHAP TreeExplainer is used for both:
- **Binary GradientBoosting** (Healthy vs Anomalous)
- **Multi-class RandomForest** (Healthy / Pre-Fault / Fault)

### Global Feature Importance (SHAP)

| Rank | Feature | SHAP Importance (Binary) | SHAP Importance (Fault Class) |
|---|---|---|---|
| 1 | `wtrm_avg_Brg_OilPres` | 0.3691 | 0.0447 |
| 2 | `wtrm_avg_Gbx_OilPres` | 0.2631 | 0.0673 |
| 3 | `wnac_avg_NacTmp` | 0.2380 | 0.0074 |
| 4 | `variability_trend` | 0.1707 | 0.0471 |
| 5 | `wgdc_avg_TriGri_A` | 0.1514 | 0.0164 |
| 6 | `bearing_temp_spread` | 0.1475 | 0.0176 |
| 7 | `oil_pressure_ratio` | 0.1343 | 0.0117 |
| 8 | `wtrm_avg_TrmTmp_GnBrgDE` | 0.0927 | 0.0090 |
| 9 | `oil_temp_trend` | 0.0912 | 0.0062 |
| 10 | `gen_thermal_load` | 0.0876 | 0.0077 |
| 11 | `wgen_sdv_Spd` | 0.0785 | 0.0088 |
| 12 | `wnac_avg_WSpd1` | 0.0696 | 0.0205 |
| 13 | `wgen_avg_Spd` | 0.0637 | 0.0262 |
| 14 | `gbx_temp_trend` | 0.0631 | 0.0070 |
| 15 | `wtrm_avg_TrmTmp_Gbx` | 0.0630 | 0.0087 |
| 16 | `wtrm_avg_TrmTmp_GbxOil` | 0.0559 | 0.0055 |
| 17 | `wtrm_avg_TrmTmp_GnBrgNDE` | 0.0536 | 0.0070 |
| 18 | `wtrm_avg_TrmTmp_GbxBrg151` | 0.0489 | 0.0064 |
| 19 | `power_efficiency` | 0.0448 | 0.0141 |
| 20 | `wgen_avg_GnTmp_phsA` | 0.0442 | 0.0046 |


### MDI (Gini) vs SHAP Comparison

| Rank | MDI (Gini) Top Feature | SHAP Top Feature |
|---|---|---|
| 1 | `variability_trend` | `wtrm_avg_Brg_OilPres` |
| 2 | `wtrm_avg_Brg_OilPres` | `wtrm_avg_Gbx_OilPres` |
| 3 | `wnac_avg_NacTmp` | `wnac_avg_NacTmp` |
| 4 | `oil_pressure_ratio` | `variability_trend` |
| 5 | `wtrm_avg_Gbx_OilPres` | `wgdc_avg_TriGri_A` |


**Key differences between MDI and SHAP:**
- MDI measures how much each feature reduces impurity across all tree splits
  (computed during training). It is biased toward high-cardinality features.
- SHAP measures the marginal contribution of each feature to each individual
  prediction (computed post-hoc). It provides both magnitude AND direction.
- Where MDI and SHAP agree on a feature's importance, there is high confidence
  that the feature is genuinely predictive.
- Where they disagree, SHAP is generally more reliable because it accounts for
  feature correlations and interaction effects.

### Per-Sample Explanations

**Healthy sample (index 16630):**
- Binary: P(anomalous) = 0.258
- Multi-class: P = [H:0.534, PF:0.443, F:0.023]
- Top features pushing toward fault:
  - `wtrm_avg_Gbx_OilPres`: SHAP=+0.6339 (value=1.18)
  - `variability_trend`: SHAP=+0.1863 (value=0.02)
  - `power_efficiency`: SHAP=+0.1206 (value=-0.13)
- Top features pushing toward healthy:
  - `wtrm_avg_Brg_OilPres`: SHAP=-0.3626 (value=0.94)
  - `wnac_avg_WSpd1`: SHAP=-0.2730 (value=2.66)
  - `wgdc_avg_TriGri_PwrAt`: SHAP=-0.1913 (value=-7.78)

**Pre-Fault sample (index 22058):**
- Binary: P(anomalous) = 0.297
- Multi-class: P = [H:0.420, PF:0.478, F:0.102]
- Top features pushing toward fault:
  - `wnac_avg_NacTmp`: SHAP=+0.4791 (value=41.73)
  - `variability_trend`: SHAP=+0.4145 (value=0.05)
  - `wtrm_sdv_TrmTmp_Gbx`: SHAP=+0.2565 (value=0.23)
- Top features pushing toward healthy:
  - `wtrm_avg_Brg_OilPres`: SHAP=-0.5421 (value=0.66)
  - `wtrm_avg_Gbx_OilPres`: SHAP=-0.4089 (value=1.99)
  - `wtrm_avg_TrmTmp_GnBrgNDE`: SHAP=-0.1531 (value=72.40)

**Fault sample (index 15157):**
- Binary: P(anomalous) = 0.159
- Multi-class: P = [H:0.549, PF:0.404, F:0.046]
- Top features pushing toward fault:
  - `oil_pressure_ratio`: SHAP=+0.0934 (value=2.59)
  - `wgdc_avg_TriGri_A`: SHAP=+0.0792 (value=299.79)
  - `gen_thermal_load`: SHAP=+0.0514 (value=33.96)
- Top features pushing toward healthy:
  - `variability_trend`: SHAP=-0.3659 (value=0.00)
  - `wgen_sdv_Spd`: SHAP=-0.0977 (value=22.29)
  - `wtrm_avg_TrmTmp_GbxOil`: SHAP=-0.0906 (value=51.80)


### Generated Plots

- `shap_binary_summary.png`
- `shap_binary_bar.png`
- `shap_multiclass_bar.png`
- `shap_dependence.png`
- `shap_dashboard.png`

### Usage in Agentic / Demo Setup

```python
from shap_explainer import SHAPExplainer

# Initialize with trained models
explainer = SHAPExplainer(
    binary_clf=binary_clf,
    multi_clf=multi_clf,
    feature_names=ALL_FEATURES,
    output_dir='RUL/'
)

# Explain a single new SCADA observation (for real-time demo)
result = explainer.explain_single_sample(
    X_new,           # shape (1, 27) - one hourly observation
    sample_id="demo_turbine_83_hour_5000",
    save_plot=True,
    verbose=True
)

# Access structured results
print(result['explanation_text'])        # Human-readable
print(result['binary_probability'])      # [p_healthy, p_anomalous]
print(result['multi_probability'])       # [p_healthy, p_prefault, p_fault]
print(result['top_features_pushing_fault'])   # [(feat, shap, val), ...]
print(result['top_features_pushing_healthy']) # [(feat, shap, val), ...]

# Full batch analysis on test set
report_md = explainer.run_full_analysis(X_test, y_test, max_samples=2000)
```

---
*Generated by SHAP Explainability Module*
*SHAP method: TreeExplainer (exact Shapley values for tree ensembles)*
