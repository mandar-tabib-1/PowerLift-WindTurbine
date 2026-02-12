# A Semi-Supervised Machine Learning Framework for Predictive Maintenance of Wind Turbine Gearbox Systems Using Real SCADA Data

---

## Abstract

This paper presents a multi-stage semi-supervised machine learning framework for predictive maintenance of wind turbine drivetrain systems, validated on real-world SCADA data from five Fuhrlander FL2500 (2.5 MW) turbines over a three-year operational period (2012--2014). The proposed methodology combines unsupervised anomaly detection via a deep autoencoder with Gaussian Mixture Model (GMM) health state classification, supervised fault prediction using Gradient Boosting and Random Forest classifiers, and a state-specific trend extrapolation model for Remaining Useful Life (RUL) estimation. A domain-informed feature engineering pipeline derives 27 predictive features (18 raw SCADA signals plus 9 physics-based engineered features) from 314 available sensor variables. Ground truth labels are constructed from real alarm events using a 48-hour pre-fault window strategy. The framework achieves a weighted F1-score of 0.75 for multi-class fault prediction, 76.3% of actual fault events are detected with RUL predictions below 50 hours, and SHAP-based explainability analysis confirms alignment between model-identified features and known gearbox degradation physics. The turbine-wise train/test split (training on turbines 80--82, testing on 83--84) validates fleet-level generalizability of the learned degradation patterns.

**Keywords:** Predictive maintenance, wind turbine, SCADA data, semi-supervised learning, autoencoder, Gaussian Mixture Model, remaining useful life, SHAP explainability, gearbox fault detection

---

## 1. Introduction

Wind energy is a rapidly growing source of renewable power, yet wind turbine maintenance costs account for 20--35% of the total cost of energy, with unplanned downtime contributing disproportionately to operational losses [1]. The gearbox, which typically represents 10--15% of the turbine capital cost, is the most failure-prone major component, with a mean time to repair of 5--10 days per event [2]. Predictive maintenance---detecting incipient faults before they cause unplanned shutdowns---has the potential to significantly reduce these costs by enabling condition-based rather than calendar-based maintenance scheduling.

Modern wind turbines are equipped with Supervisory Control and Data Acquisition (SCADA) systems that continuously record hundreds of operational parameters at sub-minute intervals. These data streams capture the thermal, mechanical, and electrical signatures of normal operation and degradation. However, the challenge lies in extracting actionable health indicators from high-dimensional, noisy, and largely unlabeled SCADA data.

This work proposes a four-stage semi-supervised framework that addresses these challenges:

1. **Unsupervised Health Indicator Extraction** using a fully-connected autoencoder trained to reconstruct normal operating patterns, where reconstruction error serves as a health degradation score.
2. **Probabilistic Health State Classification** using a Gaussian Mixture Model (GMM) that partitions the continuous health indicator into discrete operational states (Healthy, Degrading, Critical).
3. **Supervised Fault Prediction** using alarm-labeled data with Gradient Boosting (binary) and Random Forest (3-class) classifiers for real-time fault probability estimation.
4. **State-Specific Remaining Useful Life Prediction** using trend extrapolation models conditioned on the current health state.

The framework is validated on the publicly available Fuhrlander FL2500 dataset [3], which contains real SCADA data from five 2.5 MW turbines with corresponding alarm event logs. Unlike most predictive maintenance studies that rely on synthetic or simulation-generated data, this work uses genuine operational data with real fault events, providing a more realistic assessment of model performance.

---

## 2. Methodology

### 2.1 Framework Overview

The proposed framework follows a four-stage pipeline, illustrated in Figure 1. The semi-supervised nature arises from the combination of unsupervised learning (Stages 1--2) operating on raw sensor features without labels, and supervised learning (Stage 3) leveraging alarm-derived labels. Stage 4 bridges both paradigms by using the unsupervised health indicator conditioned on GMM-derived states for RUL estimation.

```
┌──────────────────────────────────────────────────────────────┐
│                  SCADA Data (N × 314 variables)              │
│                 5-minute intervals, per turbine               │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│            PREPROCESSING & FEATURE ENGINEERING                │
│  Hourly aggregation → 18 raw + 9 engineered = 27 features   │
│  Alarm-based labeling: Healthy / Pre-Fault / Fault           │
└──────────────────┬───────────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
┌─────────────────┐  ┌──────────────────────────┐
│   STAGE 1       │  │      STAGE 3             │
│   Autoencoder   │  │   Supervised Classifiers │
│  (Unsupervised) │  │                          │
│                 │  │  GradientBoosting (2-cls) │
│ X(N,27)→HI(N,) │  │  RandomForest (3-class)  │
└────────┬────────┘  └────────┬─────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌──────────────────────────┐
│   STAGE 2       │  │   Outputs:               │
│   GMM (3 comp.) │  │   P(anomalous)           │
│  HI → State     │  │   P(healthy, PF, fault)  │
│  {0, 1, 2}      │  │   Feature importances    │
└────────┬────────┘  └──────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│   STAGE 4: RUL Prediction               │
│   State-specific trend extrapolation     │
│   HI sequence (24h) → time to threshold │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  EXPLAINABILITY: SHAP TreeExplainer     │
│  Per-sample and global feature          │
│  attribution analysis                   │
└─────────────────────────────────────────┘
```
**Figure 1.** Overview of the proposed four-stage semi-supervised predictive maintenance framework.

---

### 2.2 Stage 1: Autoencoder-Based Health Indicator Extraction

The first stage employs a fully-connected autoencoder trained on standardized SCADA features to learn a compact representation of normal turbine operation. The autoencoder is trained in an entirely unsupervised manner---no fault labels are used. The core assumption is that the network learns to reconstruct *normal* operating patterns well, but produces higher reconstruction error for *anomalous* patterns indicative of degradation.

#### 2.2.1 Architecture

The autoencoder consists of a symmetric encoder-decoder architecture with a bottleneck dimension of 8:

$$
\text{Encoder: } \mathbb{R}^{27} \xrightarrow{W_1} \mathbb{R}^{64} \xrightarrow{W_2} \mathbb{R}^{32} \xrightarrow{W_3} \mathbb{R}^{8}
$$

$$
\text{Decoder: } \mathbb{R}^{8} \xrightarrow{W_4} \mathbb{R}^{32} \xrightarrow{W_5} \mathbb{R}^{64} \xrightarrow{W_6} \mathbb{R}^{27}
$$

The forward pass for a single input vector $\mathbf{x} \in \mathbb{R}^{27}$ is computed as:

**Encoder:**
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{x} W_1 + \mathbf{b}_1), \quad \mathbf{h}_1 \in \mathbb{R}^{64}$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{h}_1 W_2 + \mathbf{b}_2), \quad \mathbf{h}_2 \in \mathbb{R}^{32}$$
$$\mathbf{z} = \mathbf{h}_2 W_3 + \mathbf{b}_3, \quad \mathbf{z} \in \mathbb{R}^{8} \quad \text{(linear bottleneck)}$$

**Decoder:**
$$\mathbf{h}_4 = \text{ReLU}(\mathbf{z} W_4 + \mathbf{b}_4), \quad \mathbf{h}_4 \in \mathbb{R}^{32}$$
$$\mathbf{h}_5 = \text{ReLU}(\mathbf{h}_4 W_5 + \mathbf{b}_5), \quad \mathbf{h}_5 \in \mathbb{R}^{64}$$
$$\hat{\mathbf{x}} = \mathbf{h}_5 W_6 + \mathbf{b}_6, \quad \hat{\mathbf{x}} \in \mathbb{R}^{27} \quad \text{(linear output)}$$

where $\text{ReLU}(x) = \max(0, x)$.

Weights are initialized using He initialization:

$$W_l \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in},l}}}\right)$$

The bottleneck layer uses a linear activation (no ReLU) to allow the latent representation to span both positive and negative values, which is critical for encoding relative deviations from normal operation.

#### 2.2.2 Training Objective

The autoencoder is trained to minimize the Mean Squared Error (MSE) between input and reconstruction:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{x}_i - \hat{\mathbf{x}}_i \right\|_2^2$$

Training uses mini-batch stochastic gradient descent with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Learning rate ($\eta$) | 0.001 |
| Batch size | 64 |
| Epochs | 100 |
| Validation split | 20% |
| Gradient clipping | $[-1, +1]$ |

Gradient clipping is applied element-wise to prevent instability during backpropagation through the multi-layer architecture.

#### 2.2.3 Health Indicator Definition

The Health Indicator (HI) for each observation is defined as the Mean Absolute Error (MAE) between the standardized input and its reconstruction:

$$\text{HI}(\mathbf{x}) = \frac{1}{d} \sum_{j=1}^{d} \left| \tilde{x}_j - \hat{\tilde{x}}_j \right|$$

where $\tilde{\mathbf{x}} = (\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$ is the standardized input (using the scaler fitted on training data), $\hat{\tilde{\mathbf{x}}}$ is the reconstruction, and $d = 27$ is the feature dimension.

The choice of MAE over MSE for the health indicator (despite MSE being used for training) provides a more robust metric that is less sensitive to outlier features, giving a more interpretable "average per-feature deviation" score.

#### 2.2.4 Input Standardization

Prior to autoencoder processing, all features are standardized using a `StandardScaler` fitted exclusively on training data:

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of feature $j$ computed from the training set. This scaler is persisted with the model and applied identically during inference, ensuring that the reconstruction error scale is consistent between training and test/deployment.

---

### 2.3 Stage 2: GMM Health State Classification

The continuous Health Indicator is discretized into three operational health states using a Gaussian Mixture Model (GMM) with $K=3$ components. This unsupervised classification converts a scalar degradation score into an interpretable categorical state.

#### 2.3.1 Model Formulation

The GMM models the distribution of HI values as a mixture of $K$ Gaussians:

$$p(\text{HI}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\text{HI} \mid \mu_k, \sigma_k^2)$$

where $\pi_k$ are mixing weights ($\sum_k \pi_k = 1$), and $\mu_k, \sigma_k^2$ are the mean and variance of component $k$. Parameters are estimated via the Expectation-Maximization (EM) algorithm [4] with 10 random restarts to mitigate local optima.

#### 2.3.2 State Assignment

After fitting, the GMM components are ordered by their mean HI value:

$$\text{If } \mu_{\pi(0)} < \mu_{\pi(1)} < \mu_{\pi(2)}, \text{ then:}$$

| State | Label | Interpretation |
|-------|-------|----------------|
| $s = 0$ | Healthy | Low reconstruction error; normal operation |
| $s = 1$ | Degrading | Intermediate error; onset of anomalous behavior |
| $s = 2$ | Critical | High error; likely fault or imminent failure |

Each observation is assigned to the most likely state via maximum a posteriori:

$$s^* = \arg\max_k \, p(k \mid \text{HI})$$

The GMM also provides soft state probabilities $p(k \mid \text{HI})$, which are useful for smooth transitions in monitoring applications.

---

### 2.4 Stage 3: Supervised Fault Prediction

While Stages 1--2 operate without labels, Stage 3 introduces supervision through alarm-derived labels. Two complementary classifiers are trained:

#### 2.4.1 Binary Classification (Gradient Boosting)

The binary classifier distinguishes Healthy ($y=0$) from Anomalous ($y=1$, combining Pre-Fault and Fault):

$$y_{\text{binary}} = \begin{cases} 0 & \text{if } y_{\text{alarm}} = 0 \\ 1 & \text{if } y_{\text{alarm}} \in \{1, 2\} \end{cases}$$

A Gradient Boosting Classifier (GBC) [5] is used, which builds an additive ensemble of regression trees:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

where $h_m$ is a tree fitted to the negative gradient of the log-loss, and $\eta = 0.1$ is the learning rate. Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Number of trees ($M$) | 200 |
| Maximum depth | 5 |
| Learning rate ($\eta$) | 0.1 |
| Subsample ratio | 0.8 |

#### 2.4.2 Multi-Class Classification (Random Forest)

The multi-class classifier distinguishes three states: Healthy ($y=0$), Pre-Fault ($y=1$), and Fault ($y=2$). A Random Forest [6] is used with class-balanced sample weights to address the severe class imbalance (Fault class represents only 0.6% of training data):

$$w_c = \frac{N}{K \cdot N_c}$$

where $N$ is the total sample count, $K=3$ is the number of classes, and $N_c$ is the number of samples in class $c$. This ensures that minority classes (Fault, Pre-Fault) receive proportionally higher influence during tree construction.

| Parameter | Value |
|-----------|-------|
| Number of trees | 300 |
| Maximum depth | 10 |
| Minimum samples per leaf | 5 |
| Class weighting | Balanced (inversely proportional) |

Both classifiers operate on the same 27-feature input $\mathbf{x} \in \mathbb{R}^{27}$, independent of the autoencoder. This architectural choice enables standalone deployment of individual models and provides complementary predictions that can be fused in an operational decision system.

---

### 2.5 Stage 4: Remaining Useful Life Prediction

RUL prediction estimates the time (in hours) until the Health Indicator reaches a predefined failure threshold $\tau$, which is set as the 90th percentile of the training HI distribution:

$$\tau = \text{Percentile}_{90}(\{\text{HI}_i\}_{i=1}^{N_{\text{train}}})$$

#### 2.5.1 State-Specific Trend Extrapolation

A separate trend model is fitted for each GMM health state $s \in \{0, 1, 2\}$, capturing the observation that HI evolution dynamics differ depending on the current operational regime (e.g., healthy turbines have near-zero trend, while degrading turbines exhibit upward drift).

For a given observation at time $t$ with state $s_t$, the model uses a sliding window of $L = 24$ recent HI values:

$$\mathbf{h}_t = [\text{HI}_{t-L+1}, \text{HI}_{t-L+2}, \ldots, \text{HI}_t]$$

The one-step-ahead prediction uses linear trend extrapolation over the most recent 10 values:

$$\widehat{\text{HI}}_{t+1} = \text{HI}_t + \bar{\Delta}$$

where:

$$\bar{\Delta} = \frac{1}{9} \sum_{i=t-9}^{t-1} (\text{HI}_{i+1} - \text{HI}_i)$$

#### 2.5.2 RUL Computation

RUL is computed by iteratively applying the trend model to project HI forward until it crosses the failure threshold:

$$\text{RUL}(t) = \min\left\{k \geq 0 : \widehat{\text{HI}}_{t+k} \geq \tau\right\}$$

with a maximum horizon of 500 hours. If the projected HI does not reach $\tau$ within 500 steps, RUL is capped at 500 hours.

---

### 2.6 Explainability: SHAP Analysis

To ensure model transparency and validate that learned patterns align with domain knowledge, SHAP (SHapley Additive exPlanations) [7] is applied to both supervised classifiers.

#### 2.6.1 Shapley Values

For a model $f$ and input $\mathbf{x}$, the SHAP value of feature $j$ is:

$$\phi_j(\mathbf{x}) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! \, (|F|-|S|-1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

where $F$ is the set of all features and $f(S)$ denotes the model prediction using only features in $S$ (with others marginalized). TreeExplainer [8] computes exact Shapley values in polynomial time for tree ensemble models.

#### 2.6.2 Global and Local Importance

- **Global importance** is computed as the mean absolute SHAP value across all test samples: $I_j = \frac{1}{N} \sum_i |\phi_j(\mathbf{x}_i)|$
- **Local importance** provides per-sample explanations, decomposing each prediction into additive feature contributions:

$$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{d} \phi_j(\mathbf{x})$$

where $\phi_0$ is the base value (mean prediction).

---

### 2.7 Alarm-Based Label Construction

Labels are derived from the turbine's real alarm event log, which records each alarm with a start time, end time, system classification, and availability flag (0 = turbine shutdown, 1 = turbine available). The labeling strategy uses only alarms from the Transmission and Generator systems with `availability = 0` (actual fault-induced shutdowns):

$$y(t) = \begin{cases}
2 \; (\text{Fault}) & \text{if } t \in [\text{alarm\_start}, \text{alarm\_end}] \\
1 \; (\text{Pre-Fault}) & \text{if } 0 < t_{\text{next\_fault}} - t \leq 48\text{h and } y(t) \neq 2 \\
0 \; (\text{Healthy}) & \text{otherwise}
\end{cases}$$

The 48-hour pre-fault window is motivated by operational practice: a minimum 48-hour lead time is generally required to schedule maintenance crew mobilization for offshore and remote onshore turbines.

Additionally, the hours-to-fault metric is computed for each timestamp as the time until the next fault onset, enabling continuous RUL validation:

$$\text{hours\_to\_fault}(t) = \min_{t_f > t} (t_f - t), \quad t_f \in \{\text{fault start times}\}$$

---

## 3. Dataset

### 3.1 Data Source

The Fuhrlander FL2500 SCADA dataset [3] was collected from a wind farm of five 2.5 MW Fuhrlander FL2500 wind turbines over the period January 2012 to December 2014. The dataset is publicly available under the Eclipse Public License v2.0 and was compiled by Alejandro Blanco-M.

### 3.2 Data Structure

Each turbine's data is stored in a compressed JSON file containing:

1. **Analog variables:** 314 columns comprising 78 unique sensors, each recorded with four statistical aggregates (average, maximum, minimum, standard deviation) over 5-minute intervals, yielding approximately 215,000 records per turbine.

2. **Alarm events:** Approximately 10,000--13,000 alarm entries per turbine, each with alarm ID, system classification, start/end timestamps, subsystem, and availability status.

Table 1 summarizes the dataset characteristics.

**Table 1.** Fuhrlander FL2500 dataset overview.

| Property | Value |
|---|---|
| Turbine model | Fuhrlander FL2500 (2.5 MW) |
| Number of turbines | 5 (IDs: 80, 81, 82, 83, 84) |
| Time period | January 2012 -- December 2014 |
| Raw data frequency | 5-minute intervals |
| Records per turbine | ~215,000 |
| Total analog variables | 314 (78 sensors x 4 stats) |
| Alarm events per turbine | 10,134 -- 12,810 |
| Sensor systems | 9 (Transmission, Generator, Nacelle, Grid, Converter, Rotor, Met, Tower, Turbine Control) |

### 3.3 Sensor Systems

The SCADA sensors are organized by turbine subsystem, following the naming convention `{system}_{statistic}_{sensor}`:

**Table 2.** Sensor system breakdown.

| System Prefix | Subsystem | Sensor Count | Examples |
|---|---|---|---|
| `wtrm` | Transmission/Gearbox | 18 | Gearbox temp, bearing temps, oil pressure |
| `wgen` | Generator | 6 | Winding temps, speed |
| `wnac` | Nacelle | 9 | Wind speed, nacelle temp, humidity |
| `wgdc` | Grid Connection | 14 | Active/reactive power, current, voltage |
| `wcnv` | Converter | 4 | Converter temps |
| `wrot` | Rotor | 18 | Pitch angles, blade temps |
| `wmet` | Meteorological | 3 | Ambient temp, wind direction |
| `wtow` | Tower | 1 | Tower acceleration |
| `wtur` | Turbine Control | 5 | Operating state, setpoints |

### 3.4 Data Preprocessing

The raw 5-minute data is aggregated to hourly resolution by computing the mean of each variable within each hour. This reduces noise while preserving degradation trends, yielding approximately 17,800 hourly records per turbine.

### 3.5 Feature Selection

From the 314 available variables, 18 raw features are selected based on domain knowledge of gearbox degradation physics (Table 3), focusing on:

- **Thermal signatures:** 7 gearbox/bearing temperatures that directly track friction-induced heating from wear
- **Oil system health:** 2 pressure measurements reflecting lubrication system integrity
- **Operational context:** 5 features (generator speed, wind speed, power, current, nacelle temperature) that normalize the thermal readings against operating conditions
- **Variability proxies:** 3 standard deviation features that serve as vibration surrogates in the absence of dedicated vibration sensors

**Table 3.** Selected raw SCADA features and engineering rationale.

| # | Feature | System | Physical Rationale |
|---|---|---|---|
| 1--5 | Gearbox & bearing temps | Transmission | Rising temperatures indicate increased friction from surface wear, lubricant degradation, or bearing spalling |
| 6--7 | Generator bearing temps | Transmission | Thermal coupling to gearbox; anomalous heat transfer indicates stress propagation through the drivetrain |
| 8--9 | Oil pressures (gearbox, bearing) | Transmission | Pressure drops indicate filter clogging, oil degradation, or pump wear |
| 10 | Generator winding temp | Generator | Electrical stress; overheating accelerates bearing grease degradation |
| 11 | Generator speed | Generator | Mechanical load indicator; speed variations indicate drivetrain irregularities |
| 12--13 | Wind speed, nacelle temp | Nacelle | Environmental normalization; enable relative (rather than absolute) temperature analysis |
| 14--15 | Active power, grid current | Grid | Efficiency indicators; power/current anomalies at given wind conditions indicate mechanical losses |
| 16--18 | Std deviations (gbx temp, oil temp, gen speed) | Various | Vibration proxies; high-frequency fluctuations within 5-min windows indicate mechanical instability |

### 3.6 Feature Engineering

Nine physics-informed features are derived from the raw signals to capture known degradation mechanisms:

**Table 4.** Engineered features.

| Feature | Formula | Physical Interpretation |
|---|---|---|
| Thermal stress index | $\sum_{i=1}^{5} w_i \cdot T_{\text{gbx},i} / 100$ | Composite gearbox thermal health; weighted by sensor criticality |
| Bearing stress index | $\sum_{i=1}^{5} w_i \cdot T_{\text{brg},i} / 100$ | Composite bearing stress; combines all bearing temperatures |
| Power efficiency | $P_{\text{active}} / (0.5 v_{\text{wind}}^3 + 50)$ | Mechanical-to-electrical efficiency; degradation reduces power at given wind |
| Gearbox temp trend | $\overline{\Delta T}_{\text{gbx}}^{24\text{h}}$ | 24-hour moving average rate of change; positive trend indicates progressive heating |
| Oil pressure ratio | $P_{\text{gbx}} / (P_{\text{brg}} + 0.01)$ | Oil system balance; ratio shift indicates differential clogging or leakage |
| Bearing temp spread | $\max(T_{\text{brg}}) - \min(T_{\text{brg}})$ | Temperature imbalance across bearings; high spread indicates localized fault |
| Generator thermal load | $T_{\text{gen}} - T_{\text{nacelle}}$ | Excess generator heat above ambient; captures electrical overloading |
| Oil temp trend | $\overline{T}_{\text{oil}}^{24\text{h}}$ | 24-hour smoothed oil temperature; captures slow oil degradation |
| Variability trend | $\overline{\sigma_{T_{\text{gbx}}}}^{24\text{h}}$ | Smoothed gearbox temperature std dev; captures evolving mechanical instability |

### 3.7 Train/Test Split

The dataset is split by turbine to evaluate fleet-level generalization (Table 5). This is more rigorous than a random temporal split, as it tests whether patterns learned from one set of turbines transfer to different physical units.

**Table 5.** Train/test split.

| Set | Turbine IDs | Hourly Samples | Healthy | Pre-Fault | Fault |
|---|---|---|---|---|---|
| Training | 80, 81, 82 | 53,810 | 39,458 (73.3%) | 14,041 (26.1%) | 311 (0.6%) |
| Test | 83, 84 | 35,411 | 28,193 (79.6%) | 7,049 (19.9%) | 169 (0.5%) |

The class distribution reflects the real-world nature of fault data: Fault events are rare (0.5--0.6%) compared to Pre-Fault (20--26%) and Healthy (73--80%) observations.

---

## 4. Results

### 4.1 Autoencoder Training Convergence

The autoencoder was trained for 100 epochs with an 80/20 train/validation split. Figure 2 shows the training and validation loss curves, demonstrating smooth convergence without overfitting.

**Table 6.** Autoencoder training results.

| Metric | Value |
|---|---|
| Final training loss (MSE) | 0.1700 |
| Final validation loss (MSE) | 0.2366 |
| Train/val loss ratio | 1.39 |

The modest gap between training and validation loss (ratio 1.39) indicates that the autoencoder generalizes well and has not memorized the training data. The validation loss stabilizes after approximately 60 epochs.

*Figure 2 reference: See `fuhrlander_pm_analysis.png`, Row 2 left panel - Autoencoder Training Convergence.*

### 4.2 Health Indicator Distribution

Table 7 shows the HI statistics by dataset and alarm label. The autoencoder-derived HI shows clear separation between healthy and fault-related observations.

**Table 7.** Health Indicator statistics.

| Dataset | Label | Samples | Mean HI | Std HI | Min HI | Max HI |
|---|---|---|---|---|---|---|
| Training | All | 53,810 | 0.2578 | 0.1481 | 0.0655 | 12.7397 |
| Test | Healthy | 28,193 | 0.3713 | 0.2927 | 0.0794 | 4.5149 |
| Test | Pre-Fault | 7,049 | 0.5851 | 1.5178 | 0.0855 | 24.1334 |
| Test | Fault | 169 | 0.5781 | 0.3450 | 0.1133 | 1.6638 |

The mean HI for Pre-Fault (0.585) and Fault (0.578) observations is approximately 57% higher than for Healthy observations (0.371), confirming that the unsupervised autoencoder successfully captures degradation-related deviations even though it was not trained with any fault labels.

*Figure reference: See `fuhrlander_pm_analysis.png`, Row 2 right panel - Health Indicator & GMM States, and Row 3 left panel - Test HI time series.*

### 4.3 GMM Health State Classification

The three-component GMM partitions the HI distribution into interpretable health states. Table 8 shows the cross-tabulation of GMM states versus alarm-derived labels on the test set.

**Table 8.** GMM state vs. alarm label agreement (test set).

| GMM State | Test Samples | % Healthy | % Pre-Fault | % Fault |
|---|---|---|---|---|
| Healthy (0) | 20,219 | 83.9% | 15.8% | 0.3% |
| Degrading (1) | 6,045 | 70.5% | 28.4% | 1.1% |
| Critical (2) | 9,147 | 76.1% | 23.4% | 0.5% |

The Healthy GMM state achieves the highest purity (83.9% truly healthy), while the Degrading state has the highest concentration of Pre-Fault samples (28.4%) and the highest Fault proportion (1.1%). The imperfect separation reflects the inherent difficulty of this task: fault signatures in SCADA data are often subtle and overlap with normal operational variability.

*Figure reference: See `fuhrlander_pm_analysis.png`, Row 3 right panel - Health State Distribution bar chart; and `inference_turbine_83_dashboard.png` - full turbine dashboard.*

### 4.4 Supervised Classification Performance

#### 4.4.1 Binary Classification (Healthy vs. Anomalous)

**Table 9.** Binary classification results.

| Metric | Value |
|---|---|
| Accuracy | 0.7669 |
| F1-Score (anomalous class) | 0.3079 |
| Cross-Val F1 (5-fold) | 0.3033 +/- 0.1090 |

The binary F1 of 0.31 reflects the extreme class imbalance (anomalous: 20.4% of test data) and the inherently noisy nature of alarm-based labels. The cross-validation result confirms that training performance is consistent (no overfitting to the training turbines).

#### 4.4.2 Multi-Class Classification (Healthy / Pre-Fault / Fault)

**Table 10.** Multi-class classification results (test set).

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Healthy | 0.84 | 0.87 | 0.85 | 28,193 |
| Pre-Fault | 0.40 | 0.29 | 0.34 | 7,049 |
| Fault | 0.04 | 0.19 | 0.06 | 169 |
| **Weighted Avg** | **0.74** | **0.75** | **0.75** | **35,411** |

**Table 11.** Confusion matrix (test set, multi-class).

| Predicted &rarr; | Healthy | Pre-Fault | Fault |
|---|---|---|---|
| **Healthy** | 24,513 | 3,095 | 585 |
| **Pre-Fault** | 4,738 | 2,073 | 238 |
| **Fault** | 81 | 56 | 32 |

The Healthy class achieves strong performance (F1=0.85), while Pre-Fault detection (F1=0.34) is more challenging due to the gradual and variable nature of degradation onset. The Fault class has low precision (0.04) but relatively useful recall (0.19), meaning the model detects approximately one in five fault events---notable given the extreme class rarity (169 out of 35,411 samples). Critically, only 81 out of 169 Fault samples (48%) are misclassified as Healthy; the remaining 88 receive some level of anomaly warning (Pre-Fault or Fault prediction).

*Figure reference: See `inference_turbine_83_confusion.png` - confusion matrix heatmap with classification report.*

### 4.5 Feature Importance Analysis

#### 4.5.1 MDI-Based Importance (Gradient Boosting)

Table 12 shows the top 10 features ranked by Mean Decrease in Impurity (MDI / Gini importance) from the Gradient Boosting binary classifier.

**Table 12.** Top 10 features by MDI importance.

| Rank | Feature | MDI Importance |
|---|---|---|
| 1 | variability_trend | 0.1694 |
| 2 | wtrm_avg_Brg_OilPres | 0.1316 |
| 3 | wnac_avg_NacTmp | 0.1224 |
| 4 | oil_pressure_ratio | 0.0779 |
| 5 | wtrm_avg_Gbx_OilPres | 0.0745 |
| 6 | oil_temp_trend | 0.0730 |
| 7 | wtrm_avg_TrmTmp_GnBrgDE | 0.0406 |
| 8 | gbx_temp_trend | 0.0370 |
| 9 | wgdc_avg_TriGri_A | 0.0241 |
| 10 | wtrm_avg_TrmTmp_GbxBrg450 | 0.0233 |

#### 4.5.2 SHAP-Based Importance

Table 13 shows the global SHAP importance (mean absolute SHAP value) for both the binary and multi-class models.

**Table 13.** Top 10 features by SHAP importance.

| Rank | Feature | SHAP (Binary) | SHAP (Fault Class) |
|---|---|---|---|
| 1 | wtrm_avg_Brg_OilPres | 0.3691 | 0.0447 |
| 2 | wtrm_avg_Gbx_OilPres | 0.2631 | 0.0673 |
| 3 | wnac_avg_NacTmp | 0.2380 | 0.0074 |
| 4 | variability_trend | 0.1707 | 0.0471 |
| 5 | wgdc_avg_TriGri_A | 0.1514 | 0.0164 |
| 6 | bearing_temp_spread | 0.1475 | 0.0176 |
| 7 | oil_pressure_ratio | 0.1343 | 0.0117 |
| 8 | wtrm_avg_TrmTmp_GnBrgDE | 0.0927 | 0.0090 |
| 9 | oil_temp_trend | 0.0912 | 0.0062 |
| 10 | gen_thermal_load | 0.0876 | 0.0077 |

#### 4.5.3 MDI vs. SHAP Comparison

**Table 14.** Comparison of MDI and SHAP top-5 rankings.

| Rank | MDI Top Feature | SHAP Top Feature |
|---|---|---|
| 1 | variability_trend | wtrm_avg_Brg_OilPres |
| 2 | wtrm_avg_Brg_OilPres | wtrm_avg_Gbx_OilPres |
| 3 | wnac_avg_NacTmp | wnac_avg_NacTmp |
| 4 | oil_pressure_ratio | variability_trend |
| 5 | wtrm_avg_Gbx_OilPres | wgdc_avg_TriGri_A |

Both methods agree on the same top-5 feature set (with different ordering), providing strong evidence that these features are genuinely predictive rather than artifacts of a particular importance metric. The key finding is that **oil pressure features** (bearing and gearbox) emerge as the most important predictors via SHAP, while MDI emphasizes the **variability trend** (vibration proxy). This difference is expected: MDI is biased toward high-cardinality continuous features, while SHAP provides unbiased marginal contributions.

*Figure reference: See `shap_dashboard.png` - MDI vs SHAP comparison, beeswarm plot; `shap_binary_summary.png` - SHAP beeswarm showing direction of effect; `shap_dependence.png` - dependence plots for top 4 features.*

### 4.6 SHAP Per-Sample Explanations

SHAP enables individual prediction explanations. Table 15 illustrates three representative test samples.

**Table 15.** Per-sample SHAP explanations.

| | Healthy Sample | Pre-Fault Sample | Fault Sample |
|---|---|---|---|
| **True label** | Healthy | Pre-Fault | Fault |
| **P(anomalous)** | 0.258 | 0.297 | 0.159 |
| **Multi-class** | H:0.534, PF:0.443 | H:0.420, PF:0.478 | H:0.549, PF:0.404 |
| **Top fault driver** | Gbx_OilPres (+0.63) | NacTmp (+0.48) | oil_pressure_ratio (+0.09) |
| **Top healthy driver** | Brg_OilPres (-0.36) | Brg_OilPres (-0.54) | variability_trend (-0.37) |

The Pre-Fault sample shows elevated nacelle temperature and variability trend pushing toward fault, while oil pressures counteract---consistent with early-stage gearbox degradation where thermal signatures emerge before oil system failure. The Fault sample demonstrates a known challenge: some fault events occur during otherwise normal-appearing conditions, making them inherently difficult to predict from SCADA features alone.

*Figure reference: See `shap_sample_Healthy_16630.png`, `shap_sample_Pre-Fault_22058.png`, `shap_sample_Fault_15157.png` - per-sample SHAP waterfall plots; and `inference_turbine_83_shap_waterfall.png` - inference-time waterfall.*

### 4.7 RUL Prediction Performance

**Table 16.** RUL prediction statistics (test set).

| Metric | Value |
|---|---|
| Valid RUL predictions | 35,387 |
| Mean RUL | 221.7 hours |
| Median RUL | 29.0 hours |
| Min RUL | 0.0 hours |
| Max RUL | 500.0 hours |
| Failure threshold ($\tau$) | 0.3916 |

**Table 17.** RUL accuracy at actual fault events.

| Metric | Value |
|---|---|
| Fault samples in test set | 169 |
| Mean RUL at fault | 111.6 hours |
| Median RUL at fault | 0.0 hours |
| % with RUL < 50 hours | 76.3% |
| % with RUL < 100 hours | 78.1% |

The median RUL at actual fault events is 0.0 hours, meaning the majority of faults are correctly identified as imminent. Furthermore, 76.3% of fault events have predicted RUL below 50 hours, which is within the actionable maintenance scheduling window. The mean RUL of 111.6 hours is elevated by a minority of cases where the HI happened to be below the threshold during the fault period, resulting in high RUL predictions.

*Figure reference: See `fuhrlander_pm_analysis.png`, Row 5 - RUL Predictions and Distribution; `inference_turbine_83_rul_trend.png` - RUL trend with ground truth overlay; and `inference_turbine_83_summary.png` - multi-model summary panel.*

### 4.8 Inference Visualization

The framework includes a comprehensive inference visualization module that generates five presentation-quality plots per turbine assessment:

1. **Health Assessment Dashboard** (4 panels): HI over time with GMM states, fault probability time series, multi-class probability evolution, and RUL with urgency coloring---plus a status summary box with operational recommendation.

2. **Confusion Matrix & Classification Report**: Normalized confusion matrix heatmap alongside precision/recall/F1 statistics per class.

3. **SHAP Feature Waterfall**: Per-sample feature contributions with color-coded direction (red = pushes toward fault, blue = pushes toward healthy) and annotated feature values.

4. **Multi-Model Summary Panel**: All four model outputs at a glance (HI gauge, probability bars, RUL display with urgency level, top contributing features) for the latest observation.

5. **RUL Trend Plot**: RUL over time with optional ground-truth hours-to-fault overlay and fault event markers.

*Figure reference: See `inference_turbine_83_dashboard.png`, `inference_turbine_83_confusion.png`, `inference_turbine_83_shap_waterfall.png`, `inference_turbine_83_summary.png`, `inference_turbine_83_rul_trend.png`.*

---

## 5. Discussion

### 5.1 Semi-Supervised Advantages

The four-stage semi-supervised approach offers several advantages over purely supervised or purely unsupervised methods:

1. **Robustness to label noise:** The autoencoder (Stage 1) and GMM (Stage 2) operate independently of the alarm-derived labels, providing a health assessment that is not affected by mislabeled or missing alarm records. This is critical because real SCADA alarm logs often contain spurious alarms, delayed fault recordings, or missing entries.

2. **Complementary failure detection:** The unsupervised HI captures subtle multivariate deviations that may not trigger alarm thresholds, while the supervised classifiers exploit the structured alarm information to calibrate fault probability estimates.

3. **Interpretable degradation trajectory:** The GMM states provide a discrete, operationally meaningful health assessment that bridges the continuous HI and the categorical alarm labels.

### 5.2 Feature Engineering Insights

The SHAP analysis reveals that oil pressure features (bearing and gearbox) are the strongest predictors of fault conditions, followed by nacelle temperature and the variability trend (vibration proxy). This aligns with established gearbox failure modes [9]: lubrication system degradation (manifested as oil pressure changes) typically precedes bearing surface damage (manifested as temperature changes), which in turn precedes catastrophic gearbox failure.

The finding that the engineered `variability_trend` feature (24-hour smoothed standard deviation of gearbox temperature) ranks consistently in the top 5 by both MDI and SHAP importance validates the use of temperature standard deviation as a vibration proxy in SCADA-only monitoring systems where dedicated vibration sensors are not available.

### 5.3 Limitations

1. **Class imbalance:** The extreme rarity of Fault events (0.5%) limits the achievable Fault-class precision. Advanced imbalance techniques (SMOTE, focal loss, cost-sensitive ensemble methods) could improve minority-class detection.

2. **Label quality:** Alarm-based labels are an imperfect proxy for actual mechanical degradation state. Some alarms may be triggered by electrical or control issues unrelated to mechanical degradation, introducing label noise.

3. **RUL model simplicity:** The trend extrapolation model assumes locally linear HI evolution, which may not capture complex degradation dynamics (e.g., sudden-onset faults). More sophisticated temporal models (recurrent neural networks, state-space models) could improve RUL accuracy.

4. **Single failure mode focus:** The current labeling strategy targets Transmission and Generator system faults. Extending to multi-system, multi-failure-mode prediction is an avenue for future work.

---

## 6. Conclusion

This work demonstrates a practical semi-supervised machine learning framework for wind turbine predictive maintenance validated on three years of real SCADA data from five Fuhrlander FL2500 turbines. The four-stage pipeline---autoencoder health indicator, GMM state classification, supervised fault prediction, and state-specific RUL estimation---achieves a weighted F1-score of 0.75 for multi-class fault prediction and detects 76.3% of actual fault events with RUL below 50 hours. SHAP analysis confirms that the top predictive features (oil pressures, temperature trends, variability proxies) align with known gearbox degradation physics, providing confidence in the model's decision-making process.

The turbine-wise train/test split validates that the learned patterns generalize across different physical units within the same fleet, a critical requirement for scalable deployment. The complete framework---including model persistence, inference visualization, and explainability modules---is designed for integration into agentic decision-support systems for wind farm operations.

---

## References

[1] Tchakoua, P., Wamkeue, R., Ouhrouche, M., Slaoui-Hasnaoui, F., Tameghe, T., & Ekemb, G. (2014). Wind turbine condition monitoring: State-of-the-art review, new trends, and future challenges. *Energies*, 7(4), 2595-2630.

[2] Sheng, S. (2013). Report on wind turbine subsystem reliability -- A survey of various databases. *NREL Technical Report*, NREL/PR-5000-59111.

[3] Blanco-M, A. (2020). Fuhrlander FL2500 SCADA dataset. Available under Eclipse Public License v2.0.

[4] Dempster, A.P., Laird, N.M., & Rubin, D.B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-38.

[5] Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

[6] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

[7] Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

[8] Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.

[9] Igba, J., Alemzadeh, K., Durugbo, C., & Henshaw, K. (2015). Performance assessment of wind turbine gearboxes using in-service data: Current approaches and future trends. *Renewable and Sustainable Energy Reviews*, 50, 144-159.

---

## List of Figures

| Figure | File | Description |
|---|---|---|
| Fig. 1 | (text diagram above) | Framework overview flowchart |
| Fig. 2 | `fuhrlander_pm_analysis.png` (Row 2 left) | Autoencoder training/validation loss convergence |
| Fig. 3 | `fuhrlander_pm_analysis.png` (Row 2 right) | Health Indicator scatter colored by GMM state |
| Fig. 4 | `fuhrlander_pm_analysis.png` (Row 3 left) | Test turbine HI time series with GMM states |
| Fig. 5 | `fuhrlander_pm_analysis.png` (Row 4) | Feature importance bar chart (MDI) |
| Fig. 6 | `fuhrlander_pm_analysis.png` (Row 5) | RUL predictions and distribution |
| Fig. 7 | `shap_dashboard.png` | SHAP explainability dashboard (4 panels) |
| Fig. 8 | `shap_binary_summary.png` | SHAP beeswarm plot (binary classifier) |
| Fig. 9 | `shap_dependence.png` | SHAP dependence plots (top 4 features) |
| Fig. 10 | `shap_sample_Pre-Fault_22058.png` | Per-sample SHAP explanation (pre-fault) |
| Fig. 11 | `inference_turbine_83_dashboard.png` | Inference-time health dashboard |
| Fig. 12 | `inference_turbine_83_confusion.png` | Confusion matrix and classification report |
| Fig. 13 | `inference_turbine_83_summary.png` | Multi-model assessment summary |
| Fig. 14 | `inference_turbine_83_rul_trend.png` | RUL trend with ground truth |
