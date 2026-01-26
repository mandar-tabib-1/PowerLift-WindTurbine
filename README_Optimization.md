# Wind Farm Wake Steering Optimization

## Overview

This module implements gradient-based optimization for **two-turbine wind farm wake steering** using PyTorch Automatic Differentiation (AD). The optimizer finds optimal yaw misalignment angles for each turbine to maximize total farm power output.

## Problem Formulation

### Objective
Maximize total farm power:
$$P_{total}(\gamma_1, \gamma_2) = P_1(\gamma_1) + P_2(\gamma_1, \gamma_2)$$

where:
- $\gamma_1$ = upstream turbine yaw misalignment (0° to 12°)
- $\gamma_2$ = downstream turbine yaw misalignment (0° to 12°)

### Physics
1. **Yaw Power Loss**: Power decreases with yaw misalignment
   $$P = P_{base} \cdot \cos^3(\gamma)$$

2. **Wake Steering** (Bastankhah & Porté-Agel, 2016): Yaw deflects the wake laterally
   $$\delta_{eff} = \delta_{base} \cdot (1 - k \cdot \sin(\gamma) \cdot \cos^2(\gamma))$$
   where $k = 0.6$ is the steering effectiveness coefficient.

3. **Downstream Power**: Affected by both wake deficit and its own yaw
   $$P_2 = P_{base} \cdot (1 - \delta_{eff})^3 \cdot \cos^3(\gamma_2)$$

---

## Optimization Methods

### Method 1: True AD through ML Surrogates (Recommended)

**File**: `wind_turbine_gui.py` → `DifferentiablePowerSurrogate`, `DifferentiableWakeSurrogate`

This method creates **differentiable polynomial surrogates** fitted to the actual ML model predictions, enabling genuine gradient-based optimization through the ML models.

#### How It Works

1. **Surrogate Construction**:
   ```python
   # Sample GP power model at multiple yaw angles
   yaw_samples = np.linspace(270, 285, 16)  # 16 points
   power_samples = [power_agent.predict(yaw)['power_mean_MW'].mean() for yaw in yaw_samples]
   
   # Fit cubic polynomial: P(γ) = a₀ + a₁γ + a₂γ² + a₃γ³
   coeffs = torch.linalg.lstsq(A, power_samples).solution
   ```

2. **Differentiable Forward Pass**:
   ```python
   class DifferentiablePowerSurrogate(torch.nn.Module):
       def forward(self, nacelle_direction):
           yaw_norm = (nacelle_direction - 270) / 15  # Normalize to [0,1]
           power = (self.coeffs[0] + 
                    self.coeffs[1] * yaw_norm + 
                    self.coeffs[2] * yaw_norm**2 + 
                    self.coeffs[3] * yaw_norm**3)
           return power
   ```

3. **Gradient Computation**: PyTorch AD computes exact gradients through the polynomial:
   $$\frac{\partial P}{\partial \gamma} = \frac{1}{15}(a_1 + 2a_2\tilde{\gamma} + 3a_3\tilde{\gamma}^2)$$

#### Advantages
- ✅ Gradients reflect actual ML model behavior
- ✅ Captures non-linear relationships learned from CFD data
- ✅ More accurate when ML models capture physics better than analytical models

#### Trade-offs
- ⏱️ Requires initial surrogate fitting (~2-5 seconds)
- 📊 Polynomial approximation introduces small fitting error

---

### Method 2: AD through Analytical Physics

This method uses **analytical physics equations** that are directly differentiable, while ML models provide baseline values as constants.

#### How It Works

1. **ML Models as Constants**:
   ```python
   # Evaluate ML models once at baseline (270°)
   P_base = power_agent.predict(270)['power_mean_MW'].mean()  # Constant
   δ_base = compute_wake_deficit(wake_agent, 270)             # Constant
   ```

2. **Differentiable Physics**:
   ```python
   def compute_farm_power_physics(misalignment_tensor):
       γ1, γ2 = misalignment_tensor[0], misalignment_tensor[1]
       
       # Upstream power (differentiable)
       P1 = P_base * torch.cos(γ1 * π/180)**3
       
       # Wake steering (differentiable)
       steering = k * torch.sin(γ1 * π/180) * torch.cos(γ1 * π/180)**2
       δ_eff = δ_base * (1 - steering)
       
       # Downstream power (differentiable)
       P2 = P_base * (1 - δ_eff)**3 * torch.cos(γ2 * π/180)**3
       
       return P1 + P2
   ```

#### Advantages
- ✅ Fast (no surrogate fitting needed)
- ✅ Physics-interpretable gradients
- ✅ Well-validated analytical models

#### Trade-offs
- ⚠️ ML models are treated as constants (not differentiated through)
- ⚠️ May miss complex relationships captured by ML models

---

### Method 3: Grid Search (Reference)

Brute-force evaluation on a discrete grid. No gradients used.

```python
misalign_range = [0, 2, 4, 6, 8, 10, 12]  # degrees
for γ1 in misalign_range:
    for γ2 in [0, 2, 4]:
        P_total = evaluate(γ1, γ2)
```

- **Grid**: 7 × 3 = 21 combinations
- **Advantage**: Guaranteed global optimum on grid
- **Disadvantage**: Limited to grid resolution

---

## Optimizer Configuration

### Adam Optimizer
```python
params = torch.tensor([4.0, 0.0], dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([params], lr=1.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
```

### Box Constraints
Parameters are projected to feasible region after each step:
```python
with torch.no_grad():
    params.clamp_(min=torch.tensor([0.0, 0.0]), 
                  max=torch.tensor([12.0, 12.0]))
```

### Convergence Criteria
- Maximum iterations: 50
- Early stopping: gradient norm < 10⁻⁴

---

## Yaw Angle Convention

### ⚠️ Important: ML Model Input is a Proxy for Yaw Misalignment

The ML models (GP Power Predictor and TT-OpInf Wake ROM) were trained with **nacelle direction angles from 270° to 285°**. This is a **proxy** for the actual physical quantity of interest: **yaw misalignment from 0° to 15°**.

**Key Insight**: The nacelle direction used as ML model input does NOT represent the actual turbine heading. Instead:
- **270° nacelle direction** = 0° yaw misalignment (turbine aligned with wind)
- **285° nacelle direction** = 15° yaw misalignment (maximum training range)

Therefore, in the GUI and all outputs, we report results in terms of **yaw misalignment** (the physically meaningful quantity), and then compute the **actual yaw angle** as:

$$\text{Actual Yaw Angle} = \text{Wind Direction (from Agent 1)} + \text{Yaw Misalignment (from Optimizer)}$$

### Conversion Table

| ML Model Input | Yaw Misalignment | Description |
|----------------|------------------|-------------|
| 270° | 0° | Aligned with wind |
| 275° | 5° | Mild wake steering |
| 280° | 10° | Moderate steering |
| 282° | 12° | Aggressive steering |
| 285° | 15° | Maximum (training limit) |

### Conversions

```python
# ML model input from yaw misalignment
nacelle_direction_for_ml = 270° + yaw_misalignment

# Actual turbine yaw angle for real-world operation
actual_yaw_angle = wind_direction + yaw_misalignment
```

**Example**: If wind direction is 290° and optimal yaw misalignment is 8°:
- ML model input: 270° + 8° = 278°
- Actual turbine yaw setting: 290° + 8° = 298°

---

## ML Models Used

| Model | Type | Input | Output |
|-------|------|-------|--------|
| **Power Predictor** | Gaussian Process Regressor | Nacelle direction (270°-285°) | Time-varying rotor power (MW) |
| **Wake Flow ROM** | TT-OpInf (Tensor Decomposition + Operator Inference) | Nacelle direction (270°-285°) | 3D velocity field over time |

Both models were trained at SINTEF on high-fidelity CFD simulations of the NREL 5MW reference turbine.

---

## Usage

```python
from wind_turbine_gui import optimize_two_turbine_farm, load_agents

# Load ML models
power_agent, wake_agent = load_agents()

# Method 1: True AD through ML surrogates
results = optimize_two_turbine_farm(
    power_agent=power_agent,
    wake_agent=wake_agent,
    turbine_spacing_D=7.0,
    n_timesteps=30,
    optimization_method='ml_surrogate'  # or 'analytical_physics' or 'grid_search'
)

# Results
print(f"Optimal upstream yaw: {results['optimal_upstream_misalignment']:.1f}°")
print(f"Optimal downstream yaw: {results['optimal_downstream_misalignment']:.1f}°")
print(f"Power gain: {results['power_gain_percent']:.1f}%")
```

---

## Output Structure

```python
{
    'optimal_upstream_misalignment': 8.0,      # degrees
    'optimal_downstream_misalignment': 0.0,    # degrees
    'optimal_upstream_nacelle': 278.0,         # degrees
    'optimal_downstream_nacelle': 270.0,       # degrees
    'optimal_upstream_power': 4.85,            # MW
    'optimal_downstream_power': 3.92,          # MW
    'optimal_total_power': 8.77,               # MW
    'baseline_total_power': 8.45,              # MW (both at 0°)
    'power_gain_MW': 0.32,                     # MW
    'power_gain_percent': 3.8,                 # %
    'optimization_method': 'ML Surrogate AD',
    'all_results': [...]                       # Iteration history
}
```

---

## References

1. **Wake Steering Model**: Bastankhah, M., & Porté-Agel, F. (2016). Experimental and theoretical study of wind turbine wakes in yawed conditions. *Journal of Fluid Mechanics*, 806, 506-541.

2. **NREL 5MW Turbine**: Jonkman, J., et al. (2009). Definition of a 5-MW reference wind turbine for offshore system development. *NREL Technical Report*, NREL/TP-500-38060.

3. **TT-OpInf ROM**: Tensor-Train decomposition with Operator Inference for parametric model order reduction.

---

## Contact

📧 mandar.tabib@sintef.no

*Developed at SINTEF Energy Research*
