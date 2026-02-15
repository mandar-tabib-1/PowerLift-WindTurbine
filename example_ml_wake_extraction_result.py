"""
Example visualization showing expected output from ML Wake Extraction method
"""
import matplotlib.pyplot as plt
import numpy as np

# Simulate data for a turbine pair at 7.5D spacing (945m)
yaw_angles = np.array([0.0, 2.14, 4.29, 6.43, 8.57, 10.71, 12.86, 15.0])

# Lateral wake migration increases with yaw angle
lateral_migrations = np.array([0.0, 8.3, 16.1, 23.5, 30.2, 36.5, 42.1, 47.2])

# Upstream power decreases (cos³ loss)
upstream_powers = np.array([5.15, 5.14, 5.11, 5.06, 4.98, 4.88, 4.76, 4.61])

# Downstream power increases as wake misses
downstream_powers = np.array([3.084, 3.147, 3.246, 3.381, 3.418, 3.398, 3.362, 3.310])

# Total power
total_powers = upstream_powers + downstream_powers

# Baseline at 0° yaw
baseline_total_power = total_powers[0]

# Power gains
power_gain_pcts = (total_powers - baseline_total_power) / baseline_total_power * 100

# Optimal yaw (peak power at ~6.43°)
optimal_yaw = 6.43
turbine_spacing_D = 7.5

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Lateral Migration
ax1 = axes[0]
ax1.plot(yaw_angles, lateral_migrations, 'bo-', linewidth=2, markersize=8, label='Lateral Migration')
ax1.axvline(optimal_yaw, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal Yaw: {optimal_yaw:.1f}°')
ax1.set_xlabel('Upstream Yaw Misalignment (°)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Lateral Wake Migration (m)', fontsize=12, fontweight='bold')
ax1.set_title(f'Wake Deflection vs Yaw Angle\nTurbine Spacing: {turbine_spacing_D:.1f}D', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Subplot 2: Power Components
ax2 = axes[1]
ax2.plot(yaw_angles, upstream_powers, 'b^-', linewidth=2, markersize=8, label='Upstream Power')
ax2.plot(yaw_angles, downstream_powers, 'gs-', linewidth=2, markersize=8, label='Downstream Power')
ax2.plot(yaw_angles, total_powers, 'ro-', linewidth=3, markersize=10, label='Total Farm Power')
ax2.axhline(baseline_total_power, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Baseline (0° yaw)')
ax2.axvline(optimal_yaw, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal: {optimal_yaw:.1f}°')
ax2.set_xlabel('Upstream Yaw Misalignment (°)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
ax2.set_title('Power Generation vs Yaw Angle', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='best')

# Subplot 3: Power Gain
ax3 = axes[2]
colors = ['green' if pg > 0 else 'red' for pg in power_gain_pcts]
ax3.bar(yaw_angles, power_gain_pcts, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.axvline(optimal_yaw, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal: {optimal_yaw:.1f}°')
ax3.set_xlabel('Upstream Yaw Misalignment (°)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Power Gain (%)', fontsize=12, fontweight='bold')
ax3.set_title('Farm Power Gain vs Baseline', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(fontsize=10)

plt.tight_layout()
plt.savefig('example_ml_wake_extraction_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Example visualization saved to: example_ml_wake_extraction_visualization.png")
print("\nExpected Results:")
print(f"  Turbine Spacing: {turbine_spacing_D:.1f}D (945m)")
print(f"  Optimal Yaw: {optimal_yaw:.1f}°")
print(f"  Baseline Power: {baseline_total_power:.3f} MW")
print(f"  Optimal Power: {total_powers[3]:.3f} MW")
print(f"  Power Gain: {total_powers[3] - baseline_total_power:.3f} MW ({power_gain_pcts[3]:.2f}%)")
print(f"\n  Key Observation:")
print(f"    - Wake migrates {lateral_migrations[3]:.1f}m at optimal yaw")
print(f"    - Upstream loses {upstream_powers[0] - upstream_powers[3]:.3f} MW (cos³ loss)")
print(f"    - Downstream gains {downstream_powers[3] - downstream_powers[0]:.3f} MW (reduced wake)")
print(f"    - Net gain: {(total_powers[3] - baseline_total_power)*1000:.1f} kW")
plt.show()
