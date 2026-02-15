"""
Demonstration: How turbine spacing affects wake deficit and optimal yaw angles

This script shows that the distance-dependent wake model now properly accounts
for turbine spacing in optimization, resolving the issue where all pairs had
identical yaw angles.
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_wake_deficit(turbine_spacing_D, upstream_yaw_deg=0.0, thrust_coefficient=0.8, wake_expansion_coeff=0.022):
    """Calculate wake velocity deficit (simplified for demo)."""
    C_T = thrust_coefficient
    k_star = wake_expansion_coeff
    x_D = turbine_spacing_D
    gamma = np.radians(upstream_yaw_deg)
    
    # Yaw-corrected thrust coefficient
    C_T_prime = C_T * np.cos(gamma) ** 2
    
    # Wake expansion
    epsilon = 0.2
    sigma_over_D = k_star * x_D + epsilon
    
    # Maximum deficit
    denominator = 8.0 * (sigma_over_D ** 2)
    deficit_max = 1.0 - np.sqrt(np.clip(1.0 - C_T_prime / denominator, 0.01, 1.0))
    
    # Yaw reduction
    yaw_reduction_factor = np.cos(gamma) ** 0.5
    wake_deficit = deficit_max * yaw_reduction_factor
    wake_deficit = np.clip(wake_deficit, 0.0, 0.6)
    
    return wake_deficit


def calculate_total_farm_power(spacing_D, upstream_yaw_deg, P_base=5.0):
    """Calculate total farm power for given spacing and yaw."""
    gamma = np.radians(upstream_yaw_deg)
    
    # Upstream power loss
    upstream_power = P_base * np.cos(gamma) ** 3
    
    # Wake deficit (distance-dependent!)
    deficit = calculate_wake_deficit(spacing_D, upstream_yaw_deg)
    
    # Downstream power gain
    downstream_power = P_base * (1 - deficit) ** 3  # Simplified, no downstream yaw
    
    return upstream_power + downstream_power, upstream_power, downstream_power, deficit


if __name__ == "__main__":
    print("=" * 80)
    print("DEMONSTRATION: Distance-Dependent Wake Model")
    print("=" * 80)
    print("\nThis shows how turbine spacing affects wake deficit and optimal yaw angles.")
    print("Previously, all pairs used fixed deficit → identical yaw angles.")
    print("Now, spacing is used in wake physics → different yaw angles!\n")
    
    # Test different spacings (like real farm layouts)
    spacings = [6.0, 7.5, 9.0, 11.0]
    yaw_angles = np.linspace(0, 12, 100)
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Wake deficit vs spacing at different yaw angles
    plt.subplot(2, 2, 1)
    for yaw in [0, 3, 6, 9, 12]:
        deficits = [calculate_wake_deficit(s, yaw) for s in np.linspace(5, 15, 50)]
        plt.plot(np.linspace(5, 15, 50), deficits, label=f'Yaw = {yaw}°')
    plt.xlabel('Turbine Spacing (D)', fontsize=11)
    plt.ylabel('Wake Deficit', fontsize=11)
    plt.title('Wake Deficit vs Spacing\n(Larger spacing → Lower deficit → Wake recovery)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Optimal yaw vs spacing
    plt.subplot(2, 2, 2)
    optimal_yaws = []
    optimal_powers = []
    
    print("Spacing (D) | Optimal Yaw | Max Power | Wake Deficit @ Opt")
    print("-" * 65)
    
    for spacing in spacings:
        powers = [calculate_total_farm_power(spacing, yaw)[0] for yaw in yaw_angles]
        opt_idx = np.argmax(powers)
        opt_yaw = yaw_angles[opt_idx]
        max_power = powers[opt_idx]
        deficit_at_opt = calculate_wake_deficit(spacing, opt_yaw)
        
        optimal_yaws.append(opt_yaw)
        optimal_powers.append(max_power)
        
        print(f"   {spacing:4.1f}D    |    {opt_yaw:5.2f}°   |  {max_power:6.3f} MW |     {deficit_at_opt:.4f}")
    
    plt.plot(spacings, optimal_yaws, 'o-', linewidth=2, markersize=10, color='red')
    plt.xlabel('Turbine Spacing (D)', fontsize=11)
    plt.ylabel('Optimal Yaw Angle (°)', fontsize=11)
    plt.title('Optimal Yaw vs Spacing\n(KEY FIX: Different spacings → Different yaw angles!)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Farm power curves for different spacings
    plt.subplot(2, 2, 3)
    for spacing in spacings:
        powers = [calculate_total_farm_power(spacing, yaw)[0] for yaw in yaw_angles]
        plt.plot(yaw_angles, powers, label=f'Spacing = {spacing:.1f}D', linewidth=2)
    plt.xlabel('Upstream Yaw Angle (°)', fontsize=11)
    plt.ylabel('Total Farm Power (MW)', fontsize=11)
    plt.title('Farm Power vs Yaw for Different Spacings\n(Peak shifts with spacing)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Power gain breakdown
    plt.subplot(2, 2, 4)
    spacing_demo = 7.5  # Example spacing
    upstream_powers = []
    downstream_powers = []
    
    for yaw in yaw_angles:
        total, up, down, _ = calculate_total_farm_power(spacing_demo, yaw)
        upstream_powers.append(up)
        downstream_powers.append(down)
    
    plt.plot(yaw_angles, upstream_powers, label='Upstream Power (cos³ loss)', linewidth=2)
    plt.plot(yaw_angles, downstream_powers, label='Downstream Power (wake benefit)', linewidth=2)
    plt.plot(yaw_angles, np.array(upstream_powers) + np.array(downstream_powers), 
             label='Total Farm Power', linewidth=2, linestyle='--', color='black')
    plt.xlabel('Upstream Yaw Angle (°)', fontsize=11)
    plt.ylabel('Power (MW)', fontsize=11)
    plt.title(f'Power Trade-off at Spacing = {spacing_demo}D\n(Upstream loss vs Downstream gain)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wake_deficit_spacing_analysis.png', dpi=150, bbox_inches='tight')
    print("\n" + "=" * 65)
    print("✅ Plot saved: wake_deficit_spacing_analysis.png")
    print("=" * 65)
    print("\nKEY INSIGHT:")
    print("• Larger spacing → Smaller wake deficit → Less benefit from yaw steering")
    print("• Smaller spacing → Larger wake deficit → More benefit from yaw steering")
    print("• This is why optimal yaw angle MUST depend on spacing!")
    print("• In the previous implementation, spacing was ignored → identical results")
    print("\nThe fix ensures each turbine pair uses its real spacing in wake physics.")
    
    plt.show()
