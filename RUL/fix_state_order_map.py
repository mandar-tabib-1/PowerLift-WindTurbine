"""
Fix state_order_map in saved models based on GMM cluster centers.

This script corrects the state ordering map to ensure:
- State 0 (Healthy) = cluster with lowest mean HI
- State 1 (Degrading) = cluster with medium mean HI  
- State 2 (Critical) = cluster with highest mean HI
"""

import os
import numpy as np
import joblib
from datetime import datetime

def fix_state_order_map():
    """Fix the state_order_map in saved models based on GMM cluster centers."""
    
    model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    model_path = os.path.join(model_dir, 'fuhrlander_fl2500_pm_models.joblib')
    
    print("=" * 80)
    print("FIXING GMM STATE ORDER MAP IN SAVED MODELS")
    print("=" * 80)
    
    # Load current model
    print(f"\n1. Loading model from: {model_path}")
    artifacts = joblib.load(model_path)
    
    gmm = artifacts['gmm']
    old_map = artifacts['state_order_map']
    
    print(f"\n2. Current (incorrect) state_order_map: {old_map}")
    print(f"   GMM cluster centers: {gmm.means_.flatten()}")
    
    # Compute correct mapping based on cluster centers
    cluster_means = gmm.means_.flatten()
    state_order = np.argsort(cluster_means)
    correct_map = {int(old): int(new) for new, old in enumerate(state_order)}
    
    print(f"\n3. Correct state_order_map: {correct_map}")
    print(f"   Sorted cluster means: {sorted(cluster_means)}")
    print(f"\n   Mapping explanation:")
    for raw_cluster, sorted_state in correct_map.items():
        state_labels = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
        print(f"      Raw cluster {raw_cluster} (HI={cluster_means[raw_cluster]:.4f}) → State {sorted_state} ({state_labels[sorted_state]})")
    
    # Update the state_order_map
    artifacts['state_order_map'] = correct_map
    artifacts['model_info']['state_order_map_fixed'] = datetime.now().isoformat()
    
    # Backup old model
    backup_path = model_path + '.backup'
    print(f"\n4. Creating backup: {backup_path}")
    joblib.dump(joblib.load(model_path), backup_path, compress=3)
    
    # Save corrected model
    print(f"\n5. Saving corrected model: {model_path}")
    joblib.dump(artifacts, model_path, compress=3)
    
    print("\n" + "=" * 80)
    print("✅ STATE ORDER MAP CORRECTED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOld mapping: {old_map}")
    print(f"New mapping: {correct_map}")
    print(f"\nBackup saved at: {backup_path}")
    print("\n💡 You can now run inference with the corrected state ordering.")
    print("   States will correctly assign:")
    print("   - State 0 (Healthy) = lowest HI samples")
    print("   - State 1 (Degrading) = medium HI samples")
    print("   - State 2 (Critical) = highest HI samples")
    print("=" * 80)

if __name__ == '__main__':
    fix_state_order_map()
