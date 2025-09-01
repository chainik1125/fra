#!/usr/bin/env python
"""Test dataset search for feature interactions."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_search import run_dataset_search
from fra.dataset_search_viz import create_dashboard_from_search
from fra.data_ind_viz import create_package

torch.set_grad_enabled(False)

print("Testing dataset search for feature interactions...")
print("="*50)

# Test with a small number of samples for quick testing
print("\n1. Running dataset search...")
results = run_dataset_search(
    layer=5,
    head=0,
    num_samples=10,  # 10 samples for quick test
    save_path="fra/results/dataset_search_test.pkl"
)

print("\n2. Creating visualization dashboard...")
dashboard_path = create_dashboard_from_search(
    results=results,
    top_k_interactions=20,
    use_timestamp=False
)

print("\n3. Creating package...")
package_path = create_package()

print("\n" + "="*50)
print("✅ Dataset search test complete!")
print(f"📊 Dashboard: {dashboard_path}")
print(f"📦 Package: {package_path}")
print("\nThe dashboard shows:")
print("  - Top feature interactions averaged across the dataset")
print("  - Number of times each interaction occurred")
print("  - Specific text examples where interactions were strongest")
print("  - Token highlighting showing where features activate")