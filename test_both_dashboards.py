#!/usr/bin/env python
"""Test both data-dependent and data-independent dashboards with packaging."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.single_sample_viz import create_fra_dashboard
from fra.data_ind_viz import create_data_independent_dashboard, create_package

torch.set_grad_enabled(False)

print("Testing both dashboard types...")

# Load model and SAE once
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

# Test text
text = "The cat sat on the mat. The dog sat on the rug."

# Generate data-dependent dashboard
print("\n1. Generating data-dependent dashboard (with text)...")
dashboard1 = create_fra_dashboard(
    model=model,
    sae=sae,
    text=text,
    layer=5,
    head=0,
    top_k_features=10,
    top_k_interactions=20,
    use_timestamp=False
)
print(f"   ✅ Saved: {dashboard1}")

# Generate data-independent dashboard
print("\n2. Generating data-independent dashboard (without text)...")
dashboard2 = create_data_independent_dashboard(
    model=model,
    sae=sae,
    layer=5,
    head=0,
    top_k_features=10,
    top_k_interactions=20,
    use_timestamp=False
)
print(f"   ✅ Saved: {dashboard2}")

# Create package
print("\n3. Creating tar.gz package...")
package_path = create_package()

print("\n" + "="*50)
print("✅ Both dashboards generated successfully!")
print("📦 Package includes:")
print("   - Data-dependent FRA (with text highlighting)")
print("   - Data-independent FRA (structural patterns only)")
print(f"📥 Download: scp remote:{package_path} ./")