#!/usr/bin/env python
"""Quick test of data-independent FRA visualization."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.data_ind_viz import create_data_independent_dashboard

torch.set_grad_enabled(False)

print("Testing data-independent FRA visualization...")

# Load model and SAE
print("Loading model and SAE...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

# Generate dashboard
print("\nGenerating data-independent dashboard...")
dashboard_path = create_data_independent_dashboard(
    model=model,
    sae=sae,
    layer=5,
    head=0,
    top_k_features=20,
    top_k_interactions=30,
    use_timestamp=False
)

print(f"\n✅ Dashboard saved to: {dashboard_path}")
print("This shows the inherent feature interaction patterns without any input data.")