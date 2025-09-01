#!/usr/bin/env python
"""Test script for dashboard integration."""

import torch
import tarfile
from pathlib import Path
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.single_sample_viz import create_fra_dashboard

torch.set_grad_enabled(False)

print("Testing dashboard integration...")

# Load model and SAE
print("Loading model and SAE...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

# Test text
text = "The cat sat on the mat. The cat was happy."

# Generate dashboard
print("\nGenerating dashboard...")
dashboard_path = create_fra_dashboard(
    model=model,
    sae=sae,
    text=text,
    layer=5,
    head=0,
    top_k_features=10,
    top_k_interactions=20,
    use_timestamp=False
)
print(f"Dashboard saved to: {dashboard_path}")

# Create tar.gz package
results_dir = Path("fra/results")
package_path = results_dir / "results_package.tar.gz"

print(f"\nCreating package...")
with tarfile.open(package_path, "w:gz") as tar:
    for html_file in results_dir.glob("*.html"):
        tar.add(html_file, arcname=html_file.name)
        print(f"  Added: {html_file.name}")

print(f"\n✅ Package created: {package_path}")
print(f"📥 To download: scp remote:{package_path.absolute()} ./")

# Clean up
if torch.cuda.is_available():
    torch.cuda.empty_cache()