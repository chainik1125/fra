#!/usr/bin/env python
"""Quick test with 2 samples and predefined texts."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.data_ind_viz import create_data_independent_dashboard
from fra.dataset_search import search_dataset_for_interactions, DatasetSearchResults
from fra.dataset_search_viz import create_dashboard_from_search
from fra.single_sample_viz import create_fra_dashboard
import tarfile
from pathlib import Path

torch.set_grad_enabled(False)

print("Quick unified test with 2 samples...")
print("="*60)

# Load model and SAE once
print("\n1. Loading model and SAE...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

# Use simple predefined texts for quick testing
test_texts = [
    "The cat sat on the mat. The cat was happy.",
    "Alice went to the store. She bought milk."
]

print("\n2. Generating data-independent dashboard...")
data_ind_path = create_data_independent_dashboard(
    model=model,
    sae=sae,
    layer=5,
    head=0,
    top_k_features=20,
    top_k_interactions=30,
    use_timestamp=False
)
print(f"✅ Created: {Path(data_ind_path).name}")

print("\n3. Running dataset search (2 samples, filtered self-interactions)...")
search_results = search_dataset_for_interactions(
    model=model,
    sae=sae,
    dataset_texts=test_texts,
    layer=5,
    head=0,
    num_samples=2,
    top_k_per_pair=3,
    filter_self_interactions=True,  # Filter out self-interactions
    verbose=True
)

dataset_path = create_dashboard_from_search(
    results=search_results,
    top_k_interactions=20,
    use_timestamp=False
)
print(f"✅ Created: {Path(dataset_path).name}")

print("\n4. Generating single sample dashboard...")
single_path = create_fra_dashboard(
    model=model,
    sae=sae,
    text="The student who studied hard passed the exam.",
    layer=5,
    head=0,
    top_k_features=10,
    top_k_interactions=15,
    use_timestamp=False
)
print(f"✅ Created: {Path(single_path).name}")

print("\n5. Creating package...")
results_dir = Path("fra/results")
package_path = results_dir / "results_package.tar.gz"

with tarfile.open(package_path, "w:gz") as tar:
    for html_file in results_dir.glob("*.html"):
        tar.add(html_file, arcname=html_file.name)
        print(f"  Added: {html_file.name}")

print(f"\n✅ Package created: {package_path}")
print(f"📥 Download: scp remote:{package_path.absolute()} ./")

print("\n" + "="*60)
print("🎉 Quick Test Complete!")
print("="*60)
print("\nKey features demonstrated:")
print("  ✅ Data-independent FRA (structural patterns)")
print("  ✅ Dataset search with self-interaction filtering")
print("  ✅ Single sample analysis")
print("  ✅ All packaged in results_package.tar.gz")