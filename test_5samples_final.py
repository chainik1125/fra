#!/usr/bin/env python
"""Final test with 5 samples using all optimizations."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.data_ind_viz import create_data_independent_dashboard
from fra.dataset_search import search_dataset_for_interactions
from fra.dataset_search_viz import create_dashboard_from_search
from fra.single_sample_viz import create_fra_dashboard
import tarfile
from pathlib import Path
import time

torch.set_grad_enabled(False)

print("Final test with 5 samples (all optimizations)...")
print("="*60)

# Load model and SAE once
print("\n1. Loading model and SAE...")
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")
print(f"   Loading time: {time.time()-t0:.1f}s")

# Use 5 predefined texts
test_texts = [
    "The cat sat on the mat. The cat was happy.",
    "Alice went to the store. She bought milk.",
    "The student who studied hard passed. The students who studied hard passed.",
    "Bob likes to play chess. Bob is very good at chess.",
    "The weather today is sunny. The weather yesterday was rainy."
]

print("\n2. Generating data-independent dashboard...")
t0 = time.time()
data_ind_path = create_data_independent_dashboard(
    model=model,
    sae=sae,
    layer=5,
    head=0,
    top_k_features=20,
    top_k_interactions=30,
    use_timestamp=False
)
print(f"   ✅ Created: {Path(data_ind_path).name} ({time.time()-t0:.1f}s)")

print("\n3. Running dataset search (5 samples, filtered self-interactions)...")
t0 = time.time()
search_results = search_dataset_for_interactions(
    model=model,
    sae=sae,
    dataset_texts=test_texts,
    layer=5,
    head=0,
    num_samples=5,
    top_k_per_pair=3,
    filter_self_interactions=True,
    verbose=True
)
print(f"   Search time: {time.time()-t0:.1f}s")

print("\n4. Creating dataset search dashboard...")
t0 = time.time()
dataset_path = create_dashboard_from_search(
    results=search_results,
    top_k_interactions=30,
    use_timestamp=False
)
print(f"   ✅ Created: {Path(dataset_path).name} ({time.time()-t0:.1f}s)")

print("\n5. Generating single sample dashboard...")
t0 = time.time()
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
print(f"   ✅ Created: {Path(single_path).name} ({time.time()-t0:.1f}s)")

print("\n6. Creating package...")
t0 = time.time()
results_dir = Path("fra/results")
package_path = results_dir / "results_package.tar.gz"

with tarfile.open(package_path, "w:gz") as tar:
    html_files = list(results_dir.glob("*.html"))
    for html_file in html_files:
        tar.add(html_file, arcname=html_file.name)
    print(f"   Packaged {len(html_files)} dashboards")

print(f"   ✅ Package created: {package_path} ({time.time()-t0:.1f}s)")
print(f"   📥 Download: scp remote:{package_path.absolute()} ./")

print("\n" + "="*60)
print("🎉 Test Complete!")
print("="*60)
print("\nSummary:")
print("  ✅ Data-independent FRA (structural patterns)")
print("  ✅ Dataset search with 5 samples (self-interactions filtered)")
print("  ✅ Single sample analysis")
print("  ✅ All dashboards packaged in results_package.tar.gz")
print("\nKey optimizations:")
print("  • Sparse tensors throughout (no dense conversions)")
print("  • Dictionary-based accumulation")
print("  • Optimized sorting (only at end)")
print("  • Memory efficient processing (~2-3s per sample)")

# Clean up GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("\n✨ GPU memory cleared")