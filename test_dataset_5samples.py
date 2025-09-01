#!/usr/bin/env python
"""Test just dataset search with 5 samples."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_search import search_dataset_for_interactions
from fra.dataset_search_viz import create_dashboard_from_search
import time

torch.set_grad_enabled(False)

print("Testing dataset search with 5 samples...")
print("="*60)

# Load model and SAE
print("\nLoading model and SAE...")
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")
print(f"Loading time: {time.time()-t0:.1f}s")

# Use 5 predefined texts
test_texts = [
    "The cat sat on the mat. The cat was happy.",
    "Alice went to the store. She bought milk.",
    "The student who studied hard passed. The students who studied hard passed.",
    "Bob likes to play chess. Bob is very good at chess.",
    "The weather today is sunny. The weather yesterday was rainy."
]

print("\nRunning dataset search (5 samples, self-interactions filtered)...")
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
t1 = time.time()

print(f"\n⏱️ Search time: {t1-t0:.1f}s total")
print(f"⏱️ Per sample: {(t1-t0)/5:.1f}s")

print("\nCreating dashboard...")
t0 = time.time()
dataset_path = create_dashboard_from_search(
    results=search_results,
    top_k_interactions=30,
    use_timestamp=False
)
print(f"✅ Dashboard created: {dataset_path} ({time.time()-t0:.1f}s)")

print("\n" + "="*60)
print("✅ Test Complete!")
print(f"Results show sparse tensors: {search_results.avg_interactions.is_sparse}")
print(f"Found {len(search_results.top_examples)} unique feature pairs")