#!/usr/bin/env python
"""Simple test of dataset search with predefined texts."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_search import search_dataset_for_interactions
from fra.dataset_search_viz import create_dashboard_from_search

torch.set_grad_enabled(False)

print("Simple dataset search test...")
print("="*50)

# Use predefined texts instead of loading dataset
test_texts = [
    "The cat sat on the mat. The cat was happy.",
    "Alice went to the store. She bought milk.",
    "The student who studied hard passed. The students who studied hard passed.",
]

print("\n1. Loading model and SAE...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

print("\n2. Running search on 3 samples...")
results = search_dataset_for_interactions(
    model=model,
    sae=sae,
    dataset_texts=test_texts,
    layer=5,
    head=0,
    num_samples=3,
    top_k_per_pair=2,
    batch_size=1,
    verbose=True
)

print("\n3. Creating dashboard...")
dashboard_path = create_dashboard_from_search(
    results=results,
    top_k_interactions=10,
    use_timestamp=False
)

print("\n" + "="*50)
print(f"✅ Test complete!")
print(f"📊 Dashboard: {dashboard_path}")
print(f"\nFound {len(results.top_examples)} unique feature interactions")