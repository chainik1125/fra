#!/usr/bin/env python
"""Test actual timing of the full pipeline."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_search import search_dataset_for_interactions
import time

torch.set_grad_enabled(False)

print("Testing actual timing with single sample...")

# Load model and SAE
print("Loading model and SAE...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

test_texts = ["The cat sat on the mat. The cat was happy."]

print("\nRunning search with 1 sample...")
t0 = time.time()

# Run the actual search function
search_results = search_dataset_for_interactions(
    model=model,
    sae=sae,
    dataset_texts=test_texts,
    layer=5,
    head=0,
    num_samples=1,
    top_k_per_pair=3,
    filter_self_interactions=True,
    verbose=True
)

t1 = time.time()

print(f"\n⏱️ ACTUAL time for 1 sample: {t1-t0:.1f}s")
print(f"Results are sparse: {search_results.avg_interactions.is_sparse}")

if t1-t0 > 20:
    print("\n⚠️ WAY TOO SLOW! Something is wrong beyond just accumulation.")
    print("Check if we're accidentally densifying or doing something expensive.")
else:
    print("\n✅ Timing is reasonable (~11-15s expected)")