#!/usr/bin/env python
"""Test optimized dataset search."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_search import run_dataset_search
import time

torch.set_grad_enabled(False)

print("Testing optimized dataset search with 5 samples...")
print("="*60)

# Load model and SAE
print("\nLoading model...")
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
print(f"Model loaded in {time.time()-t0:.1f}s")

print("\nLoading SAE...")
t0 = time.time()
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")
print(f"SAE loaded in {time.time()-t0:.1f}s")

# Run search on 5 samples
print("\n" + "="*60)
print("Running dataset search on 5 samples...")
print("="*60)

t0 = time.time()
results = run_dataset_search(
    model=model,
    sae=sae,
    layer=5,
    head=0,
    num_samples=5,
    filter_self_interactions=True
)
t1 = time.time()

print(f"\n⏱️ Total time: {t1-t0:.1f}s")
print(f"⏱️ Per sample: {(t1-t0)/5:.1f}s")
print(f"\n✅ Target: ~3-4s per sample")
print(f"✅ Achieved: {(t1-t0)/5:.1f}s per sample")