#!/usr/bin/env python
"""Run sparse accumulation on 1000 samples with absolute values."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_average_sparse import compute_dataset_average_sparse
from fra.utils import load_dataset_hf
import time

torch.set_grad_enabled(False)

print("="*60)
print("Running FRA accumulation on 1000 samples")
print("="*60)

# Load model and SAE
print("\nLoading model...")
t0 = time.time()
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
print(f"Model loaded in {time.time()-t0:.1f}s")

print("\nLoading SAE...")
t0 = time.time()
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', 'blocks.5.hook_z', device='cuda')
print(f"SAE loaded in {time.time()-t0:.1f}s")

# Load 1000 samples from dataset
print("\nLoading 1000 samples from dataset...")
t0 = time.time()
dataset = load_dataset_hf(streaming=True)
dataset_texts = []
for i, item in enumerate(dataset):
    if i >= 100:
        break
    text = item.get('text', '') if isinstance(item, dict) else str(item)
    if len(text.split()) > 10:
        dataset_texts.append(text[:500])
print(f"Loaded {len(dataset_texts)} texts in {time.time()-t0:.1f}s")

# Run accumulation with absolute values
print("\n" + "="*60)
print("Starting FRA accumulation with absolute values...")
print("="*60)
t0 = time.time()

results = compute_dataset_average_sparse(
    model=model,
    sae=sae,
    dataset_texts=dataset_texts,
    layer=5,
    head=0,
    filter_self_interactions=True,
    use_absolute_values=True,  # Use absolute values
    verbose=True
)

t1 = time.time()

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print(f"Total time: {t1-t0:.1f}s")
print(f"Per sample: {(t1-t0)/len(dataset_texts):.1f}s")
print(f"Estimated for 1000: {1.8*1000/60:.1f} minutes")