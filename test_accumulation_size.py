#!/usr/bin/env python
"""Check the actual size of accumulated tensors."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_average_sparse import compute_dataset_average_sparse
from fra.utils import load_dataset_hf
import time

torch.set_grad_enabled(False)

print("="*60)
print("Testing Accumulation Tensor Sizes")
print("="*60)

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', 'blocks.5.hook_z', device='cuda')

# Load just 3 samples to check sizes
print("\nLoading 3 samples...")
dataset = load_dataset_hf(streaming=True)
dataset_texts = []
for i, item in enumerate(dataset):
    if i >= 3:
        break
    text = item.get('text', '') if isinstance(item, dict) else str(item)
    if len(text.split()) > 10:
        dataset_texts.append(text[:500])

print(f"Processing {len(dataset_texts)} texts...")

# Run accumulation
t0 = time.time()
results = compute_dataset_average_sparse(
    model=model,
    sae=sae,
    dataset_texts=dataset_texts,
    layer=5,
    head=0,
    filter_self_interactions=False,
    use_absolute_values=True,
    verbose=True
)
t1 = time.time()

print(f"\nAccumulation time: {t1-t0:.1f}s")

# Check tensor sizes
print("\n" + "="*60)
print("Checking accumulated tensor sizes...")
print("="*60)

sum_sparse = results['interaction_sum_sparse']
count_sparse = results['interaction_count_sparse']

print(f"Sum tensor: {sum_sparse.shape}")
print(f"  nnz before coalesce: {sum_sparse._nnz()}")

t0 = time.time()
sum_coalesced = sum_sparse.coalesce()
print(f"  Coalesce time: {time.time()-t0:.2f}s")
print(f"  nnz after coalesce: {sum_coalesced._nnz()}")

print(f"\nCount tensor: {count_sparse.shape}")
print(f"  nnz before coalesce: {count_sparse._nnz()}")

t0 = time.time()
count_coalesced = count_sparse.coalesce()
print(f"  Coalesce time: {time.time()-t0:.2f}s")
print(f"  nnz after coalesce: {count_coalesced._nnz()}")

# Test extraction
print("\nExtracting top pairs...")
t0 = time.time()

# Get values
sum_vals = sum_coalesced.values()
count_vals = count_coalesced.values()

# Compute averages
avg_vals = sum_vals / count_vals

# Get top 100
k = min(100, len(avg_vals))
top_vals, top_idx = torch.topk(avg_vals, k)

print(f"  Extraction time: {time.time()-t0:.2f}s")
print(f"  Top average: {top_vals[0]:.4f}")
print(f"  10th average: {top_vals[9]:.4f}" if len(top_vals) > 9 else "")

print("\n" + "="*60)
print("Summary:")
print(f"  Unique feature pairs: {sum_coalesced._nnz():,}")
print(f"  Tensor dimensions: {sum_sparse.shape}")
print(f"  Density: {sum_coalesced._nnz() / (49152**2) * 100:.6f}%")