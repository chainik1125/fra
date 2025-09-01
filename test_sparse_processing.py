#!/usr/bin/env python
"""Test sparse tensor processing to find the bottleneck."""

import torch
import time
import numpy as np

torch.set_grad_enabled(False)

# Simulate sparse tensors like we'd get from FRA accumulation
print("Creating simulated sparse tensors...")
d_sae = 49152  # Actual SAE dimension
n_pairs = 500000  # Typical number of unique pairs

# Create random sparse tensors
print(f"  Creating sparse tensors with {n_pairs} non-zero entries...")
t0 = time.time()

# Random indices (ensure no duplicates)
indices = torch.stack([
    torch.randint(0, d_sae, (n_pairs,)),
    torch.randint(0, d_sae, (n_pairs,))
])

# Random values
sum_values = torch.rand(n_pairs) * 10
count_values = torch.randint(1, 100, (n_pairs,)).float()

# Create sparse tensors
interaction_sum = torch.sparse_coo_tensor(
    indices, sum_values,
    size=(d_sae, d_sae),
    dtype=torch.float32
)

interaction_count = torch.sparse_coo_tensor(
    indices, count_values,
    size=(d_sae, d_sae),
    dtype=torch.float32
)

print(f"  Created in {time.time()-t0:.2f}s")

# Test 1: Coalesce operation
print("\nTest 1: Coalesce...")
t0 = time.time()
sum_coalesced = interaction_sum.coalesce()
count_coalesced = interaction_count.coalesce()
print(f"  Coalesce: {time.time()-t0:.2f}s")

# Test 2: Extract indices and values
print("\nTest 2: Extract indices/values...")
t0 = time.time()
sum_indices = sum_coalesced.indices()
sum_vals = sum_coalesced.values()
count_indices = count_coalesced.indices()
count_vals = count_coalesced.values()
print(f"  Extraction: {time.time()-t0:.2f}s")
print(f"  Shapes: indices={sum_indices.shape}, values={sum_vals.shape}")

# Test 3: Compute averages (vectorized)
print("\nTest 3: Compute averages (vectorized)...")
t0 = time.time()
avg_values = sum_vals / count_vals
print(f"  Division: {time.time()-t0:.2f}s")

# Test 4: Find top-k
print("\nTest 4: Find top-k...")
for k in [10, 100, 1000]:
    t0 = time.time()
    top_vals, top_idx = torch.topk(avg_values, min(k, len(avg_values)))
    print(f"  Top-{k}: {time.time()-t0:.2f}s")

# Test 5: Python loop extraction (OLD METHOD)
print("\nTest 5: Python loop extraction (OLD METHOD)...")
t0 = time.time()
n_to_test = min(1000, len(sum_vals))
print(f"  Testing with {n_to_test} pairs...")

feature_pairs = []
for i in range(n_to_test):
    q_feat = sum_indices[0, i].item()
    k_feat = sum_indices[1, i].item()
    sum_val = sum_vals[i].item()
    count_val = count_vals[i].item()
    avg_val = sum_val / count_val
    
    feature_pairs.append({
        'query_feature': q_feat,
        'key_feature': k_feat,
        'sum': sum_val,
        'count': count_val,
        'average': avg_val
    })
print(f"  Loop extraction ({n_to_test} items): {time.time()-t0:.2f}s")

# Test 6: Sorting (OLD METHOD)
print("\nTest 6: Sorting Python list...")
t0 = time.time()
feature_pairs.sort(key=lambda x: x['average'], reverse=True)
print(f"  Sort {len(feature_pairs)} items: {time.time()-t0:.2f}s")

print("\n" + "="*60)
print("Summary:")
print(f"  Sparse tensor size: {d_sae} x {d_sae}")
print(f"  Non-zero entries: {n_pairs}")
print(f"  Density: {n_pairs / (d_sae * d_sae) * 100:.4f}%")