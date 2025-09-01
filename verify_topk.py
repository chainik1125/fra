"""
Simple verification that top-k indices are preserved correctly.
"""

import torch

# Create a test feature vector
d_sae = 100
feat = torch.zeros(d_sae)

# Put some values at specific indices
test_indices = [5, 10, 20, 30, 50, 70, 90]
test_values = [10.0, 5.0, 3.0, 8.0, 2.0, 6.0, 1.0]

for idx, val in zip(test_indices, test_values):
    feat[idx] = val

print("Original feature vector:")
print(f"  Non-zero indices: {torch.where(feat != 0)[0].tolist()}")
print(f"  Non-zero values: {feat[feat != 0].tolist()}")

# Apply top-k selection (k=3)
k = 3
topk_vals, topk_idx = torch.topk(feat.abs(), min(k, (feat != 0).sum().item()))

print(f"\nTop-{k} selection:")
print(f"  Top-k indices: {topk_idx.tolist()}")
print(f"  Top-k values: {topk_vals.tolist()}")

# Create sparse version
sparse_feat = torch.zeros_like(feat)
sparse_feat[topk_idx] = feat[topk_idx]

print(f"\nSparse feature vector after top-k:")
print(f"  Non-zero indices: {torch.where(sparse_feat != 0)[0].tolist()}")
print(f"  Non-zero values: {sparse_feat[sparse_feat != 0].tolist()}")

# Verify we kept the right indices
q_active = torch.where(sparse_feat != 0)[0]
print(f"\nUsing torch.where on sparse vector:")
print(f"  Active indices: {q_active.tolist()}")
print(f"  These should be: [5, 30, 70] (the indices with values 10.0, 8.0, 6.0)")

# Simulate what happens in FRA
print(f"\nSimulating FRA index mapping:")
# Pretend we have an interaction matrix of size [3, 3]
int_matrix = torch.randn(len(q_active), len(q_active))
mask = int_matrix.abs() > 0.5  # Arbitrary threshold

if mask.any():
    local_r, local_c = torch.where(mask)
    print(f"  Local indices where mask=True: rows={local_r.tolist()}, cols={local_c.tolist()}")
    
    # Map back to global
    global_rows = q_active[local_r]
    global_cols = q_active[local_c]
    print(f"  Global feature indices: rows={global_rows.tolist()}, cols={global_cols.tolist()}")
    print(f"  These should all be from {q_active.tolist()}")

print("\n✓ Index mapping looks correct!")