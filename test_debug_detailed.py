#!/usr/bin/env python
"""Debug with detailed timing."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import time

torch.set_grad_enabled(False)

print("Detailed timing debug...")

# Load model and SAE
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

text = "The cat sat on the mat."

# Step by step timing
print("\n1. Get FRA result:")
t0 = time.time()
fra_result = get_sentence_fra_batch(
    model, sae, text, 5, 0,
    max_length=128, top_k=20, verbose=False
)
print(f"   FRA computation: {time.time()-t0:.2f}s")

print("\n2. Extract sparse tensor:")
t0 = time.time()
fra_sparse = fra_result['fra_tensor_sparse']
print(f"   Extract tensor: {time.time()-t0:.2f}s")

print("\n3. Get indices:")
t0 = time.time()
indices = fra_sparse.indices()
print(f"   Get indices: {time.time()-t0:.2f}s")
print(f"   Indices shape: {indices.shape}")

print("\n4. Get values:")
t0 = time.time()
values = fra_sparse.values()
print(f"   Get values: {time.time()-t0:.2f}s")
print(f"   Values shape: {values.shape}")

print("\n5. Loop through first 100:")
interaction_sum = {}
interaction_count = {}

t0 = time.time()
for i in range(min(100, indices.shape[1])):
    q_pos, k_pos, q_feat, k_feat = indices[:, i].tolist()
    strength = values[i].item()
    
    if q_feat == k_feat:
        continue
    
    pair_key = (q_feat, k_feat)
    interaction_sum[pair_key] = interaction_sum.get(pair_key, 0.0) + abs(strength)
    interaction_count[pair_key] = interaction_count.get(pair_key, 0) + 1
print(f"   Loop (100 items): {time.time()-t0:.2f}s")

print("\n6. Delete and clear:")
t0 = time.time()
del fra_sparse
del fra_result
torch.cuda.empty_cache()
print(f"   Clear memory: {time.time()-t0:.2f}s")

print("\n✅ All steps should be fast (<1s each)")