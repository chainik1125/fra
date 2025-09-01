#!/usr/bin/env python
"""Debug why random ablation has same effect as targeted."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import random

torch.set_grad_enabled(False)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12
D_SAE = 49152

print("Loading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

prompt = "The capital of France is"

# Get FRA for one head
print(f"\nAnalyzing prompt: {prompt}")
fra_result = get_sentence_fra_batch(
    model, sae, prompt, 
    layer=LAYER, head=0,
    max_length=128, top_k=20,
    verbose=False
)

if fra_result:
    fra_sparse = fra_result['fra_tensor_sparse']
    indices = fra_sparse.indices()
    values = fra_sparse.values()
    
    print(f"FRA sparse shape: {indices.shape}")
    print(f"Number of non-zero values: {len(values)}")
    
    # Find max non-self
    q_feats = indices[2, :]
    k_feats = indices[3, :]
    mask = q_feats != k_feats
    
    if mask.any():
        non_self_values = values[mask].abs()
        max_idx_in_filtered = non_self_values.argmax()
        non_self_indices = torch.where(mask)[0]
        max_idx = non_self_indices[max_idx_in_filtered].item()
        
        print(f"\nMax non-self interaction:")
        print(f"  Position: ({indices[0, max_idx]}, {indices[1, max_idx]})")
        print(f"  Features: F{indices[2, max_idx]} → F{indices[3, max_idx]}")
        print(f"  Value: {values[max_idx]:.4f}")
        
        # Generate random features
        random_feat_i = random.randint(0, D_SAE - 1)
        random_feat_j = random.randint(0, D_SAE - 1)
        while random_feat_j == random_feat_i:
            random_feat_j = random.randint(0, D_SAE - 1)
        
        print(f"\nRandom features: F{random_feat_i} → F{random_feat_j}")
        
        # Check if random features exist in the FRA
        random_exists = False
        for i in range(indices.shape[1]):
            if (indices[2, i].item() == random_feat_i and 
                indices[3, i].item() == random_feat_j):
                random_exists = True
                print(f"  Random pair EXISTS in FRA with value: {values[i]:.4f}")
                break
        
        if not random_exists:
            print(f"  Random pair does NOT exist in FRA (would be zero)")
        
        # Check how many unique feature pairs exist
        unique_pairs = set()
        for i in range(indices.shape[1]):
            pair = (indices[2, i].item(), indices[3, i].item())
            unique_pairs.add(pair)
        
        print(f"\nTotal unique feature pairs in FRA: {len(unique_pairs)}")
        print(f"Probability random pair exists: {len(unique_pairs)/(D_SAE*D_SAE):.6f}")

print("\nKey insight: Random features likely don't exist in the sparse FRA,")
print("so ablating them has no effect (removing zero from zero).")
print("We need to sample FROM the existing non-zero features!")