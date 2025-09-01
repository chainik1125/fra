#!/usr/bin/env python
"""Debug what patterns are actually being created."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import random

torch.set_grad_enabled(False)

LAYER = 5
DEVICE = 'cuda'
D_SAE = 49152

print("Loading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

prompt = "The capital of France is"

# Get FRA for head 0
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
    seq_len = fra_result['seq_len']
    
    # Find max non-self
    q_feats = indices[2, :]
    k_feats = indices[3, :]
    mask = q_feats != k_feats
    
    if mask.any():
        non_self_values = values[mask].abs()
        max_idx_in_filtered = non_self_values.argmax()
        non_self_indices = torch.where(mask)[0]
        max_idx = non_self_indices[max_idx_in_filtered].item()
        
        max_pair = {
            'query_pos': indices[0, max_idx].item(),
            'key_pos': indices[1, max_idx].item(),
            'feature_i': indices[2, max_idx].item(),
            'feature_j': indices[3, max_idx].item(),
        }
        
        print(f"\nMax pair: pos ({max_pair['query_pos']}, {max_pair['key_pos']}), "
              f"features F{max_pair['feature_i']}→F{max_pair['feature_j']}")
        
        # Create pattern by REMOVING the max pair
        targeted_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
        removed_count = 0
        kept_count = 0
        
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            feat_i = indices[2, i].item()
            feat_j = indices[3, i].item()
            
            # Skip if this matches the max pair
            if (q_pos == max_pair['query_pos'] and 
                k_pos == max_pair['key_pos'] and
                feat_i == max_pair['feature_i'] and
                feat_j == max_pair['feature_j']):
                removed_count += 1
                print(f"  REMOVED: pos ({q_pos}, {k_pos}), F{feat_i}→F{feat_j}, val={values[i]:.4f}")
            else:
                targeted_pattern[q_pos, k_pos] += values[i].item()
                kept_count += 1
        
        print(f"Targeted pattern: removed {removed_count}, kept {kept_count}")
        
        # Now try with random features that DON'T exist
        random_feat_i = random.randint(0, D_SAE - 1)
        random_feat_j = random.randint(0, D_SAE - 1)
        
        # Make sure they don't exist
        exists = False
        for i in range(indices.shape[1]):
            if (indices[2, i].item() == random_feat_i and 
                indices[3, i].item() == random_feat_j):
                exists = True
                break
        
        if not exists:
            print(f"\nRandom pair F{random_feat_i}→F{random_feat_j} does NOT exist in FRA")
            
            random_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
            removed_count = 0
            kept_count = 0
            
            for i in range(indices.shape[1]):
                q_pos = indices[0, i].item()
                k_pos = indices[1, i].item()
                feat_i = indices[2, i].item()
                feat_j = indices[3, i].item()
                
                # This will NEVER match since random features don't exist
                if (q_pos == max_pair['query_pos'] and  # Using max_pair positions
                    k_pos == max_pair['key_pos'] and
                    feat_i == random_feat_i and
                    feat_j == random_feat_j):
                    removed_count += 1
                    print(f"  REMOVED: This should never print!")
                else:
                    random_pattern[q_pos, k_pos] += values[i].item()
                    kept_count += 1
            
            print(f"Random pattern: removed {removed_count}, kept {kept_count}")
            print(f"Random pattern sum: {random_pattern.sum():.4f}")
            print(f"Targeted pattern sum: {targeted_pattern.sum():.4f}")
            print(f"Original FRA sum: {values.sum():.4f}")
            
            # The bug is that random pattern keeps EVERYTHING since the condition never matches!
            print("\nBUT WAIT - I'm also checking positions! Let me check that...")
            
            # Actually, the positions are from max_pair, so let's check what happens
            print(f"\nChecking position match at ({max_pair['query_pos']}, {max_pair['key_pos']}):")
            for i in range(indices.shape[1]):
                if (indices[0, i].item() == max_pair['query_pos'] and 
                    indices[1, i].item() == max_pair['key_pos']):
                    print(f"  Found at position: F{indices[2, i]}→F{indices[3, i]}")
            
            print("\nAH! The issue is that at the SAME POSITION, there are MULTIPLE feature pairs!")
            print("So when I ablate a 'random' pair at the max position, I'm removing ALL")
            print("feature pairs at that position IF the random features don't match any!")