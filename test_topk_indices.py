"""
Test that top-k implementation correctly preserves original feature indices.
"""

import torch
from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations

torch.set_grad_enabled(False)

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

text = "Hi."
layer = 5
head = 0

print(f"\nText: '{text}'")

# Get activations and encode
activations = get_attention_activations(model, text, layer=layer, max_length=10)
features = sae.encode(activations)
seq_len = features.shape[0]

print(f"Sequence length: {seq_len}")

# Test top-k selection for one position
pos = 0
feat = features[pos]
print(f"\nTesting position {pos}:")

# Original non-zero features
orig_nonzero = torch.where(feat != 0)[0]
orig_values = feat[orig_nonzero]
print(f"  Original non-zero features: {len(orig_nonzero)}")
print(f"  First 10 indices: {orig_nonzero[:10].cpu().tolist()}")
print(f"  First 10 values: {orig_values[:10].cpu().tolist()}")

# Apply top-k selection
k = 20
topk_vals, topk_idx = torch.topk(feat.abs(), min(k, (feat != 0).sum().item()))
print(f"\n  Top-{k} selection:")
print(f"  Top-k indices (in original space): {topk_idx[:10].cpu().tolist()}")
print(f"  Top-k values: {feat[topk_idx[:10]].cpu().tolist()}")

# Create sparse version with only top-k
sparse_feat = torch.zeros_like(feat)
sparse_feat[topk_idx] = feat[topk_idx]

# Verify indices are preserved
sparse_nonzero = torch.where(sparse_feat != 0)[0]
print(f"\n  After sparsification:")
print(f"  Non-zero indices: {sparse_nonzero[:10].cpu().tolist()}")
print(f"  Values at those indices: {sparse_feat[sparse_nonzero[:10]].cpu().tolist()}")

# Check they match
print(f"\n  Indices match top-k? {torch.equal(sparse_nonzero, topk_idx)}")

# Now test in the FRA computation
print(f"\n\nTesting in FRA computation:")

W_Q = model.blocks[layer].attn.W_Q[head]
W_K = model.blocks[layer].attn.W_K[head]

# Use two positions
if seq_len >= 2:
    # Original computation with all features
    q_feat_orig = features[1]
    k_feat_orig = features[0]
    
    q_active_orig = torch.where(q_feat_orig != 0)[0]
    k_active_orig = torch.where(k_feat_orig != 0)[0]
    
    print(f"Original features:")
    print(f"  Query position 1: {len(q_active_orig)} active features")
    print(f"  Key position 0: {len(k_active_orig)} active features")
    
    # Top-k version
    topk_q_vals, topk_q_idx = torch.topk(q_feat_orig.abs(), min(k, (q_feat_orig != 0).sum().item()))
    topk_k_vals, topk_k_idx = torch.topk(k_feat_orig.abs(), min(k, (k_feat_orig != 0).sum().item()))
    
    q_feat_topk = torch.zeros_like(q_feat_orig)
    q_feat_topk[topk_q_idx] = q_feat_orig[topk_q_idx]
    
    k_feat_topk = torch.zeros_like(k_feat_orig)
    k_feat_topk[topk_k_idx] = k_feat_orig[topk_k_idx]
    
    q_active_topk = torch.where(q_feat_topk != 0)[0]
    k_active_topk = torch.where(k_feat_topk != 0)[0]
    
    print(f"\nTop-{k} features:")
    print(f"  Query: {len(q_active_topk)} features")
    print(f"  Key: {len(k_active_topk)} features")
    print(f"  Query indices: {q_active_topk.cpu().tolist()}")
    print(f"  Key indices: {k_active_topk.cpu().tolist()}")
    
    # Compute interactions with top-k
    q_vecs = sae.W_dec[q_active_topk]
    k_vecs = sae.W_dec[k_active_topk]
    
    q_proj = torch.matmul(q_vecs, W_Q)
    k_proj = torch.matmul(k_vecs, W_K)
    int_matrix = torch.matmul(q_proj, k_proj.T)
    
    # Scale
    int_matrix = int_matrix * q_feat_topk[q_active_topk].unsqueeze(1) * k_feat_topk[k_active_topk].unsqueeze(0)
    
    # Find non-zeros and map back to original indices
    mask = int_matrix.abs() > 1e-10
    if mask.any():
        local_r, local_c = torch.where(mask)
        
        # These should map back to original feature space
        global_rows = q_active_topk[local_r]
        global_cols = k_active_topk[local_c]
        
        print(f"\nInteraction matrix:")
        print(f"  Shape: {int_matrix.shape}")
        print(f"  Non-zeros: {mask.sum().item()}")
        print(f"  First 5 global row indices: {global_rows[:5].cpu().tolist()}")
        print(f"  First 5 global col indices: {global_cols[:5].cpu().tolist()}")
        
        # Verify these are valid indices in original feature space
        print(f"\n  All row indices < {sae.d_sae}? {(global_rows < sae.d_sae).all().item()}")
        print(f"  All col indices < {sae.d_sae}? {(global_cols < sae.d_sae).all().item()}")
        print(f"  Row indices are subset of query's top-k? {set(global_rows.cpu().tolist()).issubset(set(q_active_topk.cpu().tolist()))}")
        print(f"  Col indices are subset of key's top-k? {set(global_cols.cpu().tolist()).issubset(set(k_active_topk.cpu().tolist()))}")

import os
os._exit(0)