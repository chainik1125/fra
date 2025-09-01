"""
Test with actual sentence to see sparsity.
"""

import torch
import time

torch.set_grad_enabled(False)

from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

text = "The cat sat on the mat."
print(f"\nAnalyzing: '{text}'")

activations = get_attention_activations(model, text, layer=5, max_length=128)
print(f"Activations shape: {activations.shape}")

features = sae.encode(activations)
print(f"Features shape: {features.shape}")

print(f"\nActive features per position:")
total_active = 0
for i in range(features.shape[0]):
    active = torch.where(features[i] != 0)[0]
    total_active += len(active)
    sparsity = 1.0 - (len(active) / features.shape[1])
    print(f"  Position {i}: {len(active):5d} active ({sparsity:.1%} sparse)")

avg_active = total_active / features.shape[0]
print(f"\nAverage active features per position: {avg_active:.1f}")
print(f"Average sparsity: {1.0 - avg_active/features.shape[1]:.1%}")

# Test timing for one pair with high activity
W_Q = model.blocks[5].attn.W_Q[0]
W_K = model.blocks[5].attn.W_K[0]

# Find positions with most active features
max_pos = 0
max_active = 0
for i in range(features.shape[0]):
    active = torch.where(features[i] != 0)[0]
    if len(active) > max_active:
        max_active = len(active)
        max_pos = i

print(f"\nPosition with most active features: {max_pos} with {max_active} features")

# Time computation for worst-case pair
if features.shape[0] >= 2:
    q_active = torch.where(features[max_pos] != 0)[0]
    k_active = q_active  # Same position for worst case
    
    print(f"\nTiming worst-case interaction ({max_active} x {max_active}):")
    t0 = time.time()
    
    q_vecs = sae.W_dec[q_active]
    k_vecs = sae.W_dec[k_active]
    
    q_proj = torch.matmul(q_vecs, W_Q)
    k_proj = torch.matmul(k_vecs, W_K)
    int_matrix = torch.matmul(q_proj, k_proj.T)
    
    # Scale
    int_matrix = int_matrix * features[max_pos, q_active].unsqueeze(1) * features[max_pos, k_active].unsqueeze(0)
    
    # Count non-zeros
    nnz = (int_matrix.abs() > 1e-10).sum().item()
    
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Matrix shape: {int_matrix.shape}")
    print(f"  Non-zeros: {nnz} out of {int_matrix.numel()} ({nnz/int_matrix.numel():.1%})")

import os
os._exit(0)