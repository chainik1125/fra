"""
Debug sparsity of SAE features at each position.
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

text = "The cat sat on the mat."
layer = 5
head = 0

print(f"\nAnalyzing: '{text}'")

# Get activations
activations = get_attention_activations(model, text, layer=layer, max_length=128)
print(f"Sequence length: {activations.shape[0]}")

# Encode to features
features = sae.encode(activations)
print(f"Feature shape: {features.shape}")

# Check sparsity at each position
print(f"\nSparsity per position:")
for i in range(min(10, features.shape[0])):  # First 10 positions
    active = torch.where(features[i] != 0)[0]
    sparsity = 1.0 - (len(active) / features.shape[1])
    print(f"  Position {i:2d}: {len(active):5d} active features ({sparsity:.1%} sparse)")

# Get model attention weights
W_Q = model.blocks[layer].attn.W_Q[head]
W_K = model.blocks[layer].attn.W_K[head]

# Process a few position pairs and track accumulation
print(f"\nProcessing position pairs and tracking accumulation:")
print(f"{'Pair':<10} {'Q_active':<10} {'K_active':<10} {'Interactions':<15} {'Total_accumulated':<15}")
print("-" * 70)

accumulated_nnz = 0
all_interactions = set()

for key_idx in range(min(5, features.shape[0])):
    for query_idx in range(key_idx, min(5, features.shape[0])):
        q_feat = features[query_idx]
        k_feat = features[key_idx]
        
        q_active = torch.where(q_feat != 0)[0]
        k_active = torch.where(k_feat != 0)[0]
        
        if len(q_active) > 0 and len(k_active) > 0:
            # Get decoder vectors
            q_vecs = sae.W_dec[q_active]
            k_vecs = sae.W_dec[k_active]
            
            # Compute attention
            q_proj = torch.matmul(q_vecs, W_Q)
            k_proj = torch.matmul(k_vecs, W_K)
            int_matrix = torch.matmul(q_proj, k_proj.T)
            
            # Scale
            int_matrix = int_matrix * q_feat[q_active].unsqueeze(1) * k_feat[k_active].unsqueeze(0)
            
            # Count non-zeros
            nnz_this_pair = (int_matrix.abs() > 1e-10).sum().item()
            
            # Add to accumulated set
            mask = int_matrix.abs() > 1e-10
            if mask.any():
                local_r, local_c = torch.where(mask)
                for r, c in zip(local_r.cpu().numpy(), local_c.cpu().numpy()):
                    global_r = q_active[r].item()
                    global_c = k_active[c].item()
                    all_interactions.add((global_r, global_c))
        else:
            nnz_this_pair = 0
        
        print(f"({query_idx},{key_idx})      {len(q_active):<10d} {len(k_active):<10d} {nnz_this_pair:<15d} {len(all_interactions):<15d}")

print(f"\nTotal unique feature pairs after {5*(5+1)//2} position pairs: {len(all_interactions)}")

import os
if torch.cuda.is_available():
    torch.cuda.empty_cache()
os._exit(0)