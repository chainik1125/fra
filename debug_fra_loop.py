"""
Debug the FRA computation loop to understand sparsity and accumulation.
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

text = "The cat sat."  # Short text for debugging
layer = 5
head = 0

print(f"\nAnalyzing: '{text}'")

# Get activations and encode
activations = get_attention_activations(model, text, layer=layer, max_length=128)
features = sae.encode(activations)
seq_len = features.shape[0]

print(f"Sequence length: {seq_len}")
print(f"Feature tensor shape: {features.shape}")

# Check sparsity at each position
print(f"\nActive features per position:")
for i in range(seq_len):
    active = torch.where(features[i] != 0)[0]
    sparsity = 1.0 - (len(active) / features.shape[1])
    print(f"  Position {i}: {len(active):5d} active ({sparsity:.1%} sparse)")

# Get attention weights
W_Q = model.blocks[layer].attn.W_Q[head]
W_K = model.blocks[layer].attn.W_K[head]

# Process position pairs with detailed tracking
print(f"\nProcessing lower triangle position pairs:")
print(f"{'Pair':<12} {'Q_active':<10} {'K_active':<10} {'Q*K_possible':<15} {'Non-zeros':<12} {'Total_NNZ':<12}")
print("-" * 85)

total_accumulated_nnz = 0
all_feature_pairs = set()

pair_count = 0
for key_idx in range(seq_len):
    for query_idx in range(key_idx, seq_len):  # Lower triangle
        q_feat = features[query_idx]
        k_feat = features[key_idx]
        
        q_active = torch.where(q_feat != 0)[0]
        k_active = torch.where(k_feat != 0)[0]
        
        possible_interactions = len(q_active) * len(k_active)
        
        if len(q_active) > 0 and len(k_active) > 0:
            # Compute interaction matrix for this pair
            q_vecs = sae.W_dec[q_active]
            k_vecs = sae.W_dec[k_active]
            
            q_proj = torch.matmul(q_vecs, W_Q)
            k_proj = torch.matmul(k_vecs, W_K)
            int_matrix = torch.matmul(q_proj, k_proj.T)
            
            # Scale by activations
            int_matrix = int_matrix * q_feat[q_active].unsqueeze(1) * k_feat[k_active].unsqueeze(0)
            
            # Count non-zeros for this pair
            mask = int_matrix.abs() > 1e-10
            nnz_this_pair = mask.sum().item()
            
            # Add to global accumulation
            if mask.any():
                local_r, local_c = torch.where(mask)
                for r, c in zip(local_r.cpu().numpy(), local_c.cpu().numpy()):
                    global_r = q_active[r].item()
                    global_c = k_active[c].item()
                    all_feature_pairs.add((global_r, global_c))
        else:
            nnz_this_pair = 0
        
        total_accumulated_nnz = len(all_feature_pairs)
        
        print(f"({query_idx:2d},{key_idx:2d})      {len(q_active):10d} {len(k_active):10d} {possible_interactions:15d} {nnz_this_pair:12d} {total_accumulated_nnz:12d}")
        
        pair_count += 1

print(f"\n" + "="*85)
print(f"Summary:")
print(f"  Total position pairs processed: {pair_count}")
print(f"  Total unique feature pairs with non-zero interaction: {len(all_feature_pairs)}")
print(f"  Density in {sae.d_sae}x{sae.d_sae} matrix: {len(all_feature_pairs)/(sae.d_sae**2):.8%}")

import os
if torch.cuda.is_available():
    torch.cuda.empty_cache()
os._exit(0)