"""
Simple test of L1 norm share for top features.
"""

import torch
from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations

torch.set_grad_enabled(False)

print("Loading...", flush=True)
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

text = "The cat sat on the mat."
print(f"\nText: '{text}'", flush=True)

# Get features
activations = get_attention_activations(model, text, layer=5, max_length=128)
features = sae.encode(activations)
print(f"Shape: {features.shape}", flush=True)

# For each position
print(f"\nL1 norm share for top-k features:", flush=True)
print(f"{'Pos':<5} {'Top-10':<10} {'Top-20':<10} {'Top-50':<10} {'Top-100':<10}", flush=True)

for pos in range(features.shape[0]):
    feat = features[pos]
    
    # Sort by absolute value
    sorted_vals, _ = torch.sort(feat.abs(), descending=True)
    total_l1 = sorted_vals.sum().item()
    
    if total_l1 > 0:
        top_10_share = sorted_vals[:10].sum().item() / total_l1
        top_20_share = sorted_vals[:20].sum().item() / total_l1
        top_50_share = sorted_vals[:50].sum().item() / total_l1
        top_100_share = sorted_vals[:100].sum().item() / total_l1
        
        print(f"{pos:<5} {top_10_share:<10.2%} {top_20_share:<10.2%} {top_50_share:<10.2%} {top_100_share:<10.2%}", flush=True)

# Overall average
print(f"\nCalculating average across all positions...", flush=True)
all_top20_shares = []
for pos in range(features.shape[0]):
    feat = features[pos]
    sorted_vals, _ = torch.sort(feat.abs(), descending=True)
    total_l1 = sorted_vals.sum().item()
    if total_l1 > 0:
        all_top20_shares.append(sorted_vals[:20].sum().item() / total_l1)

print(f"\nAverage L1 share for top-20: {sum(all_top20_shares)/len(all_top20_shares):.2%}", flush=True)

# Also check actual values
print(f"\nExample: Position 0 top features:", flush=True)
feat = features[0]
sorted_vals, sorted_idx = torch.sort(feat.abs(), descending=True)
for i in range(20):
    idx = sorted_idx[i].item()
    val = feat[idx].item()
    print(f"  Rank {i+1}: Feature {idx:5d} = {val:8.4f}", flush=True)

import os
os._exit(0)