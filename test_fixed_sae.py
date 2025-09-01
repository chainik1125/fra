"""
Test the fixed SAE wrapper to see if we get proper sparsity now.
"""

import torch
from fra.utils import load_model, load_sae
from fra.sae_wrapper_fixed import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations

torch.set_grad_enabled(False)

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')

# Use the FIXED wrapper
sae = SimpleAttentionSAE(sae_data)

text = "The cat sat on the mat. The cat was happy."
print(f"\nText: '{text}'")

# Get activations
activations = get_attention_activations(model, text, layer=5, max_length=128)
print(f"Activations shape: {activations.shape}")

# Encode with FIXED method
features = sae.encode(activations)
print(f"Features shape: {features.shape}")

# Check sparsity
print(f"\n🎉 SPARSITY CHECK WITH FIXED ENCODER:")
print(f"{'='*50}")

# L0 norm (average number of active features)
l0 = sae.compute_l0(features)
print(f"Average L0 (active features per position): {l0:.1f}")

# Check each position
print(f"\nPer-position statistics:")
for i in range(min(10, features.shape[0])):
    feat = features[i]
    n_active = (feat > 0).sum().item()
    sparsity = 1.0 - n_active / feat.shape[0]
    max_val = feat.max().item()
    print(f"  Position {i:2d}: {n_active:4d} active ({sparsity:6.2%} sparse), max={max_val:.3f}")

# Overall statistics
all_active = []
for i in range(features.shape[0]):
    n_active = (features[i] > 0).sum().item()
    all_active.append(n_active)

print(f"\n📊 OVERALL STATISTICS:")
print(f"  Average active features: {sum(all_active)/len(all_active):.1f}")
print(f"  Min active features: {min(all_active)}")
print(f"  Max active features: {max(all_active)}")
print(f"  Average sparsity: {1.0 - sum(all_active)/len(all_active)/features.shape[1]:.2%}")

# Check L1 norm share of top features
print(f"\n💪 L1 NORM SHARE OF TOP FEATURES:")
top_k_values = [10, 20, 50, 100]
for k in top_k_values:
    shares = []
    for i in range(features.shape[0]):
        feat = features[i]
        sorted_vals, _ = torch.sort(feat.abs(), descending=True)
        total_l1 = sorted_vals.sum().item()
        if total_l1 > 0:
            top_k_l1 = sorted_vals[:k].sum().item()
            shares.append(top_k_l1 / total_l1)
    if shares:
        avg_share = sum(shares) / len(shares)
        print(f"  Top-{k:3d}: {avg_share:6.2%} of L1 norm")

print(f"\n✅ Expected L0 ~16 according to paper")
print(f"   Actual L0: {l0:.1f}")
print(f"   {'SUCCESS!' if l0 < 100 else 'Still seems high...'}")

import os
os._exit(0)