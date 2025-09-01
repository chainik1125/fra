"""
Check what's going on with SAE sparsity.
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

text = "The cat sat."

print(f"\nText: '{text}'")

# Get activations
activations = get_attention_activations(model, text, layer=5, max_length=128)
print(f"Activations shape: {activations.shape}")
print(f"Activations range: [{activations.min().item():.3f}, {activations.max().item():.3f}]")

# Check SAE encoding
features = sae.encode(activations)
print(f"\nSAE encoding:")
print(f"  Features shape: {features.shape}")
print(f"  Features dtype: {features.dtype}")
print(f"  Features device: {features.device}")

# Check actual zeros vs near-zeros
for pos in range(min(3, features.shape[0])):
    feat = features[pos]
    
    # Different thresholds
    exact_zeros = (feat == 0).sum().item()
    near_zeros_1e10 = (feat.abs() < 1e-10).sum().item()
    near_zeros_1e6 = (feat.abs() < 1e-6).sum().item()
    near_zeros_1e3 = (feat.abs() < 1e-3).sum().item()
    near_zeros_1e2 = (feat.abs() < 1e-2).sum().item()
    near_zeros_1e1 = (feat.abs() < 1e-1).sum().item()
    
    print(f"\nPosition {pos}:")
    print(f"  Exactly zero: {exact_zeros:,} / {feat.shape[0]:,} ({exact_zeros/feat.shape[0]:.1%})")
    print(f"  < 1e-10: {near_zeros_1e10:,} ({near_zeros_1e10/feat.shape[0]:.1%})")
    print(f"  < 1e-6:  {near_zeros_1e6:,} ({near_zeros_1e6/feat.shape[0]:.1%})")
    print(f"  < 1e-3:  {near_zeros_1e3:,} ({near_zeros_1e3/feat.shape[0]:.1%})")
    print(f"  < 1e-2:  {near_zeros_1e2:,} ({near_zeros_1e2/feat.shape[0]:.1%})")
    print(f"  < 1e-1:  {near_zeros_1e1:,} ({near_zeros_1e1/feat.shape[0]:.1%})")
    
    # Show some non-zero values
    nonzero_idx = torch.where(feat != 0)[0][:10]
    if len(nonzero_idx) > 0:
        print(f"  First 10 non-zero values: {feat[nonzero_idx].cpu().tolist()}")

# Check the SAE's ReLU
print(f"\nChecking SAE architecture:")
print(f"  W_enc shape: {sae.W_enc.shape}")
print(f"  b_enc shape: {sae.b_enc.shape}")
print(f"  W_dec shape: {sae.W_dec.shape}")

# Manually compute to check
x = activations[0:1]  # First position
pre_activation = x @ sae.W_enc + sae.b_enc
print(f"\nManual computation for position 0:")
print(f"  Pre-activation range: [{pre_activation.min().item():.3f}, {pre_activation.max().item():.3f}]")
print(f"  Negative pre-activations: {(pre_activation < 0).sum().item()} / {pre_activation.shape[1]}")

# Apply ReLU
import torch.nn.functional as F
manual_features = F.relu(pre_activation)
print(f"  After ReLU, zeros: {(manual_features == 0).sum().item()} / {manual_features.shape[1]}")
print(f"  Matches encode?: {torch.allclose(manual_features[0], features[0])}")

import os
os._exit(0)