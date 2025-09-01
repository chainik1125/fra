"""
Check what activation function the SAE uses.
"""

import torch
from fra.utils import load_sae
from fra.sae_wrapper import SimpleAttentionSAE

print("Loading SAE...")
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')

# Check what's in the SAE data
print("\nSAE data keys:", sae_data.keys())

# Check config
if 'config' in sae_data:
    print("\nSAE config:")
    for key, value in sae_data['config'].items():
        print(f"  {key}: {value}")

# Check state dict keys
print("\nState dict keys:", list(sae_data['state_dict'].keys()))

# Load into wrapper
sae = SimpleAttentionSAE(sae_data)

print("\nSAE wrapper attributes:")
print(f"  d_in: {sae.d_in}")
print(f"  d_sae: {sae.d_sae}")
print(f"  W_enc shape: {sae.W_enc.shape}")
print(f"  b_enc shape: {sae.b_enc.shape}")
print(f"  W_dec shape: {sae.W_dec.shape}")

# Let's check our encode function
print("\nOur encode function in SimpleAttentionSAE:")
import inspect
print(inspect.getsource(sae.encode))

# Test encoding with a simple input
print("\nTesting encoding behavior:")
test_input = torch.randn(1, sae.d_in).cuda()

# Manual computation
pre_act = test_input @ sae.W_enc + sae.b_enc
print(f"Pre-activation shape: {pre_act.shape}")
print(f"Pre-activation range: [{pre_act.min().item():.3f}, {pre_act.max().item():.3f}]")
print(f"Negative pre-activations: {(pre_act < 0).sum().item()} / {pre_act.shape[1]}")

# Apply ReLU
import torch.nn.functional as F
post_relu = F.relu(pre_act)
print(f"After ReLU, zeros: {(post_relu == 0).sum().item()} / {post_relu.shape[1]}")
print(f"Sparsity: {(post_relu == 0).sum().item() / post_relu.numel():.1%}")

# Compare with encode function
encoded = sae.encode(test_input)
print(f"Encoded zeros: {(encoded == 0).sum().item()} / {encoded.shape[1]}")
print(f"Matches manual ReLU? {torch.allclose(encoded, post_relu)}")

# Check if there's a different activation mentioned
print("\nSearching for activation info in config...")
if 'config' in sae_data:
    config_str = str(sae_data['config'])
    if 'relu' in config_str.lower():
        print("  Found 'relu' in config")
    if 'gelu' in config_str.lower():
        print("  Found 'gelu' in config")
    if 'activation' in config_str.lower():
        print("  Found 'activation' in config")

import os
os._exit(0)