"""
Test loading SAEs directly through SAE Lens with correct hook names.
"""

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

print("Loading GPT-2 small model...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

print("\nLoading SAE from SAE Lens...")

# Load SAE for layer 5 attention
# Based on the metadata, the actual hook name is blocks.5.attn.hook_z
layer = 5
sae = SAE.from_pretrained(
    release="gpt2-small-hook-z-kk",
    sae_id=f"blocks.{layer}.attn.hook_z",  # Correct hook name
    device="cuda"
)

print(f"SAE loaded successfully!")
print(f"  Dictionary size: {sae.cfg.d_sae}")
print(f"  Input dimension: {sae.cfg.d_in}")
print(f"  Apply b_dec to input: {sae.cfg.apply_b_dec_to_input}")
print(f"  Reshape activations: {sae.cfg.reshape_activations}")

# Test on sample text
text = "The cat sat on the mat. The cat was happy."
print(f"\nTest text: '{text}'")

# Get tokens
tokens = model.to_tokens(text)
print(f"Tokens shape: {tokens.shape}")

# Run model and get activations at the correct hook point
hook_name = f"blocks.{layer}.attn.hook_z"
_, cache = model.run_with_cache(tokens, names_filter=[hook_name])
activations = cache[hook_name]
print(f"Raw activations shape: {activations.shape}")

# The activations should be [batch, seq, n_heads, d_head]
# SAE Lens should handle the reshaping based on cfg.reshape_activations
if len(activations.shape) == 4:
    batch, seq, n_heads, d_head = activations.shape
    print(f"  Detected: batch={batch}, seq={seq}, n_heads={n_heads}, d_head={d_head}")

# Remove batch dimension
activations = activations[0]  # [seq, n_heads, d_head]

# Encode with SAE - it should handle the reshaping internally
print("\nEncoding with SAE...")
feature_acts = sae.encode(activations)
print(f"Feature activations shape: {feature_acts.shape}")

# Calculate L0
l0_per_position = (feature_acts > 0).sum(dim=-1).float()
avg_l0 = l0_per_position.mean().item()

print(f"\n📊 Results:")
print(f"  Average L0: {avg_l0:.1f} active features")
print(f"  Min L0: {l0_per_position.min().item():.0f}")
print(f"  Max L0: {l0_per_position.max().item():.0f}")
print(f"  Sparsity: {1.0 - avg_l0/sae.cfg.d_sae:.2%}")

# Per-position breakdown
print(f"\nPer-position L0:")
for i in range(min(10, feature_acts.shape[0])):
    l0 = (feature_acts[i] > 0).sum().item()
    sparsity = 1.0 - l0/sae.cfg.d_sae
    print(f"  Position {i:2d}: {l0:4.0f} active features ({sparsity:.2%} sparse)")

print(f"\n{'='*50}")
print(f"Expected L0: ~16-50 (from paper)")
print(f"Actual L0: {avg_l0:.1f}")
print(f"{'SUCCESS!' if avg_l0 < 100 else 'Still high, but better!'}")

# Also test reconstruction
print(f"\nTesting reconstruction...")
reconstructed = sae.decode(feature_acts)
print(f"Reconstructed shape: {reconstructed.shape}")

# Check reconstruction error
if reconstructed.shape == activations.shape:
    mse = ((reconstructed - activations) ** 2).mean().item()
    print(f"Reconstruction MSE: {mse:.6f}")

import os
os._exit(0)