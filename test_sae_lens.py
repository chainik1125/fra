"""
Test loading SAEs directly through SAE Lens.
"""

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

print("Loading GPT-2 small model...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

# From the SAE table, we can load attention SAEs
# The format is usually: model_name, hook_point, sae_id
print("\nLoading SAE from SAE Lens...")

# Try loading the attention SAE for layer 5
# Based on the table, Connor Kissane's SAEs are available
sae_id = "blocks.5.hook_z"  # Hook point for attention output before projection

try:
    # Load SAE using SAE Lens
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-hook-z-kk",  # Connor Kissane's release
        sae_id=sae_id,
        device="cuda"
    )
    
    print(f"SAE loaded successfully!")
    print(f"  Config: {cfg_dict}")
    print(f"  Sparsity info: {sparsity}")
    print(f"  Dictionary size: {sae.cfg.d_sae}")
    print(f"  Input dimension: {sae.cfg.d_in}")
    
    # Test on sample text
    text = "The cat sat on the mat."
    print(f"\nTest text: '{text}'")
    
    # Get tokens
    tokens = model.to_tokens(text)
    print(f"Tokens shape: {tokens.shape}")
    
    # Run model and get activations at the hook point
    _, cache = model.run_with_cache(tokens, names_filter=[sae_id])
    activations = cache[sae_id]
    print(f"Activations shape: {activations.shape}")
    
    # If it's 4D [batch, seq, n_heads, d_head], we need to reshape
    if len(activations.shape) == 4:
        batch, seq, n_heads, d_head = activations.shape
        # Concatenate heads
        activations = activations.reshape(batch, seq, n_heads * d_head)
        print(f"Reshaped to: {activations.shape}")
    
    # Remove batch dimension for SAE
    activations = activations[0]  # [seq, d_model]
    
    # Encode with SAE
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
        print(f"  Position {i}: {l0} active features")
    
    print(f"\n✅ Expected L0 ~16-50 for attention SAEs")
    
except Exception as e:
    print(f"Error loading SAE: {e}")
    print("\nTrying alternative approach...")
    
    # Alternative: Try loading with different parameters
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    
    # Get available SAEs
    directory = get_pretrained_saes_directory()
    print("\nAvailable SAE releases:")
    for release_name in directory.keys():
        if "gpt2" in release_name.lower():
            print(f"  - {release_name}")

import os
os._exit(0)