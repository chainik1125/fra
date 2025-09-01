"""
Quick test of sparsity without reloading model.
"""

import torch
import pickle
import os

# Try to load cached model/SAE if available
cache_file = "/tmp/fra_cache.pkl"

if os.path.exists(cache_file):
    print("Loading from cache...")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    model = cache['model']
    sae = cache['sae']
else:
    print("Loading model and SAE...")
    from fra.utils import load_model, load_sae
    from fra.sae_wrapper import SimpleAttentionSAE
    
    model = load_model('gpt2-small', 'cuda')
    sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
    sae = SimpleAttentionSAE(sae_data)
    
    # Cache for next run
    with open(cache_file, 'wb') as f:
        pickle.dump({'model': model, 'sae': sae}, f)

torch.set_grad_enabled(False)

text = "Hi."
print(f"\nAnalyzing: '{text}'")

# Get activations
from fra.activation_utils import get_attention_activations
activations = get_attention_activations(model, text, layer=5, max_length=10)
print(f"Sequence length: {activations.shape[0]}")

# Encode to features
features = sae.encode(activations)
print(f"Feature shape: {features.shape}")
print(f"SAE d_sae: {sae.d_sae}")

# Check sparsity at each position
print(f"\nActive features per position:")
for i in range(features.shape[0]):
    active = torch.where(features[i] != 0)[0]
    sparsity = 1.0 - (len(active) / features.shape[1])
    print(f"  Pos {i}: {len(active):5d} active ({sparsity:.1%} sparse)")
    
    # Show first few active features
    if len(active) > 0:
        print(f"    First 5 features: {active[:5].cpu().tolist()}")
        print(f"    Activations: {features[i, active[:5]].cpu().tolist()}")

os._exit(0)