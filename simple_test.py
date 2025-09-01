"""
Very simple test to diagnose hanging.
"""

import torch
print("1. Import torch OK")

from fra.utils import load_model, load_sae
print("2. Import utils OK")

from fra.sae_wrapper import SimpleAttentionSAE
print("3. Import SAE wrapper OK")

torch.set_grad_enabled(False)
print("4. Disabled gradients")

print("5. Loading model...")
model = load_model('gpt2-small', 'cuda')
print("6. Model loaded")

print("7. Loading SAE...")
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)
print("8. SAE loaded")

# Simple forward pass
print("9. Testing forward pass...")
text = "Hi"
tokens = model.tokenizer.encode(text)
print(f"10. Tokens: {tokens}")

tokens_t = torch.tensor([tokens]).cuda()
print(f"11. Tokens tensor shape: {tokens_t.shape}")

# Use cache to avoid hanging
print("12. Running forward pass...")
with torch.no_grad():
    _, cache = model.run_with_cache(tokens_t, names_filter=["blocks.5.hook_attn_out"])
print("13. Forward pass complete")

activations = cache["blocks.5.hook_attn_out"][0]
print(f"14. Activations shape: {activations.shape}")

# Test encoding
print("15. Testing SAE encoding...")
features = sae.encode(activations)
print(f"16. Features shape: {features.shape}")

# Count non-zeros
nz = (features != 0).sum(1).float().mean().item()
print(f"17. Average non-zeros: {nz}")

print("18. Done!")
import os
os._exit(0)