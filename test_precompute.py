"""
Test just the pre-computation step.
"""

import torch
import time
from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE

torch.set_grad_enabled(False)

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

print(f"\nSAE dimensions:")
print(f"  d_in (model dim): {sae.d_in}")
print(f"  d_sae (features): {sae.d_sae}")
print(f"  W_dec shape: {sae.W_dec.shape}")

# Get attention weights
W_Q = model.blocks[5].attn.W_Q[0]  # Head 0
W_K = model.blocks[5].attn.W_K[0]

print(f"\nAttention weight shapes:")
print(f"  W_Q: {W_Q.shape}")
print(f"  W_K: {W_K.shape}")

# Test the pre-computation
print(f"\nPre-computing Q and K for all {sae.d_sae} features...")
t0 = time.time()

# This is the potentially expensive operation
all_Q = torch.matmul(sae.W_dec, W_Q)  # [49152, 64]
print(f"  Q computed in {time.time() - t0:.2f}s, shape: {all_Q.shape}")

t0 = time.time()
all_K = torch.matmul(sae.W_dec, W_K)  # [49152, 64]
print(f"  K computed in {time.time() - t0:.2f}s, shape: {all_K.shape}")

print(f"\nMemory usage:")
print(f"  all_Q size: {all_Q.element_size() * all_Q.numel() / 1024**2:.2f} MB")
print(f"  all_K size: {all_K.element_size() * all_K.numel() / 1024**2:.2f} MB")

import os
if torch.cuda.is_available():
    torch.cuda.empty_cache()
os._exit(0)