"""
Minimal debug to find where it hangs.
"""

import torch
import time

torch.set_grad_enabled(False)

print("1. Loading imports...", flush=True)
from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations

print("2. Loading model...", flush=True)
t0 = time.time()
model = load_model('gpt2-small', 'cuda')
print(f"   Model loaded in {time.time()-t0:.1f}s", flush=True)

print("3. Loading SAE...", flush=True)
t0 = time.time()
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)
print(f"   SAE loaded in {time.time()-t0:.1f}s", flush=True)

text = "Hi."
print(f"4. Getting activations for '{text}'...", flush=True)
t0 = time.time()
activations = get_attention_activations(model, text, layer=5, max_length=10)
print(f"   Activations computed in {time.time()-t0:.1f}s, shape: {activations.shape}", flush=True)

print("5. Encoding to SAE features...", flush=True)
t0 = time.time()
features = sae.encode(activations)
print(f"   Encoded in {time.time()-t0:.1f}s, shape: {features.shape}", flush=True)

print("6. Checking sparsity...", flush=True)
for i in range(features.shape[0]):
    active = torch.where(features[i] != 0)[0]
    print(f"   Position {i}: {len(active)} active features", flush=True)

print("7. Getting one interaction matrix...", flush=True)
if features.shape[0] >= 2:
    W_Q = model.blocks[5].attn.W_Q[0]
    W_K = model.blocks[5].attn.W_K[0]
    
    q_active = torch.where(features[1] != 0)[0]
    k_active = torch.where(features[0] != 0)[0]
    
    print(f"   Query has {len(q_active)} active, Key has {len(k_active)} active", flush=True)
    
    if len(q_active) > 0 and len(k_active) > 0:
        t0 = time.time()
        q_vecs = sae.W_dec[q_active]
        k_vecs = sae.W_dec[k_active]
        
        q_proj = torch.matmul(q_vecs, W_Q)
        k_proj = torch.matmul(k_vecs, W_K)
        int_matrix = torch.matmul(q_proj, k_proj.T)
        
        print(f"   Interaction matrix computed in {time.time()-t0:.3f}s", flush=True)
        print(f"   Shape: {int_matrix.shape}, max: {int_matrix.abs().max().item():.4f}", flush=True)

print("8. Done!", flush=True)

import os
os._exit(0)