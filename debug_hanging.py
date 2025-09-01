"""
Debug where the code is hanging.
"""

import torch
import time

torch.set_grad_enabled(False)

print("1. Starting imports...", flush=True)
from fra.utils import load_model, load_sae
from fra.activation_utils import get_attention_activations

print("2. Loading model (this takes ~20s)...", flush=True)
t0 = time.time()
model = load_model('gpt2-small', 'cuda')
print(f"   Model loaded in {time.time()-t0:.1f}s", flush=True)

print("3. Loading SAE...", flush=True)
t0 = time.time()
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
print(f"   SAE loaded in {time.time()-t0:.1f}s", flush=True)

print("4. Getting SAE components...", flush=True)
W_enc = sae_data['state_dict']['W_enc'].cuda()
b_enc = sae_data['state_dict']['b_enc'].cuda()
b_dec = sae_data['state_dict']['b_dec'].cuda()
print(f"   W_enc shape: {W_enc.shape}", flush=True)

text = "Hi."
print(f"5. Text: '{text}'", flush=True)

print("6. Getting attention activations...", flush=True)
t0 = time.time()
activations = get_attention_activations(model, text, layer=5, max_length=10)
print(f"   Got activations in {time.time()-t0:.1f}s, shape: {activations.shape}", flush=True)

print("7. Computing features (wrong way)...", flush=True)
t0 = time.time()
import torch.nn.functional as F
features_wrong = F.relu(activations @ W_enc + b_enc)
print(f"   Computed in {time.time()-t0:.3f}s", flush=True)

print("8. Computing L0 for wrong way...", flush=True)
l0_wrong = (features_wrong > 0).sum(1).float().mean().item()
print(f"   L0 = {l0_wrong:.1f}", flush=True)

print("9. Computing features (correct way)...", flush=True)
t0 = time.time()
x_cent = activations - b_dec
features_correct = F.relu(x_cent @ W_enc + b_enc)
print(f"   Computed in {time.time()-t0:.3f}s", flush=True)

print("10. Computing L0 for correct way...", flush=True)
l0_correct = (features_correct > 0).sum(1).float().mean().item()
print(f"   L0 = {l0_correct:.1f}", flush=True)

print("11. Done!", flush=True)

import os
os._exit(0)