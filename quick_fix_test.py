"""
Quick test of the fix.
"""

import torch
from fra.utils import load_model, load_sae
from fra.activation_utils import get_attention_activations

torch.set_grad_enabled(False)

print("Loading...", flush=True)
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')

# Get components
W_enc = sae_data['state_dict']['W_enc'].cuda()
b_enc = sae_data['state_dict']['b_enc'].cuda()
b_dec = sae_data['state_dict']['b_dec'].cuda()

text = "Hi."
activations = get_attention_activations(model, text, layer=5, max_length=10)
print(f"Activations: {activations.shape}", flush=True)

# Original (wrong) way
import torch.nn.functional as F
features_wrong = F.relu(activations @ W_enc + b_enc)
l0_wrong = (features_wrong > 0).sum(1).float().mean().item()
print(f"\nWRONG way (no b_dec subtraction): L0 = {l0_wrong:.1f}", flush=True)

# Fixed (correct) way
x_cent = activations - b_dec
features_correct = F.relu(x_cent @ W_enc + b_enc)
l0_correct = (features_correct > 0).sum(1).float().mean().item()
print(f"CORRECT way (with b_dec subtraction): L0 = {l0_correct:.1f}", flush=True)

print(f"\nDifference: {l0_wrong - l0_correct:.1f} fewer active features!", flush=True)

# Check individual positions
print(f"\nPer position:", flush=True)
for i in range(activations.shape[0]):
    wrong = (features_wrong[i] > 0).sum().item()
    correct = (features_correct[i] > 0).sum().item()
    print(f"  Pos {i}: wrong={wrong:5d}, correct={correct:3d}", flush=True)

import os
os._exit(0)