"""
Debug FRA computation to find bottleneck.
"""

import torch
import time
from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations

print("Loading model...")
t0 = time.time()
model = load_model('gpt2-small', 'cuda')
print(f"Model loaded in {time.time() - t0:.1f}s")

print("Loading SAE...")
t0 = time.time()
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)
print(f"SAE loaded in {time.time() - t0:.1f}s")

text = "Hi."
layer = 5
head = 0

print(f"\nProcessing: '{text}'")

# Step 1: Get activations
print("Getting activations...")
t0 = time.time()
activations = get_attention_activations(model, text, layer=layer, max_length=10)
print(f"  Done in {time.time() - t0:.3f}s, shape: {activations.shape}")

# Step 2: Encode
print("Encoding to features...")
t0 = time.time()
features = sae.encode(activations)
print(f"  Done in {time.time() - t0:.3f}s, shape: {features.shape}")

# Step 3: Find active features
seq_len = features.shape[0]
print(f"\nChecking {seq_len} positions:")
for i in range(seq_len):
    active = torch.where(features[i] != 0)[0]
    print(f"  Position {i}: {len(active)} active features")

# Step 4: Single position pair
print(f"\nComputing single position pair (0, 1):")
t0 = time.time()

q_feat = features[1]
k_feat = features[0]
q_active = torch.where(q_feat != 0)[0]
k_active = torch.where(k_feat != 0)[0]

print(f"  Query: {len(q_active)} active")
print(f"  Key: {len(k_active)} active")

if len(q_active) > 0 and len(k_active) > 0:
    # Matrices
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    
    # Decoder vecs
    q_vecs = sae.W_dec[q_active]
    k_vecs = sae.W_dec[k_active]
    
    # Attention
    q = torch.einsum('da,nd->na', W_Q, q_vecs)
    k = torch.einsum('da,nd->na', W_K, k_vecs)
    int_mat = torch.einsum('qa,ka->qk', q, k)
    
    print(f"  Computed in {time.time() - t0:.3f}s")
    print(f"  Interaction matrix: {int_mat.shape}")
    print(f"  Max value: {int_mat.abs().max().item():.4f}")

import os
if torch.cuda.is_available():
    torch.cuda.empty_cache()
os._exit(0)