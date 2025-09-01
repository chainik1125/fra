"""
Test FRA with threshold for near-zero features.
"""

import torch
from fra.utils import load_model, load_sae, load_dataset_hf
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis_topk import get_sentence_fra_topk, get_top_interactions_torch
from fra.activation_utils import get_attention_activations

torch.set_grad_enabled(False)

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

# Get a sample text
text = "The cat sat on the mat. The cat was happy."

print(f"\nText: '{text}'")

# Get activations
activations = get_attention_activations(model, text, layer=5, max_length=128)
features_raw = sae.encode(activations)

print(f"\nFeature statistics (raw):")
print(f"  Shape: {features_raw.shape}")

# Check sparsity with different thresholds
thresholds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
for thresh in thresholds:
    active = (features_raw.abs() > thresh).sum(1).float().mean().item()
    sparsity = 1.0 - active / features_raw.shape[1]
    print(f"  Threshold {thresh:8.0e}: {active:7.1f} active ({sparsity:.1%} sparse)")

# Apply threshold of 1e-4
print(f"\nApplying threshold of 1e-4...")
features_thresholded = features_raw.clone()
features_thresholded[features_raw.abs() < 1e-4] = 0

active_after = (features_thresholded != 0).sum(1).float().mean().item()
sparsity_after = 1.0 - active_after / features_thresholded.shape[1]
print(f"  After threshold: {active_after:.1f} active ({sparsity_after:.1%} sparse)")

# Now test FRA with thresholded features
print(f"\nTesting FRA with thresholded features...")

# Monkey-patch the encode function to apply threshold
original_encode = sae.encode
def encode_with_threshold(x):
    features = original_encode(x)
    features[features.abs() < 1e-4] = 0
    return features

sae.encode = encode_with_threshold

# Run FRA
fra_result = get_sentence_fra_topk(
    model, sae, text,
    layer=5, head=10,
    max_length=128,
    top_k=20,
    verbose=True
)

print(f"\nResults:")
print(f"  Non-zero interactions: {fra_result['nnz']:,}")
print(f"  Density: {fra_result['density']:.8%}")

# Top interactions
top = get_top_interactions_torch(fra_result['data_dep_int_matrix'], top_k=10)
print(f"\nTop 10 interactions:")
for i, (q, k, v) in enumerate(top):
    print(f"  {i+1}. F{q:5d} → F{k:5d}: {v:8.4f}")

# Check for self-interactions
self_int = [(q, k, v) for q, k, v in top if q == k]
if self_int:
    print(f"\nSelf-interactions (potential induction):")
    for q, k, v in self_int:
        print(f"  F{q}: {v:.4f}")

import os
os._exit(0)