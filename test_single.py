"""
Single test of FRA to get actual results.
"""

import torch
from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis_fast import get_sentence_fra_sparse_fast
from fra.fra_analysis import get_top_feature_interactions

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

# Test induction pattern
text = "The cat sat. The cat ran."
head = 10  # Often an induction head

print(f"\nAnalyzing: '{text}'")
print(f"Head: {head}, Layer: 5")

# Compute FRA
fra_result = get_sentence_fra_sparse_fast(
    model, sae, text, 5, head,
    max_length=20, verbose=True
)

print(f"\nResults:")
print(f"  Sequence length: {fra_result['seq_len']} tokens")
print(f"  Non-zero interactions: {fra_result['nnz']:,}")
print(f"  Density: {fra_result['density']:.8%}")

# Get top interactions
top_int = get_top_feature_interactions(
    fra_result['data_dep_int_matrix'], top_k=15
)

print(f"\nTop 15 feature interactions:")
for i, (q, k, v) in enumerate(top_int):
    print(f"  {i+1:2d}. Feature {q:5d} → Feature {k:5d}: {v:10.6f}")

# Look for patterns in the features
print(f"\nAnalyzing interaction patterns...")

# Which features appear most often?
from collections import Counter
all_features = []
for q, k, v in top_int:
    all_features.extend([q, k])

feature_counts = Counter(all_features)
print(f"\nMost active features in top interactions:")
for feat, count in feature_counts.most_common(10):
    print(f"  Feature {feat:5d}: appears {count} times")

# Look for reciprocal interactions
print(f"\nChecking for reciprocal interactions (F_i → F_j and F_j → F_i):")
interactions = {(q, k): v for q, k, v in top_int}
for (q, k), v1 in interactions.items():
    if (k, q) in interactions:
        v2 = interactions[(k, q)]
        print(f"  F{q} ↔ F{k}: forward={v1:.4f}, backward={v2:.4f}")

import os
if torch.cuda.is_available():
    torch.cuda.empty_cache()
os._exit(0)