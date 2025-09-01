"""
Quick test of FRA on dataset sample.
"""

import torch
import sys
from fra.utils import load_model, load_sae, load_dataset_hf
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis_topk import get_sentence_fra_topk, get_top_interactions_torch

torch.set_grad_enabled(False)

print("Loading model and SAE...", flush=True)
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

print("Loading dataset...", flush=True)
dataset = load_dataset_hf("Elriggs/openwebtext-100k", split="train", streaming=True, seed=42)

# Get sample
sample = next(iter(dataset))
text = sample['text']

# Truncate
tokens = model.tokenizer.encode(text)[:50]  # Shorter for quick test
text = model.tokenizer.decode(tokens)

print(f"\nSample ({len(tokens)} tokens):", flush=True)
print(text[:200], flush=True)

# Test one head
head = 10  # Often an induction head
print(f"\nTesting Head {head}...", flush=True)

fra_result = get_sentence_fra_topk(
    model, sae, text,
    layer=5, head=head,
    max_length=128,
    top_k=20,
    verbose=True
)

print(f"\nResults:", flush=True)
print(f"  Non-zero interactions: {fra_result['nnz']:,}", flush=True)
print(f"  Density: {fra_result['density']:.8%}", flush=True)

# Top interactions
top = get_top_interactions_torch(fra_result['data_dep_int_matrix'], top_k=10)
print(f"\nTop 10 interactions:", flush=True)
for i, (q, k, v) in enumerate(top):
    print(f"  {i+1}. F{q:5d} → F{k:5d}: {v:8.4f}", flush=True)

# Check for self-interactions
self_int = [(q, k, v) for q, k, v in top if q == k]
if self_int:
    print(f"\nSelf-interactions (potential induction):", flush=True)
    for q, k, v in self_int:
        print(f"  F{q}: {v:.4f}", flush=True)

import os
os._exit(0)