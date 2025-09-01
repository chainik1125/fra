"""
Quick test to understand the actual timing.
"""

import torch
import time
from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis import get_sentence_fra_sparse
from fra.fra_analysis_fast import get_sentence_fra_sparse_fast

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

text = "Cat."  # Very short text - just 2 tokens
layer = 5
head = 0

print(f"\nTesting with text: '{text}'")

# Time the optimized version
print("\nOptimized version:")
t0 = time.time()
result_fast = get_sentence_fra_sparse_fast(
    model, sae, text, layer, head, 
    max_length=10, verbose=True
)
print(f"Time: {time.time() - t0:.2f}s")
print(f"Non-zero interactions: {result_fast['nnz']}")

import os
if torch.cuda.is_available():
    torch.cuda.empty_cache()
os._exit(0)