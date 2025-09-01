#!/usr/bin/env python
"""Test basic FRA computation over 5 samples."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import time

torch.set_grad_enabled(False)

print("Testing basic FRA computation over 5 samples...")
print("="*60)

# Load model and SAE
print("\nLoading model and SAE...")
t0 = time.time()
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")
print(f"Loading time: {time.time()-t0:.1f}s")

# 5 test texts
test_texts = [
    "The cat sat on the mat. The cat was happy.",
    "Alice went to the store. She bought milk.",
    "The student who studied hard passed. The students who studied hard passed.",
    "Bob likes to play chess. Bob is very good at chess.",
    "The weather today is sunny. The weather yesterday was rainy."
]

print("\nComputing FRA for each sample (no accumulation)...")
total_time = 0
for i, text in enumerate(test_texts):
    print(f"\nSample {i+1}: '{text[:30]}...'")
    
    t0 = time.time()
    fra_result = get_sentence_fra_batch(
        model, sae, text, 
        layer=5, head=0,
        max_length=128, top_k=20, 
        verbose=False
    )
    t1 = time.time()
    
    sample_time = t1 - t0
    total_time += sample_time
    
    # Print stats and immediately discard
    print(f"  Time: {sample_time:.2f}s")
    print(f"  Shape: {fra_result['shape']}")
    print(f"  NNZ: {fra_result['fra_tensor_sparse']._nnz()}")
    
    # Explicitly delete to free memory
    del fra_result
    torch.cuda.empty_cache()

print("\n" + "="*60)
print(f"Total time for 5 samples: {total_time:.1f}s")
print(f"Average per sample: {total_time/5:.1f}s")
print("\n✅ This should be ~15-20s total (3-4s per sample)")