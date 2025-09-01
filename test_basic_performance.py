#!/usr/bin/env python
"""Test basic FRA performance - just compute and discard."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
from fra.utils import load_dataset_hf
import time

torch.set_grad_enabled(False)

print("="*60)
print("Testing Basic FRA Performance (compute and discard)")
print("="*60)

# Load model and SAE
print("\nLoading model...")
t0 = time.time()
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
print(f"Model loaded in {time.time()-t0:.1f}s")

print("\nLoading SAE...")
t0 = time.time()
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', 'blocks.5.hook_z', device='cuda')
print(f"SAE loaded in {time.time()-t0:.1f}s")

# Load 5 samples from dataset
print("\nLoading 5 samples from dataset...")
t0 = time.time()
dataset = load_dataset_hf(streaming=True)
dataset_texts = []
for i, item in enumerate(dataset):
    if i >= 5:
        break
    text = item.get('text', '') if isinstance(item, dict) else str(item)
    if len(text.split()) > 10:
        dataset_texts.append(text[:500])
print(f"Loaded {len(dataset_texts)} texts in {time.time()-t0:.1f}s")

# Test raw FRA computation speed
print("\n" + "="*60)
print("Testing FRA computation speed (no accumulation)...")
print("="*60)

times = []
for i, text in enumerate(dataset_texts):
    print(f"\nSample {i+1}:")
    t0 = time.time()
    
    # Just compute FRA and discard
    fra_result = get_sentence_fra_batch(
        model, sae, text, 
        layer=5, head=0,
        max_length=128, top_k=20, 
        verbose=False
    )
    
    t1 = time.time()
    elapsed = t1 - t0
    times.append(elapsed)
    
    if fra_result:
        nnz = fra_result['total_interactions']
        seq_len = fra_result['seq_len']
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Sequence length: {seq_len}")
        print(f"  Non-zero interactions: {nnz:,}")
    
    # Explicitly delete and clear memory
    del fra_result
    torch.cuda.empty_cache()

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Samples processed: {len(times)}")
print(f"Average time per sample: {sum(times)/len(times):.2f}s")
print(f"Min time: {min(times):.2f}s")
print(f"Max time: {max(times):.2f}s")
print(f"Total time: {sum(times):.2f}s")