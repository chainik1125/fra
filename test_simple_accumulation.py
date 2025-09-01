#!/usr/bin/env python
"""Test simplified accumulation and dashboard."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_average_simple import compute_dataset_average_simple, create_simple_dashboard
from fra.utils import load_dataset_hf
import time

torch.set_grad_enabled(False)

print("="*60)
print("Testing Simplified Accumulation")
print("="*60)

# Configuration
NUM_SAMPLES = 10
LAYER = 5
HEAD = 0

# Load model and SAE
print("\nLoading model...")
t0 = time.time()
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
print(f"Model loaded in {time.time()-t0:.1f}s")

print("\nLoading SAE...")
t0 = time.time()
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device='cuda')
print(f"SAE loaded in {time.time()-t0:.1f}s")

# Load samples
print(f"\nLoading {NUM_SAMPLES} samples...")
t0 = time.time()
dataset = load_dataset_hf(streaming=True)
dataset_texts = []
for i, item in enumerate(dataset):
    if i >= NUM_SAMPLES:
        break
    text = item.get('text', '') if isinstance(item, dict) else str(item)
    if len(text.split()) > 10:
        dataset_texts.append(text[:500])
print(f"Loaded {len(dataset_texts)} texts in {time.time()-t0:.1f}s")

# Run simplified accumulation
print("\n" + "="*60)
print("Running simplified accumulation...")
print("="*60)
t0 = time.time()

results = compute_dataset_average_simple(
    model=model,
    sae=sae,
    dataset_texts=dataset_texts,
    layer=LAYER,
    head=HEAD,
    filter_self_interactions=False,  # Keep self-interactions
    use_absolute_values=True,
    verbose=True,
    top_k=100  # Get top 100 pairs
)

t1 = time.time()
print(f"\n✅ Complete in {t1-t0:.1f}s ({(t1-t0)/len(dataset_texts):.1f}s per sample)")

# Generate simple dashboard
print("\nGenerating dashboard...")
dashboard_path = create_simple_dashboard(results)

print("\n" + "="*60)
print("✅ SUCCESS!")
print("="*60)
print(f"Processing time: {t1-t0:.1f}s")
print(f"Dashboard: {dashboard_path}")