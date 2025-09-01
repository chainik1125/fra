#!/usr/bin/env python
"""Test accumulated features dashboard with 10 samples."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_average_sparse import compute_dataset_average_sparse
from fra.accumulated_features_viz import generate_accumulated_dashboard_from_results
from fra.utils import load_dataset_hf
import time

torch.set_grad_enabled(False)

print("="*60)
print("Testing Accumulated Features Dashboard with 10 samples")
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

# Load 10 samples from dataset
print("\nLoading 10 samples from dataset...")
t0 = time.time()
dataset = load_dataset_hf(streaming=True)
dataset_texts = []
for i, item in enumerate(dataset):
    if i >= 10:
        break
    text = item.get('text', '') if isinstance(item, dict) else str(item)
    if len(text.split()) > 10:
        dataset_texts.append(text[:500])
print(f"Loaded {len(dataset_texts)} texts in {time.time()-t0:.1f}s")

# Run accumulation
print("\n" + "="*60)
print("Computing FRA accumulation...")
print("="*60)
t0 = time.time()

results = compute_dataset_average_sparse(
    model=model,
    sae=sae,
    dataset_texts=dataset_texts,
    layer=5,
    head=0,
    filter_self_interactions=False,  # Keep self-interactions to find potential induction
    use_absolute_values=True,
    verbose=True
)

t1 = time.time()
print(f"\n✅ Accumulation complete in {t1-t0:.1f}s")

# Generate dashboard
print("\n" + "="*60)
print("Generating interactive dashboard...")
print("="*60)

dashboard_path = generate_accumulated_dashboard_from_results(
    results=results,
    layer=5,
    head=0,
    top_k=100,  # Show top 100 feature pairs
    fetch_neuronpedia=True  # Fetch descriptions from Neuronpedia
)

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print(f"Dashboard saved to: {dashboard_path}")
print(f"Total time: {time.time()-t0:.1f}s")