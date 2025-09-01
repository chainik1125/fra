#!/usr/bin/env python
"""Quick dashboard generation without Neuronpedia fetching."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_average_sparse import compute_dataset_average_sparse
from fra.accumulated_features_viz import generate_accumulated_dashboard_from_results
from fra.utils import load_dataset_hf
import time

torch.set_grad_enabled(False)

print("="*60)
print("Quick FRA Dashboard (no Neuronpedia fetching)")
print("="*60)

# Configuration
NUM_SAMPLES = 10  # Use 10 samples for quick test
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

# Run accumulation
print("\n" + "="*60)
print(f"Computing FRA accumulation...")
print("="*60)
t0 = time.time()

results = compute_dataset_average_sparse(
    model=model,
    sae=sae,
    dataset_texts=dataset_texts,
    layer=LAYER,
    head=HEAD,
    filter_self_interactions=False,
    use_absolute_values=True,
    verbose=True
)

t1 = time.time()
print(f"\n✅ Accumulation complete in {t1-t0:.1f}s ({(t1-t0)/len(dataset_texts):.1f}s per sample)")

# Generate dashboard WITHOUT Neuronpedia fetching
print("\n" + "="*60)
print("Generating dashboard (no Neuronpedia)...")
print("="*60)

dashboard_path = generate_accumulated_dashboard_from_results(
    results=results,
    layer=LAYER,
    head=HEAD,
    top_k=50,  # Show top 50 pairs
    fetch_neuronpedia=False  # DISABLE Neuronpedia fetching
)

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print(f"Dashboard saved to: {dashboard_path}")