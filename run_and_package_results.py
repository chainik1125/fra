#!/usr/bin/env python
"""Run accumulation, generate dashboard, and package results."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.dataset_average_simple import compute_dataset_average_simple, create_simple_dashboard
from fra.utils import load_dataset_hf
import time
import tarfile
from pathlib import Path

torch.set_grad_enabled(False)

print("="*60)
print("Running FRA Analysis and Packaging Results")
print("="*60)

# Configuration
NUM_SAMPLES = 20  # Process 20 samples for meaningful results
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
print("Computing FRA accumulation...")
print("="*60)
t0 = time.time()

results = compute_dataset_average_simple(
    model=model,
    sae=sae,
    dataset_texts=dataset_texts,
    layer=LAYER,
    head=HEAD,
    filter_self_interactions=False,  # Keep self-interactions for induction analysis
    use_absolute_values=True,
    verbose=True,
    top_k=200  # Get top 200 pairs
)

t1 = time.time()
print(f"\n✅ Accumulation complete in {t1-t0:.1f}s ({(t1-t0)/len(dataset_texts):.1f}s per sample)")

# Generate main dashboard
print("\nGenerating dashboards...")
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Main dashboard
main_dashboard = create_simple_dashboard(
    results, 
    output_path=results_dir / f"accumulated_fra_L{LAYER}H{HEAD}_{NUM_SAMPLES}samples.html"
)

# Create a summary report
summary_path = results_dir / "analysis_summary.txt"
with open(summary_path, 'w') as f:
    f.write("Feature-Resolved Attention Analysis Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  Model: GPT-2 Small\n")
    f.write(f"  Layer: {LAYER}\n")
    f.write(f"  Head: {HEAD}\n")
    f.write(f"  Samples analyzed: {NUM_SAMPLES}\n")
    f.write(f"  Processing time: {t1-t0:.1f}s ({(t1-t0)/NUM_SAMPLES:.1f}s per sample)\n\n")
    
    f.write(f"Results:\n")
    f.write(f"  Total unique feature pairs: {results['total_pairs']:,}\n")
    f.write(f"  Top pairs extracted: {len(results['top_pairs'])}\n\n")
    
    # Count self-interactions
    self_interactions = [p for p in results['top_pairs'] if p['query_feature'] == p['key_feature']]
    f.write(f"  Self-interactions in top pairs: {len(self_interactions)}\n\n")
    
    f.write("Top 20 Feature Interactions:\n")
    f.write("-"*50 + "\n")
    for i, pair in enumerate(results['top_pairs'][:20]):
        is_self = pair['query_feature'] == pair['key_feature']
        f.write(f"{i+1:3}. F{pair['query_feature']:5} → F{pair['key_feature']:5} ")
        f.write(f"{'(self)' if is_self else '      '} ")
        f.write(f"avg={pair['average']:.4f} count={pair['count']:.0f}\n")
    
    f.write("\n" + "="*50 + "\n")
    f.write("Analysis complete!\n")

print(f"Summary saved to: {summary_path}")

# Package results
print("\n" + "="*60)
print("Packaging results...")
print("="*60)

package_path = results_dir / "results_package.tar.gz"

with tarfile.open(package_path, "w:gz") as tar:
    # Add all HTML files
    for html_file in results_dir.glob("*.html"):
        tar.add(html_file, arcname=html_file.name)
        print(f"  Added: {html_file.name}")
    
    # Add summary
    if summary_path.exists():
        tar.add(summary_path, arcname=summary_path.name)
        print(f"  Added: {summary_path.name}")
    
    # Add top pairs as JSON for further analysis
    import json
    json_path = results_dir / "top_pairs.json"
    with open(json_path, 'w') as f:
        json.dump({
            'layer': LAYER,
            'head': HEAD,
            'num_samples': NUM_SAMPLES,
            'total_pairs': results['total_pairs'],
            'top_pairs': results['top_pairs']
        }, f, indent=2)
    tar.add(json_path, arcname=json_path.name)
    print(f"  Added: {json_path.name}")

print(f"\n✅ Results package created: {package_path}")
print(f"📥 Download with: scp remote:{package_path} ./")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)
print(f"Dashboard: {main_dashboard}")
print(f"Package: {package_path}")
print(f"Total time: {time.time()-t0:.1f}s")