#!/usr/bin/env python
"""Test unified dashboard generation with 5 samples."""

import torch
from fra.generate_all_dashboards import generate_all_dashboards

torch.set_grad_enabled(False)

print("Testing unified dashboard generation with 5 samples...")
print("This will create:")
print("  1. Data-independent FRA dashboard")
print("  2. Dataset search dashboard (5 samples, no self-interactions)")
print("  3. Single sample dashboard")
print("  4. Package everything into results_package.tar.gz")
print()

# Generate all dashboards with 5 samples
results = generate_all_dashboards(
    layer=5,
    head=0,
    num_samples=5,
    sample_text="The student who studied hard passed the exam. The students who studied hard passed their exams.",
    filter_self_interactions=True,  # Filter out self-interactions
    create_package=True,
    verbose=True
)

print("\n" + "="*60)
print("🎉 Test Complete!")
print("="*60)
print("\nGenerated files:")
for key, path in results.items():
    print(f"  {key}: {path}")

print("\n📊 Key insights:")
print("  - Data-independent shows inherent feature biases")
print("  - Dataset search shows actual patterns across text")
print("  - Self-interactions filtered to focus on cross-feature patterns")
print("\n✅ All dashboards packaged and ready for download!")