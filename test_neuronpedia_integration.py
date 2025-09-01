#!/usr/bin/env python
"""Test that Neuronpedia feature descriptions are integrated in dashboard."""

from fra.single_sample_viz import generate_dashboard_from_config
import torch

torch.set_grad_enabled(False)

print("Testing Neuronpedia integration...")

# Generate dashboard with a simple test case
dashboard_path = generate_dashboard_from_config(
    text="The cat sat on the mat. The dog sat on the rug.",
    layer=5,
    head=0,
    top_k_features=5,
    top_k_interactions=3,
    use_timestamp=False
)

print(f"\n✅ Dashboard generated: {dashboard_path}")
print("Feature descriptions should now appear directly in the feature boxes.")
print("Open the HTML file to verify Neuronpedia explanations are displayed.")