#!/usr/bin/env python
"""Minimal test with mock data to verify search pipeline."""

import torch
from fra.dataset_search import DatasetSearchResults, InteractionExample
from fra.dataset_search_viz import create_dashboard_from_search

print("Testing search visualization with mock data...")

# Create mock search results
d_sae = 47488
results = DatasetSearchResults(
    avg_interactions=torch.zeros(d_sae, d_sae),
    interaction_counts=torch.zeros(d_sae, d_sae, dtype=torch.int32),
    top_examples={},
    layer=5,
    head=0,
    num_samples=3,
    d_sae=d_sae
)

# Add some mock interactions
results.avg_interactions[100, 200] = 0.5
results.avg_interactions[300, 400] = 0.8
results.avg_interactions[500, 500] = 0.3
results.interaction_counts[100, 200] = 2
results.interaction_counts[300, 400] = 3
results.interaction_counts[500, 500] = 1

# Add mock examples
results.top_examples[(100, 200)] = [
    InteractionExample(
        sample_idx=0,
        text="The cat sat on the mat.",
        query_pos=1, key_pos=5,
        strength=0.6,
        query_token="cat", key_token="mat"
    )
]

results.top_examples[(300, 400)] = [
    InteractionExample(
        sample_idx=1,
        text="Alice went to the store.",
        query_pos=0, key_pos=3,
        strength=0.9,
        query_token="Alice", key_token="the"
    )
]

print("Creating dashboard from mock data...")
dashboard = create_dashboard_from_search(
    results=results,
    top_k_interactions=5,
    use_timestamp=False
)

print(f"✅ Dashboard created: {dashboard}")
print("This verifies the visualization pipeline works!")