#!/usr/bin/env python
"""Test if Layer 5, Head 0 is an induction head using TransformerLens utilities."""

import torch
import numpy as np
from transformer_lens import HookedTransformer, utils
from transformer_lens.utils import get_act_name
import plotly.express as px
from tqdm import tqdm

torch.set_grad_enabled(False)

print("="*60)
print("Testing for Induction Heads in GPT-2 Small")
print("="*60)

# Load model
print("\nLoading GPT-2 Small...")
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")

# Create repeated token sequences for induction testing
def create_induction_prompt(seq_len=50, vocab_size=1000):
    """Create a sequence with repeated patterns to test induction."""
    # Create random sequence of tokens
    first_half = torch.randint(50, vocab_size, (seq_len//2,))
    # Repeat the sequence
    repeated_seq = torch.cat([first_half, first_half])
    return repeated_seq

# Test function for induction score
def get_induction_score(model, layer, head, prompts, verbose=False):
    """
    Calculate induction score for a specific head.
    Induction score = average attention paid to token at position (i - seq_len/2) from position i
    for the second half of a repeated sequence.
    """
    batch_size = prompts.shape[0]
    seq_len = prompts.shape[1]
    half_len = seq_len // 2
    
    # Run model and cache attention patterns
    _, cache = model.run_with_cache(
        prompts,
        names_filter=lambda name: name == f"blocks.{layer}.attn.hook_pattern"
    )
    
    # Get attention pattern for this head
    attention_pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head, :, :]  # [batch, seq, seq]
    
    # Calculate induction score
    # For each position in second half, check attention to corresponding position in first half
    induction_scores = []
    for i in range(half_len, seq_len):
        # Position i should attend to position (i - half_len)
        expected_pos = i - half_len
        # Get attention from position i to expected position
        attention_score = attention_pattern[:, i, expected_pos].mean().item()
        induction_scores.append(attention_score)
    
    avg_induction_score = np.mean(induction_scores)
    
    if verbose:
        print(f"  Layer {layer}, Head {head}: {avg_induction_score:.4f}")
    
    return avg_induction_score

# Generate test prompts
print("\nGenerating test sequences...")
num_prompts = 10
seq_len = 100  # 50 tokens repeated twice
test_prompts = []

for _ in range(num_prompts):
    prompt = create_induction_prompt(seq_len=seq_len)
    test_prompts.append(prompt)

test_prompts = torch.stack(test_prompts).to('cuda')
print(f"Created {num_prompts} test sequences of length {seq_len}")

# Test all heads to find induction heads
print("\nTesting all heads for induction behavior...")
print("(Higher scores indicate stronger induction behavior)")
print("-"*40)

all_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))

for layer in tqdm(range(model.cfg.n_layers), desc="Layers"):
    for head in range(model.cfg.n_heads):
        score = get_induction_score(model, layer, head, test_prompts, verbose=False)
        all_scores[layer, head] = score

# Find top induction heads
print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Flatten scores and get top heads
flat_scores = all_scores.flatten()
top_indices = np.argsort(flat_scores)[-20:][::-1]  # Top 20 heads

print("\nTop 20 Induction Heads:")
print("-"*40)
for i, idx in enumerate(top_indices):
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    score = flat_scores[idx]
    print(f"{i+1:2}. Layer {layer:2}, Head {head:2}: {score:.4f}")

# Check our specific head
print("\n" + "="*60)
print("TARGET HEAD ANALYSIS")
print("="*60)
target_layer = 5
target_head = 0
target_score = all_scores[target_layer, target_head]

print(f"\nLayer {target_layer}, Head {target_head} induction score: {target_score:.4f}")

# Calculate percentile
percentile = (flat_scores < target_score).sum() / len(flat_scores) * 100
print(f"This head is in the {percentile:.1f}th percentile of all heads")

if percentile > 90:
    print("✅ This head shows STRONG induction behavior!")
elif percentile > 70:
    print("⚠️ This head shows MODERATE induction behavior")
else:
    print("❌ This head shows WEAK induction behavior")

# Visualize attention pattern for our target head on one example
print("\n" + "="*60)
print("ATTENTION PATTERN VISUALIZATION")
print("="*60)

# Run a single example
single_prompt = test_prompts[0:1]
_, cache = model.run_with_cache(
    single_prompt,
    names_filter=lambda name: name == f"blocks.{target_layer}.attn.hook_pattern"
)

attention_pattern = cache[f"blocks.{target_layer}.attn.hook_pattern"][0, target_head, :, :].cpu()

# Save attention pattern
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Full attention pattern
im1 = ax1.imshow(attention_pattern.numpy(), cmap='hot', aspect='auto')
ax1.set_title(f'Layer {target_layer}, Head {target_head} - Full Attention Pattern')
ax1.set_xlabel('Key Position')
ax1.set_ylabel('Query Position')
plt.colorbar(im1, ax=ax1)

# Zoom in on second half attending to first half (induction pattern)
half_len = seq_len // 2
second_half_pattern = attention_pattern[half_len:, :half_len].numpy()
im2 = ax2.imshow(second_half_pattern, cmap='hot', aspect='auto')
ax2.set_title('Induction Pattern (2nd half → 1st half)')
ax2.set_xlabel('Key Position (1st half)')
ax2.set_ylabel('Query Position (2nd half)')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('/root/fra/results/induction_pattern_L5H0.png', dpi=150, bbox_inches='tight')
print(f"Saved attention pattern visualization to: /root/fra/results/induction_pattern_L5H0.png")

# Calculate diagonal strength (true induction)
diagonal_scores = []
for i in range(half_len):
    diagonal_scores.append(attention_pattern[i + half_len, i].item())

print(f"\nDiagonal attention strength (true induction):")
print(f"  Mean: {np.mean(diagonal_scores):.4f}")
print(f"  Max: {np.max(diagonal_scores):.4f}")
print(f"  Min: {np.min(diagonal_scores):.4f}")

# Create heatmap of all heads' induction scores
fig_heatmap = px.imshow(
    all_scores,
    labels=dict(x="Head", y="Layer", color="Induction Score"),
    title="Induction Scores Across All Heads",
    color_continuous_scale="Viridis"
)
fig_heatmap.write_html('/root/fra/results/induction_scores_heatmap.html')
print(f"Saved induction scores heatmap to: /root/fra/results/induction_scores_heatmap.html")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)