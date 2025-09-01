#!/usr/bin/env python
"""Visualize the induction pattern for Layer 5, Head 0."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

# Load model
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')

# Create a repeated sequence
seq_len = 100
first_half = torch.randint(50, 1000, (seq_len//2,))
repeated_seq = torch.cat([first_half, first_half]).unsqueeze(0).to('cuda')

# Get attention pattern
target_layer = 5
target_head = 0

_, cache = model.run_with_cache(
    repeated_seq,
    names_filter=lambda name: name == f"blocks.{target_layer}.attn.hook_pattern"
)

attention_pattern = cache[f"blocks.{target_layer}.attn.hook_pattern"][0, target_head, :, :].cpu()

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Full attention pattern
im1 = ax1.imshow(attention_pattern.numpy(), cmap='hot', aspect='auto')
ax1.set_title(f'Layer {target_layer}, Head {target_head} - Full Attention Pattern')
ax1.set_xlabel('Key Position')
ax1.set_ylabel('Query Position')
ax1.axhline(y=50, color='cyan', linestyle='--', alpha=0.5, label='Sequence midpoint')
ax1.axvline(x=50, color='cyan', linestyle='--', alpha=0.5)
plt.colorbar(im1, ax=ax1)

# Zoom in on induction pattern
half_len = seq_len // 2
second_half_pattern = attention_pattern[half_len:, :half_len].numpy()
im2 = ax2.imshow(second_half_pattern, cmap='hot', aspect='auto')
ax2.set_title('Induction Pattern (2nd half → 1st half)')
ax2.set_xlabel('Key Position (1st half)')
ax2.set_ylabel('Query Position (2nd half)')

# Add diagonal line for perfect induction
ax2.plot([0, half_len-1], [0, half_len-1], 'c--', alpha=0.5, label='Perfect induction')
ax2.legend()
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('/root/fra/results/induction_pattern_L5H0.png', dpi=150, bbox_inches='tight')
print(f"Saved: /root/fra/results/induction_pattern_L5H0.png")

# Calculate diagonal strength
diagonal_scores = []
for i in range(half_len):
    diagonal_scores.append(attention_pattern[i + half_len, i].item())

print(f"\nDiagonal attention (induction) statistics:")
print(f"  Mean: {np.mean(diagonal_scores):.4f}")
print(f"  Max: {np.max(diagonal_scores):.4f}")
print(f"  Std: {np.std(diagonal_scores):.4f}")

plt.show()