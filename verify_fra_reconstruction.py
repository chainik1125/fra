#!/usr/bin/env python
"""Verify that FRA actually reconstructs the original attention."""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch

torch.set_grad_enabled(False)

LAYER = 5
DEVICE = 'cuda'

print("="*60)
print("Verifying FRA Reconstruction")
print("="*60)

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

prompt = "The cat sat on the mat"
print(f"\nPrompt: '{prompt}'")

# Get the ACTUAL attention pattern from the model
tokens = model.tokenizer.encode(prompt)
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

# Capture the real attention pattern
real_patterns = {}

def capture_pattern(pattern, hook):
    # Store pattern for each head
    for head in range(pattern.shape[1]):
        real_patterns[head] = pattern[0, head, :, :].clone().cpu()
    return pattern

# Run model to capture real patterns
with model.hooks([(f"blocks.{LAYER}.attn.hook_pattern", capture_pattern)]):
    _ = model(input_ids)

print(f"\nCaptured real attention patterns for {len(real_patterns)} heads")

# Now compute FRA and reconstruct
print("\nComputing FRA for head 0...")
fra_result = get_sentence_fra_batch(
    model, sae, prompt,
    layer=LAYER, head=0,
    max_length=128, top_k=30,
    verbose=False
)

if fra_result:
    fra_sparse = fra_result['fra_tensor_sparse']
    indices = fra_sparse.indices()
    values = fra_sparse.values()
    seq_len = fra_result['seq_len']
    
    print(f"FRA has {len(values)} non-zero entries")
    print(f"Sequence length: {seq_len}")
    
    # Reconstruct attention scores from FRA
    reconstructed_scores = torch.zeros((seq_len, seq_len), device=DEVICE)
    for i in range(indices.shape[1]):
        q_pos = indices[0, i].item()
        k_pos = indices[1, i].item()
        reconstructed_scores[q_pos, k_pos] += values[i].item()
    
    print(f"\nReconstruction check (Head 0):")
    print(f"  FRA sum of values: {values.sum().item():.4f}")
    print(f"  Reconstructed scores sum: {reconstructed_scores.sum().item():.4f}")
    
    # Apply causal mask
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=DEVICE), diagonal=1).bool()
    reconstructed_scores.masked_fill_(causal_mask, float('-inf'))
    
    # Convert to pattern with softmax
    reconstructed_pattern = F.softmax(reconstructed_scores, dim=-1)
    reconstructed_pattern = torch.nan_to_num(reconstructed_pattern, 0.0)
    
    # Compare with real pattern
    real_pattern_head0 = real_patterns[0][:seq_len, :seq_len].to(DEVICE)
    
    # Calculate differences
    abs_diff = (reconstructed_pattern - real_pattern_head0).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    print(f"\nPattern comparison (Head 0):")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    
    # Show a sample position
    pos = min(5, seq_len-1)  # Look at position 5 or last position
    print(f"\nAt position {pos}:")
    print(f"  Real pattern: {real_pattern_head0[pos, :6].cpu().numpy()}")
    print(f"  FRA reconstructed: {reconstructed_pattern[pos, :6].cpu().numpy()}")
    
    # Check if reconstruction is accurate
    if mean_diff < 0.01:
        print("\n✅ FRA accurately reconstructs attention!")
    else:
        print("\n❌ FRA reconstruction has significant errors!")
        print("This explains why 'full FRA' gives different outputs than normal.")
    
    # Also check: what are the actual attention SCORES before softmax?
    print("\n" + "="*60)
    print("Checking attention scores (pre-softmax)...")
    
    def capture_scores(scores, hook):
        print(f"  Attention scores shape: {scores.shape}")
        print(f"  Head 0, position {pos} scores: {scores[0, 0, pos, :6].cpu().numpy()}")
        print(f"  Min score: {scores[0, 0, :seq_len, :seq_len].min().item():.4f}")
        print(f"  Max score: {scores[0, 0, :seq_len, :seq_len].max().item():.4f}")
        return scores
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", capture_scores)]):
        _ = model(input_ids)
    
    print(f"\nFRA reconstructed scores at position {pos}: {reconstructed_scores[pos, :6].cpu().numpy()}")
    print(f"FRA min score: {reconstructed_scores.min().item():.4f}")
    print(f"FRA max score: {reconstructed_scores.max().item():.4f}")

print("\n" + "="*60)