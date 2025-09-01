#!/usr/bin/env python
"""Check if FRA is missing the attention scaling factor."""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch

torch.set_grad_enabled(False)

LAYER = 5
DEVICE = 'cuda'

print("="*60)
print("Checking FRA Scaling")
print("="*60)

# Load model and SAE
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

prompt = "The cat sat on the mat"
tokens = model.tokenizer.encode(prompt)
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

# Get actual attention scores
captured_scores = None
def capture_scores(scores, hook):
    global captured_scores
    captured_scores = scores.clone()
    return scores

with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", capture_scores)]):
    _ = model(input_ids)

actual_scores = captured_scores[0, 0, :, :]  # Head 0
seq_len = len(tokens)

print(f"\nPrompt: '{prompt}' ({seq_len} tokens)")

# Get FRA
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
    
    # Reconstruct
    reconstructed = torch.zeros((seq_len, seq_len), device=DEVICE)
    for i in range(indices.shape[1]):
        q_pos = indices[0, i].item()
        k_pos = indices[1, i].item()
        reconstructed[q_pos, k_pos] += values[i].item()
    
    # Check what d_head is
    d_head = model.cfg.d_head
    print(f"\nModel d_head: {d_head}")
    print(f"Scaling factor (1/sqrt(d_head)): {1/d_head**0.5:.6f}")
    
    # Try scaling FRA by 1/sqrt(d_head)
    scaled_reconstructed = reconstructed / (d_head ** 0.5)
    
    print("\nComparison at position 5:")
    print(f"  Actual scores:      {actual_scores[5, :6].cpu().numpy()}")
    print(f"  FRA (no scaling):   {reconstructed[5, :6].cpu().numpy()}")
    print(f"  FRA (with scaling): {scaled_reconstructed[5, :6].cpu().numpy()}")
    
    # Calculate errors
    mask = actual_scores[:seq_len, :seq_len] > -1000  # Exclude -inf
    
    if mask.any():
        error_no_scale = (reconstructed[mask] - actual_scores[:seq_len, :seq_len][mask]).abs()
        error_with_scale = (scaled_reconstructed[mask] - actual_scores[:seq_len, :seq_len][mask]).abs()
        
        print("\nReconstruction errors (excluding -inf positions):")
        print(f"  Without scaling: max={error_no_scale.max():.4f}, mean={error_no_scale.mean():.4f}")
        print(f"  With scaling:    max={error_with_scale.max():.4f}, mean={error_with_scale.mean():.4f}")
        
        if error_with_scale.mean() < error_no_scale.mean():
            print("\n✅ Scaling by 1/sqrt(d_head) improves reconstruction!")
        else:
            print("\n❌ Scaling doesn't fix the issue")
    
    # Also check if there's another systematic issue
    print("\n" + "="*60)
    print("Checking for systematic differences...")
    
    # Compare ranges
    print(f"\nValue ranges:")
    print(f"  Actual scores (non-inf): [{actual_scores[mask].min():.4f}, {actual_scores[mask].max():.4f}]")
    print(f"  FRA reconstructed:       [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    print(f"  FRA scaled:              [{scaled_reconstructed.min():.4f}, {scaled_reconstructed.max():.4f}]")
    
    # Check a specific position pair
    q_pos, k_pos = 5, 2  # Example position
    print(f"\nDetailed check for position ({q_pos}, {k_pos}):")
    print(f"  Actual score: {actual_scores[q_pos, k_pos]:.6f}")
    print(f"  FRA (scaled): {scaled_reconstructed[q_pos, k_pos]:.6f}")
    
    # Count how many feature pairs contribute to this position
    count = 0
    total_contribution = 0
    for i in range(indices.shape[1]):
        if indices[0, i] == q_pos and indices[1, i] == k_pos:
            count += 1
            total_contribution += values[i].item()
    
    print(f"  Number of feature pairs: {count}")
    print(f"  Sum of contributions: {total_contribution:.6f}")
    print(f"  After scaling: {total_contribution / (d_head**0.5):.6f}")

print("\n" + "="*60)