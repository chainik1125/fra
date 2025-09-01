#!/usr/bin/env python
"""Debug why FRA doesn't reconstruct attention correctly."""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch

torch.set_grad_enabled(False)

LAYER = 5
DEVICE = 'cuda'

print("="*60)
print("Debugging FRA Reconstruction")
print("="*60)

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

prompt = "The cat sat on the mat"
print(f"\nPrompt: '{prompt}'")

tokens = model.tokenizer.encode(prompt)
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

# First, let's capture the ACTUAL QK^T computation
captured_q = None
captured_k = None
captured_scores = None

def capture_qk(module_output, hook):
    global captured_q, captured_k, captured_scores
    if "hook_q" in hook.name:
        captured_q = module_output.clone()
        print(f"  Captured Q shape: {captured_q.shape}")
    elif "hook_k" in hook.name:
        captured_k = module_output.clone()
        print(f"  Captured K shape: {captured_k.shape}")
    elif "hook_attn_scores" in hook.name:
        captured_scores = module_output.clone()
        print(f"  Captured attention scores shape: {captured_scores.shape}")
    return module_output

print("\nCapturing Q, K, and attention scores...")
with model.hooks([
    (f"blocks.{LAYER}.attn.hook_q", capture_qk),
    (f"blocks.{LAYER}.attn.hook_k", capture_qk),
    (f"blocks.{LAYER}.attn.hook_attn_scores", capture_qk)
]):
    _ = model(input_ids)

# Compute QK^T manually
print("\nManual QK^T computation:")
# Q shape: [batch, seq, heads, d_head]
# K shape: [batch, seq, heads, d_head]
# Need to compute: Q @ K^T for each head

# Focus on head 0
q_head0 = captured_q[0, :, 0, :]  # [seq, d_head]
k_head0 = captured_k[0, :, 0, :]  # [seq, d_head]

# Compute QK^T
manual_scores = q_head0 @ k_head0.T / (q_head0.shape[-1] ** 0.5)  # Scale by sqrt(d_head)
print(f"  Manual QK^T shape: {manual_scores.shape}")
print(f"  Manual scores at position 5: {manual_scores[5, :6].cpu().numpy()}")

# Compare with captured scores
actual_scores = captured_scores[0, 0, :, :]  # Head 0
print(f"  Actual scores at position 5: {actual_scores[5, :6].cpu().numpy()}")

score_diff = (manual_scores - actual_scores[:6, :6]).abs().max().item()
print(f"  Max difference between manual and actual: {score_diff:.6f}")

# Now let's check what FRA gives us
print("\n" + "="*60)
print("FRA Reconstruction:")

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
    
    print(f"  FRA sparse tensor: {len(values)} entries")
    print(f"  Sequence length: {seq_len}")
    
    # Reconstruct by summing over feature dimensions
    reconstructed = torch.zeros((seq_len, seq_len), device=DEVICE)
    for i in range(indices.shape[1]):
        q_pos = indices[0, i].item()
        k_pos = indices[1, i].item()
        reconstructed[q_pos, k_pos] += values[i].item()
    
    print(f"\n  FRA reconstructed at position 5: {reconstructed[5, :6].cpu().numpy()}")
    print(f"  Actual scores at position 5: {actual_scores[5, :6].cpu().numpy()}")
    
    # Check reconstruction error
    reconstruction_error = (reconstructed - actual_scores[:seq_len, :seq_len]).abs()
    print(f"\n  Max reconstruction error: {reconstruction_error.max().item():.6f}")
    print(f"  Mean reconstruction error: {reconstruction_error.mean().item():.6f}")
    
    # Let's also check the scale
    print(f"\n  Scale comparison:")
    print(f"    Actual scores range: [{actual_scores[:seq_len, :seq_len].min().item():.4f}, {actual_scores[:seq_len, :seq_len].max().item():.4f}]")
    print(f"    FRA reconstructed range: [{reconstructed.min().item():.4f}, {reconstructed.max().item():.4f}]")
    
    # Check if there's a scaling factor
    # Mask out -inf values for comparison
    mask = actual_scores[:seq_len, :seq_len] > -1000
    if mask.any():
        actual_masked = actual_scores[:seq_len, :seq_len][mask]
        reconstructed_masked = reconstructed[mask]
        
        # Calculate potential scale factor
        scale_factor = (actual_masked / (reconstructed_masked + 1e-8)).mean().item()
        print(f"\n  Potential scale factor: {scale_factor:.4f}")
        
        # Try scaling and check error
        scaled_reconstructed = reconstructed * scale_factor
        scaled_error = (scaled_reconstructed[mask] - actual_masked).abs()
        print(f"  After scaling by {scale_factor:.4f}:")
        print(f"    Max error: {scaled_error.max().item():.6f}")
        print(f"    Mean error: {scaled_error.mean().item():.6f}")

print("\n" + "="*60)
print("Key question: Is FRA computing Q_SAE @ K_SAE^T correctly?")
print("The issue might be in how the SAE features are being combined.")
print("="*60)