#!/usr/bin/env python
"""Test the effect of ablating top feature pairs on model output."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
from typing import Dict, Tuple, List
import torch.nn.functional as F

torch.set_grad_enabled(False)

print("="*60)
print("Feature Pair Ablation Experiment")
print("="*60)

# Configuration
LAYER = 5
DEVICE = 'cuda'

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)
print(f"Model and SAE loaded for Layer {LAYER}")

def find_top_feature_pair_per_head(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    text: str,
    layer: int,
    exclude_self: bool = True,
    verbose: bool = True
) -> Dict[int, Dict]:
    """
    Find the top feature pair for each head in a layer.
    
    Returns:
        Dictionary mapping head index to top feature pair info
    """
    n_heads = model.cfg.n_heads
    top_pairs = {}
    
    if verbose:
        print(f"\nFinding top feature pairs for {n_heads} heads in layer {layer}...")
    
    for head in range(n_heads):
        # Compute FRA for this head
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=20, 
            verbose=False
        )
        
        if fra_result is None or 'fra_tensor_sparse' not in fra_result:
            continue
            
        # Extract feature pairs and their values
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()  # [4, nnz]
        values = fra_sparse.values()    # [nnz]
        
        # Get feature indices
        q_feats = indices[2, :]
        k_feats = indices[3, :]
        
        # Filter self-interactions if requested
        if exclude_self:
            mask = q_feats != k_feats
            q_feats = q_feats[mask]
            k_feats = k_feats[mask]
            values = values[mask]
        
        if len(values) == 0:
            continue
        
        # Average values across all position pairs for each feature pair
        feature_pair_sums = {}
        feature_pair_counts = {}
        
        for i in range(len(values)):
            pair = (q_feats[i].item(), k_feats[i].item())
            if pair not in feature_pair_sums:
                feature_pair_sums[pair] = 0
                feature_pair_counts[pair] = 0
            feature_pair_sums[pair] += values[i].abs().item()
            feature_pair_counts[pair] += 1
        
        # Find top pair by average absolute value
        if feature_pair_sums:
            top_pair = max(feature_pair_sums.keys(), 
                          key=lambda p: feature_pair_sums[p] / feature_pair_counts[p])
            
            top_pairs[head] = {
                'feature_i': top_pair[0],
                'feature_j': top_pair[1],
                'avg_strength': feature_pair_sums[top_pair] / feature_pair_counts[top_pair],
                'count': feature_pair_counts[top_pair],
                'total_sum': feature_pair_sums[top_pair]
            }
            
            if verbose:
                print(f"  Head {head}: F{top_pair[0]} → F{top_pair[1]}, "
                      f"avg={top_pairs[head]['avg_strength']:.4f}")
    
    return top_pairs

def reconstruct_attention_with_ablation(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    text: str,
    layer: int,
    head: int,
    ablate_feature_i: int,
    ablate_feature_j: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct attention pattern with and without a specific feature pair.
    
    Returns:
        (original_pattern, ablated_pattern) both of shape [seq_len, seq_len]
    """
    # Get tokens
    tokens = model.tokenizer.encode(text)
    if len(tokens) > 128:
        tokens = tokens[:128]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    seq_len = len(tokens)
    
    # Get attention activations
    hook_name = f"blocks.{layer}.attn.hook_z"
    _, cache = model.run_with_cache(tokens_tensor, names_filter=[hook_name])
    z = cache[hook_name].squeeze(0)  # [seq_len, n_heads, d_head]
    z_flat = z.flatten(-2, -1)  # [seq_len, 768]
    
    # Encode to SAE features
    feature_acts = sae.encode(z_flat)  # [seq_len, d_sae]
    
    # Get attention weights
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    W_dec = sae.W_dec
    
    # Reconstruct full attention pattern
    original_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
    ablated_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
    
    for q_pos in range(seq_len):
        for k_pos in range(seq_len):
            # Get active features at these positions
            q_feat_acts = feature_acts[q_pos]
            k_feat_acts = feature_acts[k_pos]
            
            q_active = torch.where(q_feat_acts != 0)[0]
            k_active = torch.where(k_feat_acts != 0)[0]
            
            if len(q_active) == 0 or len(k_active) == 0:
                continue
            
            # Compute attention contributions from all feature pairs
            for q_feat_idx in q_active:
                for k_feat_idx in k_active:
                    # Get decoder vectors
                    q_vec = W_dec[q_feat_idx]
                    k_vec = W_dec[k_feat_idx]
                    
                    # Project through attention weights
                    q_proj = torch.matmul(q_vec, W_Q)
                    k_proj = torch.matmul(k_vec, W_K)
                    
                    # Compute attention score
                    att_score = torch.dot(q_proj, k_proj)
                    
                    # Scale by feature activations
                    scaled_score = att_score * q_feat_acts[q_feat_idx] * k_feat_acts[k_feat_idx]
                    
                    # Add to original pattern
                    original_pattern[q_pos, k_pos] += scaled_score
                    
                    # Add to ablated pattern unless it's the target pair
                    if not (q_feat_idx.item() == ablate_feature_i and 
                           k_feat_idx.item() == ablate_feature_j):
                        ablated_pattern[q_pos, k_pos] += scaled_score
    
    # Apply softmax to get attention probabilities
    original_pattern = F.softmax(original_pattern / np.sqrt(64), dim=-1)
    ablated_pattern = F.softmax(ablated_pattern / np.sqrt(64), dim=-1)
    
    return original_pattern, ablated_pattern

def run_model_with_ablated_attention(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
    original_pattern: torch.Tensor,
    ablated_pattern: torch.Tensor
) -> Tuple[str, str]:
    """
    Run the model with original and ablated attention patterns.
    
    Returns:
        (original_output, ablated_output) - the generated text
    """
    tokens = model.tokenizer.encode(text)
    if len(tokens) > 128:
        tokens = tokens[:128]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Hook to replace attention pattern
    def attention_pattern_hook(pattern, hook, head_idx, new_pattern):
        if hook.name == f"blocks.{layer}.attn.hook_pattern":
            pattern[:, head_idx, :, :] = new_pattern.unsqueeze(0)
        return pattern
    
    # Generate with original pattern
    original_hook = lambda p, h: attention_pattern_hook(p, h, head, original_pattern)
    with model.hooks([(f"blocks.{layer}.attn.hook_pattern", original_hook)]):
        original_logits = model(tokens_tensor)
        original_next_token = torch.argmax(original_logits[0, -1, :]).item()
        original_next = model.tokenizer.decode([original_next_token])
    
    # Generate with ablated pattern
    ablated_hook = lambda p, h: attention_pattern_hook(p, h, head, ablated_pattern)
    with model.hooks([(f"blocks.{layer}.attn.hook_pattern", ablated_hook)]):
        ablated_logits = model(tokens_tensor)
        ablated_next_token = torch.argmax(ablated_logits[0, -1, :]).item()
        ablated_next = model.tokenizer.decode([ablated_next_token])
    
    return original_next, ablated_next

# Test sentences
test_sentences = [
    "The cat sat on the mat. The cat",
    "She went to the store to buy milk. She",
    "The algorithm processed the data efficiently. The algorithm",
    "In the beginning was the Word. In the beginning",
    "Once upon a time there was a princess. Once upon a time",
]

print("\n" + "="*60)
print("ABLATION EXPERIMENTS")
print("="*60)

for sentence_idx, text in enumerate(test_sentences):
    print(f"\n{'='*60}")
    print(f"Test {sentence_idx + 1}: '{text[:50]}...'")
    print("-"*60)
    
    # Find top feature pairs for all heads
    top_pairs = find_top_feature_pair_per_head(
        model, sae, text, LAYER,
        exclude_self=True,  # Exclude self-interactions
        verbose=False
    )
    
    if not top_pairs:
        print("No feature pairs found for this text.")
        continue
    
    # Test ablation on head with strongest feature interaction
    if top_pairs:
        strongest_head = max(top_pairs.keys(), 
                           key=lambda h: top_pairs[h]['avg_strength'])
        
        pair_info = top_pairs[strongest_head]
        print(f"\nStrongest feature interaction:")
        print(f"  Head {strongest_head}: F{pair_info['feature_i']} → F{pair_info['feature_j']}")
        print(f"  Average strength: {pair_info['avg_strength']:.4f}")
        print(f"  Occurrences: {pair_info['count']}")
        
        # Reconstruct attention with and without this feature pair
        print(f"\nReconstructing attention patterns...")
        original_pattern, ablated_pattern = reconstruct_attention_with_ablation(
            model, sae, text, LAYER, strongest_head,
            pair_info['feature_i'], pair_info['feature_j']
        )
        
        # Compare outputs
        print(f"\nComparing model outputs:")
        print(f"Input text: '{text}'")
        
        # Get next token predictions
        original_next, ablated_next = run_model_with_ablated_attention(
            model, text, LAYER, strongest_head,
            original_pattern, ablated_pattern
        )
        
        print(f"Original next token: '{original_next}'")
        print(f"Ablated next token:  '{ablated_next}'")
        
        if original_next != ablated_next:
            print("✅ Ablation changed the output!")
        else:
            print("❌ No change in output")
        
        # Calculate attention pattern difference
        pattern_diff = (original_pattern - ablated_pattern).abs().sum().item()
        print(f"Total attention pattern difference: {pattern_diff:.4f}")

print("\n" + "="*60)
print("Experiment complete!")
print("="*60)