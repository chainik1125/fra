#!/usr/bin/env python
"""Implement actual ablation of top feature pair and measure effect."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import torch.nn.functional as F
from typing import Tuple, Dict

torch.set_grad_enabled(False)

print("="*60)
print("Feature Pair Ablation with Intervention")
print("="*60)

# Configuration
LAYER = 5
HEAD = 0
DEVICE = 'cuda'

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)
print(f"Model and SAE loaded for Layer {LAYER}, Head {HEAD}")

def compute_ablated_attention_pattern(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    text: str,
    layer: int,
    head: int,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Compute original and ablated attention patterns.
    
    Returns:
        (original_pattern, ablated_pattern, info_dict)
    """
    # Tokenize
    tokens = model.tokenizer.encode(text)
    if len(tokens) > 128:
        tokens = tokens[:128]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    seq_len = len(tokens)
    
    # Step 1: Compute FRA for this head
    if verbose:
        print(f"\n1. Computing FRA for head {head}...")
    
    fra_result = get_sentence_fra_batch(
        model, sae, text, 
        layer=layer, head=head,
        max_length=128, top_k=30,
        verbose=False
    )
    
    if fra_result is None or fra_result['total_interactions'] == 0:
        print("No feature interactions found!")
        return None, None, {}
    
    # Step 2: Find max absolute value feature pair
    fra_sparse = fra_result['fra_tensor_sparse']
    indices = fra_sparse.indices()
    values = fra_sparse.values()
    
    # Find max absolute value
    max_idx = values.abs().argmax()
    max_val = values[max_idx].item()
    max_abs = values[max_idx].abs().item()
    
    # Get the feature pair and positions
    query_pos = indices[0, max_idx].item()
    key_pos = indices[1, max_idx].item()
    feature_i = indices[2, max_idx].item()
    feature_j = indices[3, max_idx].item()
    
    if verbose:
        print(f"\n2. Found max feature interaction:")
        print(f"   Features: F{feature_i} → F{feature_j}")
        print(f"   Positions: query={query_pos}, key={key_pos}")
        print(f"   Value: {max_val:.4f} (abs: {max_abs:.4f})")
    
    # Step 3: Create ablated FRA tensor (set max to zero)
    if verbose:
        print(f"\n3. Creating ablated FRA tensor...")
    
    # Create mask for all entries EXCEPT the max one
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[max_idx] = False
    
    # Create new sparse tensor without the max entry
    ablated_indices = indices[:, mask]
    ablated_values = values[mask]
    
    ablated_fra = torch.sparse_coo_tensor(
        ablated_indices, 
        ablated_values,
        fra_sparse.shape,
        device=DEVICE
    )
    
    # Step 4: Sum over feature dimensions to get attention patterns
    if verbose:
        print(f"\n4. Converting to attention patterns...")
    
    # Original pattern: sum over last two dimensions
    original_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
    for i in range(indices.shape[1]):
        q_pos = indices[0, i].item()
        k_pos = indices[1, i].item()
        original_pattern[q_pos, k_pos] += values[i].item()
    
    # Ablated pattern: sum over last two dimensions
    ablated_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
    for i in range(ablated_indices.shape[1]):
        q_pos = ablated_indices[0, i].item()
        k_pos = ablated_indices[1, i].item()
        ablated_pattern[q_pos, k_pos] += ablated_values[i].item()
    
    # Apply causal mask (optional - GPT2 uses causal attention)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1) * -1e10
    original_pattern_masked = original_pattern + causal_mask
    ablated_pattern_masked = ablated_pattern + causal_mask
    
    # Apply softmax to get proper attention weights
    original_attention = F.softmax(original_pattern_masked / np.sqrt(64), dim=-1)
    ablated_attention = F.softmax(ablated_pattern_masked / np.sqrt(64), dim=-1)
    
    info = {
        'max_feature_i': feature_i,
        'max_feature_j': feature_j,
        'max_value': max_val,
        'max_abs': max_abs,
        'max_query_pos': query_pos,
        'max_key_pos': key_pos,
        'total_interactions': fra_result['total_interactions']
    }
    
    return original_attention, ablated_attention, info

def run_with_attention_intervention(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
    attention_pattern: torch.Tensor,
    generate_tokens: int = 10
) -> Dict:
    """
    Run model with a specific attention pattern substituted in.
    """
    tokens = model.tokenizer.encode(text)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Hook function to replace attention pattern
    def attention_pattern_hook(pattern, hook):
        if hook.name == f"blocks.{layer}.attn.hook_pattern":
            # Replace only the specified head's pattern
            pattern[:, head, :len(tokens), :len(tokens)] = attention_pattern.unsqueeze(0)
        return pattern
    
    # Generate with hook
    generated_tokens = []
    all_probs = []
    
    current_input = input_ids
    for _ in range(generate_tokens):
        with model.hooks([(f"blocks.{layer}.attn.hook_pattern", attention_pattern_hook)]):
            logits = model(current_input)
        
        probs = F.softmax(logits[0, -1, :], dim=-1)
        next_token = torch.argmax(probs).item()
        
        generated_tokens.append(next_token)
        all_probs.append(probs.cpu())
        
        # Append for next iteration
        current_input = torch.cat([current_input, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    generated_text = model.tokenizer.decode(generated_tokens)
    
    return {
        'tokens': generated_tokens,
        'text': generated_text,
        'probs': all_probs
    }

# Test cases
test_cases = [
    "The cat sat on the mat. The cat",
    "She went to the store. She bought",
    "Once upon a time, there was a princess. Once upon",
    "The algorithm processed the data. The algorithm",
    "Python is a programming language. Python is",
]

print("\n" + "="*60)
print("ABLATION EXPERIMENTS")
print("="*60)

for idx, prompt in enumerate(test_cases):
    print(f"\n{'='*60}")
    print(f"Test {idx + 1}")
    print(f"Prompt: '{prompt}'")
    print("-"*60)
    
    # Compute original and ablated attention patterns
    original_attn, ablated_attn, info = compute_ablated_attention_pattern(
        model, sae, prompt, LAYER, HEAD, verbose=True
    )
    
    if original_attn is None:
        continue
    
    # Generate with original attention
    print(f"\n5. Generating text with ORIGINAL attention...")
    original_result = run_with_attention_intervention(
        model, prompt, LAYER, HEAD, original_attn, generate_tokens=5
    )
    
    # Generate with ablated attention
    print(f"\n6. Generating text with ABLATED attention...")
    ablated_result = run_with_attention_intervention(
        model, prompt, LAYER, HEAD, ablated_attn, generate_tokens=5
    )
    
    # Compare results
    print(f"\n" + "="*40)
    print("RESULTS:")
    print("-"*40)
    print(f"Original: {prompt} → {original_result['text']}")
    print(f"Ablated:  {prompt} → {ablated_result['text']}")
    
    # Calculate difference
    if original_result['text'] != ablated_result['text']:
        print(f"\n✅ ABLATION CHANGED OUTPUT!")
        
        # Show token-level differences
        for i, (orig_tok, abl_tok) in enumerate(zip(original_result['tokens'], ablated_result['tokens'])):
            if orig_tok != abl_tok:
                orig_word = model.tokenizer.decode([orig_tok])
                abl_word = model.tokenizer.decode([abl_tok])
                print(f"   Position {i}: '{orig_word}' → '{abl_word}'")
    else:
        print(f"\n❌ No change in output")
    
    # Calculate KL divergence for first predicted token
    orig_probs = original_result['probs'][0]
    abl_probs = ablated_result['probs'][0]
    kl_div = F.kl_div(abl_probs.log(), orig_probs, reduction='sum').item()
    print(f"\nKL divergence (first token): {kl_div:.4f}")
    
    # Show attention pattern statistics
    attn_diff = (original_attn - ablated_attn).abs()
    print(f"Attention difference: mean={attn_diff.mean():.6f}, max={attn_diff.max():.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Tested ablation of top feature pairs in Layer {LAYER}, Head {HEAD}")
print("Key findings will vary based on the specific feature interactions found.")
print("="*60)