#!/usr/bin/env python
"""Fixed comparison of targeted vs random feature ablation."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import json
from pathlib import Path
import random

torch.set_grad_enabled(False)

print("="*60)
print("Fixed Ablation Comparison: Targeted vs Random")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def find_max_and_random_pairs(model, sae, text, layer):
    """Find both max and random non-self feature pairs for all heads."""
    max_pairs = {}
    random_pairs = {}
    
    for head in range(N_HEADS):
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=20,
            verbose=False
        )
        
        if fra_result is None or fra_result['total_interactions'] == 0:
            max_pairs[head] = None
            random_pairs[head] = None
            continue
        
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        
        if len(values) == 0:
            max_pairs[head] = None
            random_pairs[head] = None
            continue
        
        # Filter self-interactions
        q_feats = indices[2, :]
        k_feats = indices[3, :]
        mask = q_feats != k_feats
        
        if not mask.any():
            max_pairs[head] = None
            random_pairs[head] = None
            continue
        
        # Get max non-self
        non_self_indices = torch.where(mask)[0]
        non_self_values = values[mask].abs()
        max_idx_in_filtered = non_self_values.argmax()
        max_idx = non_self_indices[max_idx_in_filtered].item()
        
        max_pairs[head] = {
            'idx_in_sparse': max_idx,  # Store the actual index
            'value': values[max_idx].item(),
            'abs_value': values[max_idx].abs().item(),
            'query_pos': indices[0, max_idx].item(),
            'key_pos': indices[1, max_idx].item(),
            'feature_i': indices[2, max_idx].item(),
            'feature_j': indices[3, max_idx].item(),
            'fra_sparse': fra_sparse,
            'seq_len': fra_result['seq_len']
        }
        
        # Get a RANDOM non-self pair from the existing pairs
        if len(non_self_indices) > 1:
            # Pick a random index that's not the max
            random_choices = [idx.item() for idx in non_self_indices if idx.item() != max_idx]
            if random_choices:
                random_idx = random.choice(random_choices)
            else:
                random_idx = max_idx  # Fallback if only one non-self pair
        else:
            random_idx = max_idx  # Only one non-self pair available
        
        random_pairs[head] = {
            'idx_in_sparse': random_idx,  # Store the actual index
            'value': values[random_idx].item(),
            'abs_value': values[random_idx].abs().item(),
            'query_pos': indices[0, random_idx].item(),
            'key_pos': indices[1, random_idx].item(),
            'feature_i': indices[2, random_idx].item(),
            'feature_j': indices[3, random_idx].item(),
            'fra_sparse': fra_sparse,
            'seq_len': fra_result['seq_len']
        }
    
    return max_pairs, random_pairs

def create_ablated_patterns(pairs_to_ablate):
    """Create ablated attention patterns by removing specified pairs."""
    ablated_patterns = {}
    
    for head, pair_info in pairs_to_ablate.items():
        if pair_info is None:
            ablated_patterns[head] = None
            continue
        
        fra_sparse = pair_info['fra_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        seq_len = pair_info['seq_len']
        
        ablated = torch.zeros((seq_len, seq_len), device=DEVICE)
        
        # Add all interactions except the one to ablate
        for i in range(indices.shape[1]):
            # Skip the specific index we want to ablate
            if i == pair_info['idx_in_sparse']:
                continue
                
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            ablated[q_pos, k_pos] += values[i].item()
        
        ablated_patterns[head] = ablated
    
    return ablated_patterns

def analyze_prompt_with_benchmark(model, sae, prompt, layer):
    """Analyze prompt with both targeted and random ablation."""
    
    # Find max and random feature pairs
    max_pairs, random_pairs = find_max_and_random_pairs(model, sae, prompt, layer)
    
    # Get normal completion
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    normal_tokens = []
    current = input_ids
    for _ in range(10):
        with torch.no_grad():
            logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        normal_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Get targeted ablation completion
    targeted_patterns = create_ablated_patterns(max_pairs)
    
    targeted_tokens = []
    current = input_ids
    for _ in range(10):
        def hook_fn(attn_scores, hook):
            for head, pattern in targeted_patterns.items():
                if pattern is not None:
                    seq_len = pattern.shape[0]
                    attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
            return attn_scores
        
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(current)
        
        next_token = torch.argmax(logits[0, -1, :]).item()
        targeted_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Get random ablation completion
    random_patterns = create_ablated_patterns(random_pairs)
    
    random_ablation_tokens = []
    current = input_ids
    for _ in range(10):
        def hook_fn(attn_scores, hook):
            for head, pattern in random_patterns.items():
                if pattern is not None:
                    seq_len = pattern.shape[0]
                    attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
            return attn_scores
        
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(current)
        
        next_token = torch.argmax(logits[0, -1, :]).item()
        random_ablation_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Decode
    normal_text = model.tokenizer.decode(normal_tokens)
    targeted_text = model.tokenizer.decode(targeted_tokens)
    random_text = model.tokenizer.decode(random_ablation_tokens)
    
    # Count differences
    targeted_diffs = sum(1 for n, t in zip(normal_tokens, targeted_tokens) if n != t)
    random_diffs = sum(1 for n, r in zip(normal_tokens, random_ablation_tokens) if n != r)
    
    # Calculate total strength
    max_strength = sum(p['abs_value'] for p in max_pairs.values() if p is not None)
    random_strength = sum(p['abs_value'] for p in random_pairs.values() if p is not None)
    
    return {
        'prompt': prompt,
        'normal': normal_text,
        'targeted': targeted_text,
        'random': random_text,
        'targeted_changed': targeted_diffs > 0,
        'random_changed': random_diffs > 0,
        'targeted_diffs': targeted_diffs,
        'random_diffs': random_diffs,
        'max_strength': max_strength,
        'random_strength': random_strength,
        'max_pairs': max_pairs,
        'random_pairs': random_pairs
    }

# Test prompts
test_prompts = [
    "When John and Mary went to the store, John gave a drink to",
    "The capital of France is",
    "The Earth orbits around the",
    "Once upon a time, there lived a wise old wizard who",
]

print("\nAnalyzing prompts with fixed benchmark...")
results = []

for i, prompt in enumerate(test_prompts):
    print(f"\n{i+1}. {prompt[:50]}...")
    result = analyze_prompt_with_benchmark(model, sae, prompt, LAYER)
    results.append(result)
    
    print(f"   Normal:   → {result['normal'][:30]}...")
    print(f"   Targeted: → {result['targeted'][:30]}... ({result['targeted_diffs']}/10 changed)")
    print(f"   Random:   → {result['random'][:30]}... ({result['random_diffs']}/10 changed)")
    
    # Show which features were ablated
    if result['max_pairs'][0]:
        print(f"   Max pair H0: F{result['max_pairs'][0]['feature_i']}→F{result['max_pairs'][0]['feature_j']} "
              f"(str: {result['max_pairs'][0]['abs_value']:.3f})")
    if result['random_pairs'][0]:
        print(f"   Rnd pair H0: F{result['random_pairs'][0]['feature_i']}→F{result['random_pairs'][0]['feature_j']} "
              f"(str: {result['random_pairs'][0]['abs_value']:.3f})")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

targeted_changed = sum(1 for r in results if r['targeted_changed'])
random_changed = sum(1 for r in results if r['random_changed'])

print(f"\nTargeted ablation: {targeted_changed}/{len(results)} changed")
print(f"Random ablation:   {random_changed}/{len(results)} changed")
print(f"\nAvg token differences:")
print(f"  Targeted: {np.mean([r['targeted_diffs'] for r in results]):.1f}/10")
print(f"  Random:   {np.mean([r['random_diffs'] for r in results]):.1f}/10")
print(f"\nAvg ablation strength:")
print(f"  Max pairs:    {np.mean([r['max_strength'] for r in results]):.2f}")
print(f"  Random pairs: {np.mean([r['random_strength'] for r in results]):.2f}")

print("="*60)