#!/usr/bin/env python
"""Correct comparison: Use FRA-reconstructed attention for both normal and ablated cases."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import random
from pathlib import Path

torch.set_grad_enabled(False)

print("="*60)
print("FRA-Based Ablation: Full vs Non-Self Removed")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def create_fra_patterns(model, sae, text, layer):
    """Create both full and non-self-removed FRA patterns for all heads."""
    full_patterns = {}
    nonself_removed_patterns = {}
    statistics = {}
    
    for head in range(N_HEADS):
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=30,
            verbose=False
        )
        
        if fra_result is None or fra_result['total_interactions'] == 0:
            full_patterns[head] = None
            nonself_removed_patterns[head] = None
            statistics[head] = None
            continue
        
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        seq_len = fra_result['seq_len']
        
        # Create FULL pattern by summing over ALL feature dimensions
        full_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            full_pattern[q_pos, k_pos] += values[i].item()
        
        # Create NON-SELF-REMOVED pattern by summing only self-interactions
        nonself_removed_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
        num_self = 0
        num_nonself = 0
        self_strength = 0
        nonself_strength = 0
        
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            feat_i = indices[2, i].item()
            feat_j = indices[3, i].item()
            
            if feat_i == feat_j:  # Self-interaction
                nonself_removed_pattern[q_pos, k_pos] += values[i].item()
                num_self += 1
                self_strength += values[i].abs().item()
            else:  # Non-self interaction
                num_nonself += 1
                nonself_strength += values[i].abs().item()
        
        full_patterns[head] = full_pattern
        nonself_removed_patterns[head] = nonself_removed_pattern
        statistics[head] = {
            'num_self': num_self,
            'num_nonself': num_nonself,
            'self_strength': self_strength,
            'nonself_strength': nonself_strength,
            'seq_len': seq_len
        }
    
    return full_patterns, nonself_removed_patterns, statistics

def create_one_random_removed_patterns(model, sae, text, layer):
    """Create FRA patterns with just ONE random pair removed."""
    patterns = {}
    removed_info = None
    
    # First collect all feature pairs across all heads
    all_pairs = []
    
    for head in range(N_HEADS):
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=30,
            verbose=False
        )
        
        if fra_result is None or fra_result['total_interactions'] == 0:
            patterns[head] = None
            continue
        
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        seq_len = fra_result['seq_len']
        
        # Collect all pairs for potential removal
        for i in range(indices.shape[1]):
            all_pairs.append((head, i, values[i].abs().item()))
        
        # Initialize pattern with all pairs (will remove one later)
        pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            pattern[q_pos, k_pos] += values[i].item()
        
        patterns[head] = {
            'pattern': pattern,
            'fra_sparse': fra_sparse,
            'seq_len': seq_len
        }
    
    # Select ONE random pair to remove
    if all_pairs:
        selected_head, selected_idx, selected_value = random.choice(all_pairs)
        
        # Now remove that one pair from the selected head's pattern
        if patterns[selected_head] is not None:
            fra_sparse = patterns[selected_head]['fra_sparse']
            indices = fra_sparse.indices()
            values = fra_sparse.values()
            seq_len = patterns[selected_head]['seq_len']
            
            # Recreate pattern without the selected pair
            new_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
            for i in range(indices.shape[1]):
                if i != selected_idx:  # Skip the selected pair
                    q_pos = indices[0, i].item()
                    k_pos = indices[1, i].item()
                    new_pattern[q_pos, k_pos] += values[i].item()
            
            patterns[selected_head]['pattern'] = new_pattern
            
            # Store info about what was removed
            removed_info = {
                'head': selected_head,
                'position': (indices[0, selected_idx].item(), indices[1, selected_idx].item()),
                'features': (indices[2, selected_idx].item(), indices[3, selected_idx].item()),
                'value': selected_value
            }
    
    # Extract just the patterns
    final_patterns = {}
    for head in range(N_HEADS):
        if patterns[head] is not None:
            final_patterns[head] = patterns[head]['pattern']
        else:
            final_patterns[head] = None
    
    return final_patterns, removed_info

# Test on a single prompt
prompt = "When John and Mary went to the store, John gave a drink to"

print(f"\nPrompt: '{prompt}'")
print("-"*60)

# Create FRA patterns
print("\nComputing FRA patterns...")
full_patterns, nonself_removed_patterns, statistics = create_fra_patterns(model, sae, prompt, LAYER)
one_random_patterns, random_removed_info = create_one_random_removed_patterns(model, sae, prompt, LAYER)

# Calculate total statistics
total_self = sum(s['num_self'] for s in statistics.values() if s)
total_nonself = sum(s['num_nonself'] for s in statistics.values() if s)
total_self_strength = sum(s['self_strength'] for s in statistics.values() if s)
total_nonself_strength = sum(s['nonself_strength'] for s in statistics.values() if s)

print(f"\nStatistics across all {N_HEADS} heads:")
print(f"  Self-interactions:     {total_self:,} pairs, strength {total_self_strength:.1f}")
print(f"  Non-self interactions: {total_nonself:,} pairs, strength {total_nonself_strength:.1f}")
if random_removed_info:
    print(f"  One random removed:    Head {random_removed_info['head']}, "
          f"F{random_removed_info['features'][0]}→F{random_removed_info['features'][1]}, "
          f"strength {random_removed_info['value']:.4f}")

# Get tokens for generation
tokens = model.tokenizer.encode(prompt)
if len(tokens) > 128:
    tokens = tokens[:128]
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

print("\nGenerating with FRA-reconstructed attention...")

# 1. Generate with FULL FRA (reconstructed normal attention)
full_tokens = []
current = input_ids
for _ in range(15):
    def full_hook(attn_scores, hook):
        for head, pattern in full_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", full_hook)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    full_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

full_text = model.tokenizer.decode(full_tokens)

# 2. Generate with NON-SELF REMOVED
nonself_removed_tokens = []
current = input_ids
for _ in range(15):
    def nonself_hook(attn_scores, hook):
        for head, pattern in nonself_removed_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", nonself_hook)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    nonself_removed_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

nonself_removed_text = model.tokenizer.decode(nonself_removed_tokens)

# 3. Generate with ONE RANDOM REMOVED
one_random_tokens = []
current = input_ids
for _ in range(15):
    def one_random_hook(attn_scores, hook):
        for head, pattern in one_random_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", one_random_hook)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    one_random_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

one_random_text = model.tokenizer.decode(one_random_tokens)

# Display results
print("\n" + "="*60)
print("RESULTS (all using FRA-reconstructed attention)")
print("="*60)

print(f"\n1. FULL FRA (all features):")
print(f"   '{full_text}'")

print(f"\n2. NON-SELF REMOVED ({total_nonself:,} pairs):")
print(f"   '{nonself_removed_text}'")

print(f"\n3. ONE RANDOM REMOVED (1 pair):")
print(f"   '{one_random_text}'")

# Count differences
nonself_diffs = sum(1 for f, n in zip(full_tokens, nonself_removed_tokens) if f != n)
random_diffs = sum(1 for f, r in zip(full_tokens, one_random_tokens) if f != r)

print("\n" + "-"*60)
print("Token differences from FULL FRA:")
print(f"  Non-self removed: {nonself_diffs}/15 tokens changed")
print(f"  One random removed: {random_diffs}/15 tokens changed")

if nonself_diffs > random_diffs:
    effectiveness = nonself_diffs / max(random_diffs, 1)
    print(f"\n✅ Removing {total_nonself:,} non-self pairs is {effectiveness:.1f}x more disruptive than 1 random pair")
elif nonself_diffs == 0 and random_diffs == 0:
    print(f"\n❌ Neither ablation had any effect (FRA reconstruction may be incomplete)")
else:
    print(f"\n⚠️ Results: non-self {nonself_diffs} changes, random {random_diffs} changes")

# Save results
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FRA-Based Ablation Comparison</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .prompt {{ background: #2c3e50; color: white; padding: 20px; margin: 20px 0; font-family: monospace; border-radius: 8px; }}
        .stats {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .results {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 30px 0; }}
        .result {{ padding: 20px; border-radius: 8px; background: white; border: 2px solid #ddd; }}
        .full {{ border-color: #4caf50; }}
        .nonself {{ border-color: #f44336; }}
        .random {{ border-color: #2196f3; }}
        .text {{ font-family: monospace; margin: 15px 0; font-size: 14px; line-height: 1.5; }}
        .metric {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }}
        .highlight {{ background: #ffd54f; padding: 2px 6px; border-radius: 3px; font-weight: bold; }}
        .note {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <h1>🔬 FRA-Based Ablation Comparison</h1>
    
    <div class="note">
        <strong>⚠️ Important:</strong> All three cases use FRA-reconstructed attention patterns.
        This ensures fair comparison but limits generation to the original prompt length.
    </div>
    
    <div class="prompt">
        <strong>Prompt:</strong> {prompt}
    </div>
    
    <div class="stats">
        <h2>Feature Statistics</h2>
        <div class="metric">Self-interactions kept: <strong>{total_self:,}</strong> pairs (strength: {total_self_strength:.1f})</div>
        <div class="metric">Non-self interactions removed: <strong>{total_nonself:,}</strong> pairs (strength: {total_nonself_strength:.1f})</div>
        <div class="metric">One random removed: <strong>1</strong> pair 
            {f"(Head {random_removed_info['head']}, F{random_removed_info['features'][0]}→F{random_removed_info['features'][1]})" if random_removed_info else ""}
        </div>
    </div>
    
    <div class="results">
        <div class="result full">
            <h3>✓ Full FRA</h3>
            <div style="font-size: 12px; color: #666;">All {total_self + total_nonself:,} feature pairs</div>
            <div class="text">{full_text}</div>
        </div>
        
        <div class="result nonself">
            <h3>🔥 Non-Self Removed</h3>
            <div style="font-size: 12px; color: #666;">{total_nonself:,} pairs removed</div>
            <div class="text">{nonself_removed_text}</div>
            <div><span class="highlight">{nonself_diffs}/15 tokens changed</span></div>
        </div>
        
        <div class="result random">
            <h3>🎲 One Random Removed</h3>
            <div style="font-size: 12px; color: #666;">1 pair removed</div>
            <div class="text">{one_random_text}</div>
            <div><span class="highlight">{random_diffs}/15 tokens changed</span></div>
        </div>
    </div>
    
    <div class="stats">
        <h2>Effectiveness</h2>
        <div class="metric" style="text-align: center; font-size: 18px;">
            Removing <strong>{total_nonself:,}</strong> non-self pairs causes 
            <span class="highlight" style="font-size: 24px;">{nonself_diffs}</span> token changes<br>
            Removing <strong>1</strong> random pair causes 
            <span class="highlight" style="font-size: 24px;">{random_diffs}</span> token changes
        </div>
    </div>
</body>
</html>
"""

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
html_path = output_dir / "fra_based_ablation.html"
with open(html_path, 'w') as f:
    f.write(html)

print(f"\n✅ Results saved to: {html_path}")
print("="*60)