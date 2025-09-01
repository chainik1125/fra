#!/usr/bin/env python
"""Correct FRA ablation using attention patterns (post-softmax)."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
from pathlib import Path

torch.set_grad_enabled(False)

print("="*60)
print("Correct FRA Ablation Using Attention Patterns")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def create_attention_patterns(model, sae, text, layer):
    """Create attention patterns from FRA: full and non-self-removed."""
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
        
        # Create attention scores (pre-softmax)
        full_scores = torch.zeros((seq_len, seq_len), device=DEVICE)
        nonself_scores = torch.zeros((seq_len, seq_len), device=DEVICE)
        
        num_self = 0
        num_nonself = 0
        self_strength = 0
        nonself_strength = 0
        
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            feat_i = indices[2, i].item()
            feat_j = indices[3, i].item()
            val = values[i].item()
            
            # Add to full scores
            full_scores[q_pos, k_pos] += val
            
            # Add to non-self-removed scores only if self-interaction
            if feat_i == feat_j:
                nonself_scores[q_pos, k_pos] += val
                num_self += 1
                self_strength += abs(val)
            else:
                num_nonself += 1
                nonself_strength += abs(val)
        
        # Apply causal mask (can't attend to future)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=DEVICE), diagonal=1).bool()
        full_scores.masked_fill_(causal_mask, float('-inf'))
        nonself_scores.masked_fill_(causal_mask, float('-inf'))
        
        # Convert to patterns using softmax
        full_pattern = F.softmax(full_scores, dim=-1)
        nonself_pattern = F.softmax(nonself_scores, dim=-1)
        
        # Handle any NaNs from all -inf rows
        full_pattern = torch.nan_to_num(full_pattern, 0.0)
        nonself_pattern = torch.nan_to_num(nonself_pattern, 0.0)
        
        full_patterns[head] = full_pattern
        nonself_removed_patterns[head] = nonself_pattern
        statistics[head] = {
            'num_self': num_self,
            'num_nonself': num_nonself,
            'self_strength': self_strength,
            'nonself_strength': nonself_strength,
            'seq_len': seq_len
        }
    
    return full_patterns, nonself_removed_patterns, statistics

# Test prompt
prompt = "When John and Mary went to the store, John gave a drink to"

print(f"\nPrompt: '{prompt}'")
print("-"*60)

# Create patterns
print("\nComputing FRA-based attention patterns...")
full_patterns, nonself_patterns, statistics = create_attention_patterns(model, sae, prompt, LAYER)

# Statistics
total_self = sum(s['num_self'] for s in statistics.values() if s)
total_nonself = sum(s['num_nonself'] for s in statistics.values() if s)
total_self_strength = sum(s['self_strength'] for s in statistics.values() if s)
total_nonself_strength = sum(s['nonself_strength'] for s in statistics.values() if s)

print(f"\nFeature statistics:")
print(f"  Self-interactions:     {total_self:,} features, strength {total_self_strength:.1f}")
print(f"  Non-self interactions: {total_nonself:,} features, strength {total_nonself_strength:.1f}")
print(f"  Ratio: {total_nonself_strength / (total_self_strength + 0.001):.1f}x more non-self strength")

# Get tokens
tokens = model.tokenizer.encode(prompt)
if len(tokens) > 128:
    tokens = tokens[:128]
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

print("\nGenerating completions...")

# 1. NORMAL (no intervention)
normal_tokens = []
current = input_ids
for _ in range(15):
    logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    normal_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
normal_text = model.tokenizer.decode(normal_tokens)

# 2. FULL FRA PATTERNS
full_tokens = []
current = input_ids
for _ in range(15):
    def full_hook(pattern, hook):
        for head, fra_pattern in full_patterns.items():
            if fra_pattern is not None:
                seq_len = fra_pattern.shape[0]
                pattern[:, head, :seq_len, :seq_len] = fra_pattern.unsqueeze(0)
        return pattern
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_pattern", full_hook)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    full_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
full_text = model.tokenizer.decode(full_tokens)

# 3. NON-SELF REMOVED (only self features)
nonself_tokens = []
current = input_ids
for _ in range(15):
    def nonself_hook(pattern, hook):
        for head, fra_pattern in nonself_patterns.items():
            if fra_pattern is not None:
                seq_len = fra_pattern.shape[0]
                pattern[:, head, :seq_len, :seq_len] = fra_pattern.unsqueeze(0)
        return pattern
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_pattern", nonself_hook)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    nonself_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
nonself_text = model.tokenizer.decode(nonself_tokens)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n1. NORMAL (no intervention):")
print(f"   '{normal_text}'")

print(f"\n2. FULL FRA (all {total_self + total_nonself:,} features):")
print(f"   '{full_text}'")

print(f"\n3. NON-SELF REMOVED (only {total_self:,} self features):")
print(f"   '{nonself_text}'")

# Count differences
normal_vs_full = sum(1 for n, f in zip(normal_tokens, full_tokens) if n != f)
normal_vs_nonself = sum(1 for n, ns in zip(normal_tokens, nonself_tokens) if n != ns)
full_vs_nonself = sum(1 for f, ns in zip(full_tokens, nonself_tokens) if f != ns)

print("\n" + "-"*60)
print("Token differences:")
print(f"  Normal vs Full FRA:        {normal_vs_full}/15")
print(f"  Normal vs Non-self removed: {normal_vs_nonself}/15")
print(f"  Full vs Non-self removed:   {full_vs_nonself}/15")

if full_vs_nonself > 0:
    print(f"\n✅ Removing {total_nonself:,} non-self features changes {full_vs_nonself} tokens!")
    print(f"   Non-self features ARE important for generation")
elif normal_vs_full > 0:
    print(f"\n⚠️ FRA reconstruction changes output, but non-self removal doesn't")
else:
    print(f"\n❌ No differences detected")

print("="*60)