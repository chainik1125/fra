#!/usr/bin/env python
"""Test removing ALL features vs just non-self vs none."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
from pathlib import Path

torch.set_grad_enabled(False)

print("="*60)
print("Testing: Full FRA vs Non-Self Removed vs ALL Removed")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def create_all_patterns(model, sae, text, layer):
    """Create full, non-self-removed, and completely empty patterns."""
    full_patterns = {}
    nonself_removed_patterns = {}
    empty_patterns = {}
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
            empty_patterns[head] = None
            statistics[head] = None
            continue
        
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        seq_len = fra_result['seq_len']
        
        # Create FULL pattern
        full_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            full_pattern[q_pos, k_pos] += values[i].item()
        
        # Create NON-SELF-REMOVED pattern (only self-interactions)
        nonself_removed_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
        num_self = 0
        num_nonself = 0
        
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            feat_i = indices[2, i].item()
            feat_j = indices[3, i].item()
            
            if feat_i == feat_j:  # Keep only self-interactions
                nonself_removed_pattern[q_pos, k_pos] += values[i].item()
                num_self += 1
            else:
                num_nonself += 1
        
        # Create EMPTY pattern (all removed)
        empty_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
        
        full_patterns[head] = full_pattern
        nonself_removed_patterns[head] = nonself_removed_pattern
        empty_patterns[head] = empty_pattern
        statistics[head] = {
            'num_self': num_self,
            'num_nonself': num_nonself,
            'seq_len': seq_len,
            'full_sum': full_pattern.sum().item(),
            'nonself_removed_sum': nonself_removed_pattern.sum().item(),
            'empty_sum': empty_pattern.sum().item()
        }
    
    return full_patterns, nonself_removed_patterns, empty_patterns, statistics

# Test prompt
prompt = "When John and Mary went to the store, John gave a drink to"

print(f"\nPrompt: '{prompt}'")
print("-"*60)

# Create patterns
print("\nComputing FRA patterns...")
full_patterns, nonself_removed_patterns, empty_patterns, statistics = create_all_patterns(
    model, sae, prompt, LAYER
)

# Print statistics
total_self = sum(s['num_self'] for s in statistics.values() if s)
total_nonself = sum(s['num_nonself'] for s in statistics.values() if s)

print(f"\nStatistics:")
print(f"  Total self-interactions:   {total_self:,}")
print(f"  Total non-self interactions: {total_nonself:,}")
print(f"  Total features: {total_self + total_nonself:,}")

# Check pattern sums for verification
print(f"\nPattern sum verification (Head 0):")
if statistics[0]:
    print(f"  Full pattern sum: {statistics[0]['full_sum']:.4f}")
    print(f"  Non-self removed sum: {statistics[0]['nonself_removed_sum']:.4f}")
    print(f"  Empty pattern sum: {statistics[0]['empty_sum']:.4f}")

# Get tokens
tokens = model.tokenizer.encode(prompt)
if len(tokens) > 128:
    tokens = tokens[:128]
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

print("\nGenerating completions...")

# 1. FULL FRA
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

# 2. NON-SELF REMOVED (only self features)
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

# 3. ALL REMOVED (empty patterns)
all_removed_tokens = []
current = input_ids
for _ in range(15):
    def empty_hook(attn_scores, hook):
        for head, pattern in empty_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", empty_hook)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    all_removed_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

all_removed_text = model.tokenizer.decode(all_removed_tokens)

# 4. NORMAL (no intervention for comparison)
normal_tokens = []
current = input_ids
for _ in range(15):
    with torch.no_grad():
        logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    normal_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

normal_text = model.tokenizer.decode(normal_tokens)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n1. NORMAL (no intervention):")
print(f"   '{normal_text}'")

print(f"\n2. FULL FRA ({total_self + total_nonself:,} features):")
print(f"   '{full_text}'")

print(f"\n3. NON-SELF REMOVED (only {total_self:,} self features):")
print(f"   '{nonself_removed_text}'")

print(f"\n4. ALL REMOVED (0 features, empty attention):")
print(f"   '{all_removed_text}'")

# Count differences
print("\n" + "-"*60)
print("Comparisons:")

# Compare to normal
normal_vs_full = sum(1 for n, f in zip(normal_tokens, full_tokens) if n != f)
normal_vs_nonself = sum(1 for n, ns in zip(normal_tokens, nonself_removed_tokens) if n != ns)
normal_vs_empty = sum(1 for n, e in zip(normal_tokens, all_removed_tokens) if n != e)

print(f"\nVs. Normal generation:")
print(f"  Full FRA:        {normal_vs_full}/15 tokens different")
print(f"  Non-self removed: {normal_vs_nonself}/15 tokens different")
print(f"  All removed:      {normal_vs_empty}/15 tokens different")

# Compare between FRA versions
full_vs_nonself = sum(1 for f, ns in zip(full_tokens, nonself_removed_tokens) if f != ns)
full_vs_empty = sum(1 for f, e in zip(full_tokens, all_removed_tokens) if f != e)
nonself_vs_empty = sum(1 for ns, e in zip(nonself_removed_tokens, all_removed_tokens) if ns != e)

print(f"\nBetween FRA versions:")
print(f"  Full vs Non-self removed: {full_vs_nonself}/15 tokens different")
print(f"  Full vs All removed:      {full_vs_empty}/15 tokens different")
print(f"  Non-self vs All removed:  {nonself_vs_empty}/15 tokens different")

if full_vs_empty == 0:
    print("\n⚠️ WARNING: Even with ALL features removed (empty attention), output unchanged!")
    print("This suggests the FRA patterns are not being used correctly.")
elif full_vs_nonself == 0 and full_vs_empty > 0:
    print("\n✅ Non-self features appear redundant (no change when removed)")
    print(f"But removing ALL features changes {full_vs_empty} tokens")
elif full_vs_nonself > 0:
    print(f"\n✅ Non-self features matter: removing them changes {full_vs_nonself} tokens")

print("="*60)