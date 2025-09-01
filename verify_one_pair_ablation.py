#!/usr/bin/env python
"""Verify that we're actually only ablating one pair."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch

torch.set_grad_enabled(False)

LAYER = 5
DEVICE = 'cuda'

print("Loading...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

prompt = "When John and Mary went to the store, John gave a drink to"

# Get FRA for head 11 (where the random pair was)
fra_result = get_sentence_fra_batch(
    model, sae, prompt, 
    layer=LAYER, head=11,
    max_length=128, top_k=30,
    verbose=False
)

if fra_result:
    fra_sparse = fra_result['fra_tensor_sparse']
    indices = fra_sparse.indices()
    values = fra_sparse.values()
    seq_len = fra_result['seq_len']
    
    print(f"Head 11 FRA: {len(values)} non-zero entries")
    print(f"Original sum: {values.sum():.4f}")
    
    # Pick one random index
    import random
    random_idx = random.randint(0, len(values)-1)
    
    print(f"\nRemoving index {random_idx}:")
    print(f"  Position: ({indices[0, random_idx]}, {indices[1, random_idx]})")
    print(f"  Features: F{indices[2, random_idx]}→F{indices[3, random_idx]}")
    print(f"  Value: {values[random_idx]:.6f}")
    
    # Create pattern with ONE pair removed
    pattern_one_removed = torch.zeros((seq_len, seq_len), device=DEVICE)
    for i in range(indices.shape[1]):
        if i != random_idx:  # Skip only the one index
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            pattern_one_removed[q_pos, k_pos] += values[i].item()
    
    print(f"\nPattern with one removed: sum = {pattern_one_removed.sum():.4f}")
    print(f"Difference: {values.sum() - pattern_one_removed.sum():.6f}")
    
    # Also create pattern with ALL removed (for comparison)
    pattern_all_removed = torch.zeros((seq_len, seq_len), device=DEVICE)
    print(f"Pattern with all removed: sum = {pattern_all_removed.sum():.4f}")
    
    # Test generation
    tokens = model.tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    print("\n" + "="*50)
    print("Testing generation:")
    
    # Normal
    print("\n1. NORMAL:")
    current = input_ids
    for i in range(5):
        logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        print(f"   Token {i+1}: {model.tokenizer.decode([next_token])}")
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # With one pair removed from head 11
    print("\n2. ONE PAIR REMOVED (Head 11):")
    current = input_ids
    
    def one_removed_hook(attn_scores, hook):
        attn_scores[:, 11, :seq_len, :seq_len] = pattern_one_removed.unsqueeze(0)
        return attn_scores
    
    for i in range(5):
        with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", one_removed_hook)]):
            logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        print(f"   Token {i+1}: {model.tokenizer.decode([next_token])}")
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # With ALL removed from head 11
    print("\n3. ALL REMOVED (Head 11 zeroed):")
    current = input_ids
    
    def all_removed_hook(attn_scores, hook):
        attn_scores[:, 11, :, :] = 0
        return attn_scores
    
    for i in range(5):
        with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", all_removed_hook)]):
            logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        print(f"   Token {i+1}: {model.tokenizer.decode([next_token])}")
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # No intervention on any head
    print("\n4. NO INTERVENTION AT ALL:")
    current = input_ids
    for i in range(5):
        logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        print(f"   Token {i+1}: {model.tokenizer.decode([next_token])}")
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)