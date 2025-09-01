#!/usr/bin/env python
"""Debug exactly what gets ablated in our patterns."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import random

torch.set_grad_enabled(False)

LAYER = 5
DEVICE = 'cuda'

print("Loading...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

prompt = "The capital of France is"

# Get FRA for head 0
fra_result = get_sentence_fra_batch(
    model, sae, prompt, 
    layer=LAYER, head=0,
    max_length=128, top_k=20,
    verbose=False
)

if fra_result:
    fra_sparse = fra_result['fra_tensor_sparse']
    indices = fra_sparse.indices()
    values = fra_sparse.values()
    seq_len = fra_result['seq_len']
    
    print(f"FRA has {len(values)} non-zero entries")
    print(f"Sum of all values: {values.sum():.4f}")
    
    # Filter non-self
    q_feats = indices[2, :]
    k_feats = indices[3, :]
    mask = q_feats != k_feats
    non_self_indices = torch.where(mask)[0]
    
    print(f"Non-self entries: {len(non_self_indices)}")
    
    # Find max
    non_self_values = values[mask].abs()
    max_idx_in_filtered = non_self_values.argmax()
    max_idx = non_self_indices[max_idx_in_filtered].item()
    
    print(f"\nMax non-self:")
    print(f"  Index in sparse: {max_idx}")
    print(f"  Value: {values[max_idx]:.4f}")
    print(f"  Position: ({indices[0, max_idx]}, {indices[1, max_idx]})")
    print(f"  Features: F{indices[2, max_idx]}→F{indices[3, max_idx]}")
    
    # Create targeted ablation pattern
    targeted_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
    for i in range(indices.shape[1]):
        if i != max_idx:  # Skip the max index
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            targeted_pattern[q_pos, k_pos] += values[i].item()
    
    print(f"\nTargeted pattern sum: {targeted_pattern.sum():.4f}")
    print(f"Removed: {values.sum() - targeted_pattern.sum():.4f}")
    
    # Pick a random non-self index
    if len(non_self_indices) > 1:
        random_choices = [idx.item() for idx in non_self_indices if idx.item() != max_idx]
        random_idx = random.choice(random_choices) if random_choices else max_idx
    else:
        random_idx = max_idx
    
    print(f"\nRandom non-self:")
    print(f"  Index in sparse: {random_idx}")
    print(f"  Value: {values[random_idx]:.4f}")
    print(f"  Position: ({indices[0, random_idx]}, {indices[1, random_idx]})")
    print(f"  Features: F{indices[2, random_idx]}→F{indices[3, random_idx]}")
    
    # Create random ablation pattern
    random_pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
    for i in range(indices.shape[1]):
        if i != random_idx:  # Skip the random index
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            random_pattern[q_pos, k_pos] += values[i].item()
    
    print(f"\nRandom pattern sum: {random_pattern.sum():.4f}")
    print(f"Removed: {values.sum() - random_pattern.sum():.4f}")
    
    # Check if patterns are different
    diff = (targeted_pattern - random_pattern).abs().sum()
    print(f"\nDifference between patterns: {diff:.4f}")
    
    if diff < 0.0001:
        print("⚠️ PATTERNS ARE IDENTICAL!")
    else:
        print("✓ Patterns are different")
    
    # Test actual generation
    tokens = model.tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Normal
    current = input_ids
    normal_tokens = []
    for _ in range(5):
        logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        normal_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    print(f"\nNormal: {model.tokenizer.decode(normal_tokens)}")
    
    # With targeted pattern (only head 0)
    current = input_ids
    targeted_tokens = []
    
    def targeted_hook(attn_scores, hook):
        attn_scores[:, 0, :seq_len, :seq_len] = targeted_pattern.unsqueeze(0)
        return attn_scores
    
    for _ in range(5):
        with model.hooks([("blocks.5.attn.hook_attn_scores", targeted_hook)]):
            logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        targeted_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    print(f"Targeted H0: {model.tokenizer.decode(targeted_tokens)}")
    
    # With random pattern (only head 0)
    current = input_ids
    random_tokens = []
    
    def random_hook(attn_scores, hook):
        attn_scores[:, 0, :seq_len, :seq_len] = random_pattern.unsqueeze(0)
        return attn_scores
    
    for _ in range(5):
        with model.hooks([("blocks.5.attn.hook_attn_scores", random_hook)]):
            logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        random_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    print(f"Random H0: {model.tokenizer.decode(random_tokens)}")
    
    if targeted_tokens == random_tokens:
        print("\n⚠️ Same output despite different patterns!")