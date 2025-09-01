#!/usr/bin/env python
"""Check if attention patterns change during generation."""

import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

DEVICE = 'cuda'
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)

prompt = "When John and Mary went to the store, John gave a drink to"
tokens = model.tokenizer.encode(prompt)
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

print(f"Initial prompt length: {len(tokens)} tokens")
print("="*50)

# Hook to capture attention patterns
captured_patterns = []

def capture_hook(attn_scores, hook):
    # Capture head 0's attention pattern
    captured_patterns.append(attn_scores[0, 0, -1, :].clone().cpu())  # Last token's attention
    return attn_scores

# Generate with capturing
current = input_ids
for step in range(3):
    captured_patterns.clear()
    
    with model.hooks([("blocks.5.attn.hook_attn_scores", capture_hook)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    print(f"\nStep {step + 1}: Generated '{model.tokenizer.decode([next_token])}'")
    print(f"  Attention pattern shape: {captured_patterns[0].shape}")
    print(f"  Attention to last 5 positions: {captured_patterns[0][-5:].tolist()}")
    
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

print("\n" + "="*50)
print("Key insight: The attention pattern GROWS with each new token!")
print("When we pre-compute FRA and use it as static replacement,")
print("we're using attention patterns from the ORIGINAL prompt only,")
print("not adapting to newly generated tokens.")