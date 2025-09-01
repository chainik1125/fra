#!/usr/bin/env python
"""Minimal test to understand ablation behavior."""

import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

DEVICE = 'cuda'
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)

prompt = "The capital of France is"
tokens = model.tokenizer.encode(prompt)
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

print(f"Prompt: {prompt}")
print("="*50)

# Normal generation
print("\n1. NORMAL (no intervention):")
normal_tokens = []
current = input_ids
for i in range(5):
    with torch.no_grad():
        logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    normal_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    print(f"   Step {i+1}: {model.tokenizer.decode([next_token])}")

# Zero out head 0 of layer 5
print("\n2. ZERO HEAD 0 of LAYER 5:")
zero_tokens = []
current = input_ids

def zero_head_hook(attn_scores, hook):
    attn_scores[:, 0, :, :] = 0  # Zero out head 0
    return attn_scores

for i in range(5):
    with model.hooks([("blocks.5.attn.hook_attn_scores", zero_head_hook)]):
        logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    zero_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    print(f"   Step {i+1}: {model.tokenizer.decode([next_token])}")

# Add noise to head 0 of layer 5
print("\n3. ADD NOISE TO HEAD 0 of LAYER 5:")
noise_tokens = []
current = input_ids

def noise_head_hook(attn_scores, hook):
    seq_len = attn_scores.shape[-1]
    noise = torch.randn(seq_len, seq_len).to(DEVICE) * 0.1
    attn_scores[:, 0, :, :] += noise
    return attn_scores

for i in range(5):
    with model.hooks([("blocks.5.attn.hook_attn_scores", noise_head_hook)]):
        logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    noise_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    print(f"   Step {i+1}: {model.tokenizer.decode([next_token])}")

# Summary
print("\n" + "="*50)
print("SUMMARY:")
print(f"Normal: {model.tokenizer.decode(normal_tokens)}")
print(f"Zero H0: {model.tokenizer.decode(zero_tokens)}")
print(f"Noise H0: {model.tokenizer.decode(noise_tokens)}")

if normal_tokens == zero_tokens:
    print("\n⚠️ Zeroing head 0 had NO effect!")
if normal_tokens == noise_tokens:
    print("\n⚠️ Adding noise to head 0 had NO effect!")