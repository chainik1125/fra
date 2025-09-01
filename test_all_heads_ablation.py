#!/usr/bin/env python
"""Test ablating all heads vs single head."""

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
print("\n1. NORMAL:")
normal_tokens = []
current = input_ids
for _ in range(10):
    logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    normal_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

print(f"   {model.tokenizer.decode(normal_tokens)}")

# Zero single head
print("\n2. ZERO HEAD 0 ONLY:")
single_tokens = []
current = input_ids

def zero_single_hook(attn_scores, hook):
    attn_scores[:, 0, :, :] = 0
    return attn_scores

for _ in range(10):
    with model.hooks([("blocks.5.attn.hook_attn_scores", zero_single_hook)]):
        logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    single_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

print(f"   {model.tokenizer.decode(single_tokens)}")

# Zero ALL heads
print("\n3. ZERO ALL 12 HEADS:")
all_tokens = []
current = input_ids

def zero_all_hook(attn_scores, hook):
    attn_scores[:, :, :, :] = 0  # Zero ALL heads
    return attn_scores

for _ in range(10):
    with model.hooks([("blocks.5.attn.hook_attn_scores", zero_all_hook)]):
        logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    all_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

print(f"   {model.tokenizer.decode(all_tokens)}")

# Multiply all by 0.5
print("\n4. MULTIPLY ALL HEADS BY 0.5:")
half_tokens = []
current = input_ids

def half_all_hook(attn_scores, hook):
    attn_scores[:, :, :, :] *= 0.5
    return attn_scores

for _ in range(10):
    with model.hooks([("blocks.5.attn.hook_attn_scores", half_all_hook)]):
        logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    half_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

print(f"   {model.tokenizer.decode(half_tokens)}")

# Summary
print("\n" + "="*50)
single_changed = sum(1 for n, s in zip(normal_tokens, single_tokens) if n != s)
all_changed = sum(1 for n, a in zip(normal_tokens, all_tokens) if n != a)
half_changed = sum(1 for n, h in zip(normal_tokens, half_tokens) if n != h)

print(f"Single head zero: {single_changed}/10 changed")
print(f"All heads zero:   {all_changed}/10 changed")
print(f"All heads * 0.5:  {half_changed}/10 changed")