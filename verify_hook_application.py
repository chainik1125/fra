#!/usr/bin/env python
"""Verify that our attention hooks are actually being applied."""

import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

DEVICE = 'cuda'
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)

prompt = "The cat sat on the"
tokens = model.tokenizer.encode(prompt)
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

print("Testing attention pattern replacement...")
print("="*50)

# Test 1: Normal generation
print("\n1. NORMAL:")
with torch.no_grad():
    logits = model(input_ids)
normal_token = torch.argmax(logits[0, -1, :]).item()
print(f"   Next token: '{model.tokenizer.decode([normal_token])}'")

# Test 2: Zero out ALL attention in layer 5
print("\n2. ZERO ALL ATTENTION (layer 5):")

captured_before = []
captured_after = []

def capture_and_zero(attn_scores, hook):
    # Capture original
    captured_before.append(attn_scores[0, 0, -1, :].clone().cpu())
    
    # Zero out everything
    attn_scores[:, :, :, :] = 0
    
    # Capture after
    captured_after.append(attn_scores[0, 0, -1, :].clone().cpu())
    
    return attn_scores

with model.hooks([("blocks.5.attn.hook_attn_scores", capture_and_zero)]):
    logits = model(input_ids)

zero_token = torch.argmax(logits[0, -1, :]).item()
print(f"   Next token: '{model.tokenizer.decode([zero_token])}'")

print(f"\n   Original attention (head 0, last token): {captured_before[0][:5].tolist()}")
print(f"   After zeroing: {captured_after[0][:5].tolist()}")

if normal_token == zero_token:
    print("\n⚠️ Zeroing attention had NO effect on output!")

# Test 3: Set attention to uniform
print("\n3. UNIFORM ATTENTION (layer 5):")

def uniform_attention(attn_scores, hook):
    batch, heads, seq_len, _ = attn_scores.shape
    # Set to uniform attention
    attn_scores[:, :, :, :] = 1.0 / seq_len
    return attn_scores

with model.hooks([("blocks.5.attn.hook_attn_scores", uniform_attention)]):
    logits = model(input_ids)

uniform_token = torch.argmax(logits[0, -1, :]).item()
print(f"   Next token: '{model.tokenizer.decode([uniform_token])}'")

# Test 4: Try hook_pattern instead of hook_attn_scores
print("\n4. ZERO ATTENTION PATTERN (layer 5):")

def zero_pattern(pattern, hook):
    pattern[:, :, :, :] = 0
    return pattern

with model.hooks([("blocks.5.attn.hook_pattern", zero_pattern)]):
    logits = model(input_ids)

pattern_token = torch.argmax(logits[0, -1, :]).item()
print(f"   Next token: '{model.tokenizer.decode([pattern_token])}'")

# Summary
print("\n" + "="*50)
print("SUMMARY:")
print(f"Normal:          '{model.tokenizer.decode([normal_token])}'")
print(f"Zero scores:     '{model.tokenizer.decode([zero_token])}'")
print(f"Uniform scores:  '{model.tokenizer.decode([uniform_token])}'")
print(f"Zero pattern:    '{model.tokenizer.decode([pattern_token])}'")

if normal_token == zero_token == uniform_token:
    print("\n❌ Modifying attn_scores has no effect!")
    print("We need to use hook_pattern instead!")
elif pattern_token != normal_token:
    print(f"\n✅ hook_pattern works! Changes output from '{model.tokenizer.decode([normal_token])}' to '{model.tokenizer.decode([pattern_token])}'")