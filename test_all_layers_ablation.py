#!/usr/bin/env python
"""Test ablating attention in all layers vs just layer 5."""

import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

DEVICE = 'cuda'
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)

prompt = "The cat sat on the"
tokens = model.tokenizer.encode(prompt)
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

print("Testing multi-layer attention ablation...")
print("="*50)

# Normal
print("\n1. NORMAL:")
logits = model(input_ids)
normal_token = torch.argmax(logits[0, -1, :]).item()
print(f"   '{model.tokenizer.decode([normal_token])}'")

# Zero layer 5 only
print("\n2. ZERO LAYER 5:")
def zero_layer5(attn_scores, hook):
    if "blocks.5" in hook.name:
        attn_scores[:, :, :, :] = 0
    return attn_scores

with model.hooks([("blocks.5.attn.hook_attn_scores", zero_layer5)]):
    logits = model(input_ids)
layer5_token = torch.argmax(logits[0, -1, :]).item()
print(f"   '{model.tokenizer.decode([layer5_token])}'")

# Zero ALL layers
print("\n3. ZERO ALL LAYERS:")
hooks = []
for layer in range(12):
    def make_zero_hook(l):
        def zero_hook(attn_scores, hook):
            attn_scores[:, :, :, :] = 0
            return attn_scores
        return zero_hook
    
    hooks.append((f"blocks.{layer}.attn.hook_attn_scores", make_zero_hook(layer)))

with model.hooks(hooks):
    logits = model(input_ids)
all_layers_token = torch.argmax(logits[0, -1, :]).item()
print(f"   '{model.tokenizer.decode([all_layers_token])}'")

# Try pattern hooks on all layers
print("\n4. ZERO ALL PATTERNS:")
pattern_hooks = []
for layer in range(12):
    def make_pattern_hook(l):
        def pattern_hook(pattern, hook):
            pattern[:, :, :, :] = 0
            return pattern
        return pattern_hook
    
    pattern_hooks.append((f"blocks.{layer}.attn.hook_pattern", make_pattern_hook(layer)))

with model.hooks(pattern_hooks):
    logits = model(input_ids)
all_patterns_token = torch.argmax(logits[0, -1, :]).item()
print(f"   '{model.tokenizer.decode([all_patterns_token])}'")

print("\n" + "="*50)
print("SUMMARY:")
if normal_token == layer5_token:
    print("Layer 5 ablation: NO EFFECT")
else:
    print(f"Layer 5 ablation: '{model.tokenizer.decode([normal_token])}' → '{model.tokenizer.decode([layer5_token])}'")

if normal_token == all_layers_token:
    print("All layers scores ablation: NO EFFECT")
else:
    print(f"All layers scores ablation: '{model.tokenizer.decode([normal_token])}' → '{model.tokenizer.decode([all_layers_token])}'")

if normal_token == all_patterns_token:
    print("All patterns ablation: NO EFFECT") 
else:
    print(f"All patterns ablation: '{model.tokenizer.decode([normal_token])}' → '{model.tokenizer.decode([all_patterns_token])}'")

print("\nConclusion:")
if normal_token != all_patterns_token:
    print("✅ We need to use hook_pattern, not hook_attn_scores!")
elif normal_token != all_layers_token:
    print("✅ We need to ablate ALL layers' attention scores!")
else:
    print("❌ Something is wrong - even zeroing everything has no effect")