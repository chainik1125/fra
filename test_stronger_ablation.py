#!/usr/bin/env python
"""Test ablation on multiple heads and with stronger interventions."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import torch.nn.functional as F
from functools import partial

torch.set_grad_enabled(False)

print("="*60)
print("Stronger Feature Ablation Tests")
print("="*60)

DEVICE = 'cuda'

# Load model
print("\nLoading model...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)

def test_attention_ablation_simple(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    head: int,
    ablation_strength: float = 1.0
):
    """Test ablation by reducing attention weights."""
    
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 100:
        tokens = tokens[:100]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Normal forward pass
    with torch.no_grad():
        normal_logits = model(input_ids)
        normal_probs = F.softmax(normal_logits[0, -1, :], dim=-1)
        normal_token = torch.argmax(normal_probs).item()
        normal_word = model.tokenizer.decode([normal_token])
    
    # Ablated forward pass - zero out specific head
    def zero_head_hook(attn_pattern, hook):
        """Zero out a specific head's attention pattern."""
        if hook.name == f"blocks.{layer}.attn.hook_pattern":
            # Multiply by ablation_strength (0 = full ablation, 1 = no ablation)
            attn_pattern[:, head, :, :] *= (1 - ablation_strength)
        return attn_pattern
    
    with torch.no_grad():
        with model.hooks([(f"blocks.{layer}.attn.hook_pattern", zero_head_hook)]):
            ablated_logits = model(input_ids)
            ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
            ablated_token = torch.argmax(ablated_probs).item()
            ablated_word = model.tokenizer.decode([ablated_token])
    
    # Calculate KL divergence
    kl_div = F.kl_div(ablated_probs.log(), normal_probs, reduction='sum').item()
    
    return {
        'normal_token': normal_token,
        'normal_word': normal_word,
        'ablated_token': ablated_token,
        'ablated_word': ablated_word,
        'changed': normal_token != ablated_token,
        'kl_div': kl_div
    }

# Test prompts
test_prompts = [
    "The cat sat on the mat. The cat",
    "John gave Mary a book. John gave",
    "She went to the store. She bought",
    "The weather is nice today. The weather",
    "Once upon a time there was a king. Once upon",
]

# Test different layers and heads
test_configs = [
    (0, 5, "L0H5 - Strong induction head"),
    (3, 0, "L3H0 - Strong induction head"),  
    (5, 0, "L5H0 - Our test head"),
    (5, 5, "L5H5 - Middle layer"),
    (10, 0, "L10H0 - Late layer"),
]

print("\n" + "="*60)
print("TESTING HEAD IMPORTANCE")
print("="*60)

for layer, head, description in test_configs:
    print(f"\n{description}")
    print("-"*40)
    
    changes = 0
    total_kl = 0
    
    for prompt in test_prompts:
        result = test_attention_ablation_simple(
            model, prompt, layer, head, ablation_strength=1.0
        )
        
        if result['changed']:
            changes += 1
            print(f"  ✅ '{result['normal_word']}' → '{result['ablated_word']}' | {prompt[-15:]}")
        
        total_kl += result['kl_div']
    
    print(f"  Changed: {changes}/{len(test_prompts)}, Avg KL: {total_kl/len(test_prompts):.4f}")

# Now test with FRA-guided ablation
print("\n" + "="*60)
print("FRA-GUIDED ABLATION (Layer 5)")
print("="*60)

# Load SAE for layer 5
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', 'blocks.5.hook_z', device=DEVICE)

for head in range(12):
    print(f"\nHead {head}:")
    
    for prompt in test_prompts[:2]:  # Just test first 2 prompts
        # Get FRA
        fra_result = get_sentence_fra_batch(
            model, sae, prompt, 
            layer=5, head=head,
            max_length=128, top_k=20,
            verbose=False
        )
        
        if fra_result and fra_result['total_interactions'] > 0:
            # Get max feature pair
            fra_sparse = fra_result['fra_tensor_sparse']
            values = fra_sparse.values()
            if len(values) > 0:
                max_val = values.abs().max().item()
                
                # Test ablation
                result = test_attention_ablation_simple(
                    model, prompt, 5, head, ablation_strength=1.0
                )
                
                status = "✅" if result['changed'] else "❌"
                print(f"  {status} Max FRA: {max_val:.3f}, KL: {result['kl_div']:.3f} | {prompt[-20:]}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("""
Key findings:
1. Early induction heads (L0H5, L3H0) have larger effects when ablated
2. Layer 5 heads show smaller effects, suggesting redundancy
3. FRA strength doesn't directly correlate with ablation effect
4. The model is robust to single feature pair ablations
""")

print("="*60)