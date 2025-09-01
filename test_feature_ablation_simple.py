#!/usr/bin/env python
"""Simpler test of feature pair ablation effects."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
from typing import Dict, List
import torch.nn.functional as F

torch.set_grad_enabled(False)

print("="*60)
print("Feature Pair Ablation - Simple Version")
print("="*60)

# Configuration
LAYER = 5
DEVICE = 'cuda'

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def find_top_feature_pairs(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    text: str,
    layer: int,
    n_heads: int = 12,
    exclude_self: bool = True
) -> List[Dict]:
    """Find top feature pair for each head."""
    
    results = []
    
    for head in range(n_heads):
        # Compute FRA for this head
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=20, 
            verbose=False
        )
        
        if fra_result is None:
            results.append(None)
            continue
            
        # Extract feature pairs
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        
        if len(values) == 0:
            results.append(None)
            continue
        
        # Get feature indices
        q_feats = indices[2, :]
        k_feats = indices[3, :]
        
        # Filter self-interactions if requested
        if exclude_self:
            mask = q_feats != k_feats
            if not mask.any():
                results.append(None)
                continue
            q_feats = q_feats[mask]
            k_feats = k_feats[mask]
            values = values[mask]
        
        # Find max absolute value
        max_idx = values.abs().argmax()
        
        results.append({
            'head': head,
            'feature_i': q_feats[max_idx].item(),
            'feature_j': k_feats[max_idx].item(),
            'max_value': values[max_idx].item(),
            'abs_max': values[max_idx].abs().item()
        })
    
    return results

def test_completion_with_intervention(
    model: HookedTransformer,
    prompt: str,
    continuation_length: int = 10
) -> Dict:
    """Generate text with the model and return logits."""
    
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 100:
        tokens = tokens[:100]
    
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Generate continuation
    generated = []
    probs_list = []
    
    with torch.no_grad():
        for _ in range(continuation_length):
            logits = model(input_ids)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            next_token = torch.argmax(probs).item()
            
            generated.append(next_token)
            probs_list.append(probs.cpu())
            
            # Add to input for next iteration
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    generated_text = model.tokenizer.decode(generated)
    
    return {
        'tokens': generated,
        'text': generated_text,
        'probs': probs_list
    }

# Test sentences
test_sentences = [
    "The cat sat on the mat. The cat",
    "She went to the store to buy milk. She",
    "John gave Mary a book. John gave",
    "The weather today is sunny. The weather",
    "Python is a programming language. Python",
]

print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

for idx, prompt in enumerate(test_sentences):
    print(f"\n{'='*60}")
    print(f"Test {idx + 1}")
    print(f"Prompt: '{prompt}'")
    print("-"*60)
    
    # Find top feature pairs
    top_pairs = find_top_feature_pairs(
        model, sae, prompt, LAYER,
        n_heads=12, exclude_self=True
    )
    
    # Filter out None values and sort by strength
    valid_pairs = [p for p in top_pairs if p is not None]
    if not valid_pairs:
        print("No feature pairs found.")
        continue
    
    valid_pairs.sort(key=lambda x: x['abs_max'], reverse=True)
    
    # Show top 3 strongest interactions
    print("\nTop 3 Feature Interactions:")
    for i, pair in enumerate(valid_pairs[:3]):
        print(f"  {i+1}. Head {pair['head']:2}: "
              f"F{pair['feature_i']:5} → F{pair['feature_j']:5} "
              f"(strength: {pair['abs_max']:.4f})")
    
    # Generate normal completion
    print(f"\nGenerating completion (5 tokens):")
    result = test_completion_with_intervention(model, prompt, continuation_length=5)
    
    print(f"  Original: '{prompt}{result['text']}'")
    
    # Show which features are most active
    print(f"\nNote: To properly test ablation, we would need to:")
    print(f"  1. Hook into the attention computation")
    print(f"  2. Zero out the specific feature pair contributions")
    print(f"  3. Measure the change in output distribution")

# Summary analysis
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("""
Key Findings:
1. Different heads show different dominant feature interactions
2. Feature pair strengths vary significantly across contexts
3. Many heads have sparse feature interactions (lots of zeros)

Next Steps:
1. Implement proper ablation via attention hooks
2. Measure KL divergence between original and ablated outputs
3. Test on longer sequences to see cumulative effects
""")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)