#!/usr/bin/env python
"""Ablate the largest non-self feature interaction from ALL heads in layer 5."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from functools import partial

torch.set_grad_enabled(False)

print("="*60)
print("Multi-Head Feature Pair Ablation")
print("="*60)

# Configuration
LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)
print(f"Model loaded: Layer {LAYER}, {N_HEADS} heads")

def find_max_feature_pairs_all_heads(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    text: str,
    layer: int,
    exclude_self: bool = True
) -> Dict[int, Dict]:
    """Find the max non-self feature pair for each head."""
    
    max_pairs = {}
    
    for head in range(N_HEADS):
        # Compute FRA for this head
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=30,
            verbose=False
        )
        
        if fra_result is None or fra_result['total_interactions'] == 0:
            max_pairs[head] = None
            continue
        
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        
        if len(values) == 0:
            max_pairs[head] = None
            continue
        
        # Filter self-interactions if requested
        if exclude_self:
            q_feats = indices[2, :]
            k_feats = indices[3, :]
            mask = q_feats != k_feats
            
            if not mask.any():
                max_pairs[head] = None
                continue
                
            # Get max among non-self interactions
            non_self_values = values[mask].abs()
            max_idx_in_filtered = non_self_values.argmax()
            
            # Map back to original indices
            non_self_indices = torch.where(mask)[0]
            max_idx = non_self_indices[max_idx_in_filtered].item()
        else:
            max_idx = values.abs().argmax().item()
        
        max_pairs[head] = {
            'head': head,
            'max_idx': max_idx,
            'value': values[max_idx].item(),
            'abs_value': values[max_idx].abs().item(),
            'query_pos': indices[0, max_idx].item(),
            'key_pos': indices[1, max_idx].item(),
            'feature_i': indices[2, max_idx].item(),
            'feature_j': indices[3, max_idx].item(),
            'fra_sparse': fra_sparse,
            'seq_len': fra_result['seq_len']
        }
    
    return max_pairs

def create_ablated_attention_patterns(
    max_pairs: Dict[int, Dict]
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Create original and ablated attention patterns for each head."""
    
    patterns = {}
    
    for head, pair_info in max_pairs.items():
        if pair_info is None:
            patterns[head] = (None, None)
            continue
        
        fra_sparse = pair_info['fra_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        seq_len = pair_info['seq_len']
        
        # Create original pattern (sum over feature dimensions)
        original = torch.zeros((seq_len, seq_len), device=DEVICE)
        ablated = torch.zeros((seq_len, seq_len), device=DEVICE)
        
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            val = values[i].item()
            
            original[q_pos, k_pos] += val
            
            # Add to ablated unless it's the max pair
            if i != pair_info['max_idx']:
                ablated[q_pos, k_pos] += val
        
        patterns[head] = (original, ablated)
    
    return patterns

def multi_head_hook(
    attn_scores: torch.Tensor,
    hook,
    ablated_patterns: Dict[int, torch.Tensor]
):
    """Hook to replace attention scores for multiple heads."""
    # attn_scores shape: [batch, head, seq, seq]
    
    for head, pattern in ablated_patterns.items():
        if pattern is not None:
            seq_len = pattern.shape[0]
            # Replace with ablated pattern
            attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
    
    return attn_scores

def run_with_multi_ablation(
    model: HookedTransformer,
    text: str,
    layer: int,
    ablated_patterns: Dict[int, torch.Tensor],
    generate_n: int = 10
) -> Dict:
    """Run model with ablated patterns in multiple heads."""
    
    tokens = model.tokenizer.encode(text)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Generate tokens
    generated_tokens = []
    all_logits = []
    
    current_input = input_ids
    for _ in range(generate_n):
        # Create hook with ablated patterns
        hook_fn = partial(multi_head_hook, ablated_patterns=ablated_patterns)
        
        # Run with hook
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(current_input)
        
        # Get next token
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        
        generated_tokens.append(next_token)
        all_logits.append(next_token_logits.cpu())
        
        # Append for next iteration
        current_input = torch.cat([current_input, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    generated_text = model.tokenizer.decode(generated_tokens)
    
    return {
        'tokens': generated_tokens,
        'text': generated_text,
        'logits': torch.stack(all_logits)
    }

def analyze_multi_ablation(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    prompt: str,
    layer: int
):
    """Analyze effect of ablating max feature pairs across all heads."""
    
    print(f"\nPrompt: '{prompt}'")
    print("-"*60)
    
    # Find max feature pairs for all heads
    print("\nFinding max non-self feature pairs for all heads...")
    max_pairs = find_max_feature_pairs_all_heads(
        model, sae, prompt, layer, exclude_self=True
    )
    
    # Count how many heads have valid pairs
    valid_heads = [h for h, p in max_pairs.items() if p is not None]
    print(f"Found feature pairs in {len(valid_heads)}/{N_HEADS} heads")
    
    # Show the max pairs
    print("\nMax feature pairs per head:")
    for head in range(N_HEADS):
        if max_pairs[head] is not None:
            info = max_pairs[head]
            print(f"  Head {head:2}: F{info['feature_i']:5} → F{info['feature_j']:5} "
                  f"(strength: {info['abs_value']:.4f}, pos: {info['query_pos']},{info['key_pos']})")
        else:
            print(f"  Head {head:2}: No non-self interactions found")
    
    # Create ablated patterns
    patterns = create_ablated_attention_patterns(max_pairs)
    
    # Extract just the ablated patterns
    ablated_patterns = {h: abl for h, (orig, abl) in patterns.items()}
    original_patterns = {h: orig for h, (orig, abl) in patterns.items()}
    
    # Run with original patterns (should be close to normal)
    print(f"\nGenerating with ORIGINAL computed patterns...")
    original_result = run_with_multi_ablation(
        model, prompt, layer, original_patterns, generate_n=15
    )
    
    # Run with ablated patterns
    print(f"Generating with ABLATED patterns (max pairs removed)...")
    ablated_result = run_with_multi_ablation(
        model, prompt, layer, ablated_patterns, generate_n=15
    )
    
    # Run completely normal (no intervention)
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    normal_tokens = []
    for _ in range(15):
        with torch.no_grad():
            logits = model(input_ids)
        next_token = torch.argmax(logits[0, -1, :]).item()
        normal_tokens.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    normal_text = model.tokenizer.decode(normal_tokens)
    
    # Compare results
    print(f"\n" + "="*40)
    print("RESULTS:")
    print("-"*40)
    print(f"Normal:   {prompt} → {normal_text}")
    print(f"Original: {prompt} → {original_result['text']}")
    print(f"Ablated:  {prompt} → {ablated_result['text']}")
    
    # Check for differences
    changes_from_normal = sum(1 for n, a in zip(normal_tokens, ablated_result['tokens']) if n != a)
    changes_from_original = sum(1 for o, a in zip(original_result['tokens'], ablated_result['tokens']) if o != a)
    
    print(f"\nToken changes from normal: {changes_from_normal}/15")
    print(f"Token changes from original: {changes_from_original}/15")
    
    if changes_from_normal > 0:
        print(f"\n✅ Multi-head ablation changed the output!")
        
        # Show first difference
        for i, (n, a) in enumerate(zip(normal_tokens, ablated_result['tokens'])):
            if n != a:
                n_word = model.tokenizer.decode([n])
                a_word = model.tokenizer.decode([a])
                print(f"  First difference at position {i}: '{n_word}' → '{a_word}'")
                break
    else:
        print(f"\n❌ No change from normal output")
    
    # Calculate KL divergences
    normal_logits = model(torch.tensor(tokens).unsqueeze(0).to(DEVICE))
    normal_probs = F.softmax(normal_logits[0, -1, :], dim=-1)
    
    orig_probs = F.softmax(original_result['logits'][0], dim=-1)
    abl_probs = F.softmax(ablated_result['logits'][0], dim=-1)
    
    kl_orig = F.kl_div(orig_probs.log(), normal_probs.cpu(), reduction='sum').item()
    kl_abl = F.kl_div(abl_probs.log(), normal_probs.cpu(), reduction='sum').item()
    
    print(f"\nKL divergence from normal:")
    print(f"  Original patterns: {kl_orig:.4f}")
    print(f"  Ablated patterns: {kl_abl:.4f}")
    
    # Calculate total ablation strength
    total_ablated_strength = sum(p['abs_value'] for p in max_pairs.values() if p is not None)
    print(f"\nTotal ablated strength: {total_ablated_strength:.4f}")
    
    return {
        'changed': changes_from_normal > 0,
        'changes_from_normal': changes_from_normal,
        'changes_from_original': changes_from_original,
        'kl_abl': kl_abl,
        'total_strength': total_ablated_strength
    }

# Test cases
test_prompts = [
    "The cat sat on the mat. The cat",
    "She went to the store and bought milk. She",
    "John gave Mary a book. John gave",
    "The weather today is sunny. The weather",
    "Once upon a time there was a princess. Once upon",
    "Python is a programming language. Python",
]

print("\n" + "="*60)
print("RUNNING MULTI-HEAD ABLATION EXPERIMENTS")
print("="*60)

results = []
for idx, prompt in enumerate(test_prompts):
    print(f"\n{'='*60}")
    print(f"Test {idx + 1}/{len(test_prompts)}")
    
    result = analyze_multi_ablation(model, sae, prompt, LAYER)
    results.append(result)

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

changed_count = sum(1 for r in results if r['changed'])
print(f"\nAblation changed output in {changed_count}/{len(results)} cases")

avg_changes = np.mean([r['changes_from_normal'] for r in results])
max_changes = max(r['changes_from_normal'] for r in results)
print(f"Average token changes: {avg_changes:.1f}/15")
print(f"Maximum token changes: {max_changes}/15")

avg_kl = np.mean([r['kl_abl'] for r in results])
print(f"Average KL divergence: {avg_kl:.4f}")

avg_strength = np.mean([r['total_strength'] for r in results])
print(f"Average total ablated strength: {avg_strength:.2f}")

print("\n" + "="*60)
print("Experiment complete!")
print("="*60)