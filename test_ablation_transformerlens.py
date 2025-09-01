#!/usr/bin/env python
"""Use TransformerLens hooks to ablate feature pairs in attention computation."""

import torch
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import torch.nn.functional as F
from typing import Dict, Tuple
from functools import partial

torch.set_grad_enabled(False)

print("="*60)
print("Feature Ablation using TransformerLens Hooks")
print("="*60)

# Configuration
LAYER = 5
HEAD = 0
DEVICE = 'cuda'

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)
print(f"Model: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
print(f"Testing Layer {LAYER}, Head {HEAD}")

def get_fra_and_find_max(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    text: str,
    layer: int,
    head: int
) -> Dict:
    """Get FRA tensor and find the max absolute value feature pair."""
    
    # Compute FRA
    fra_result = get_sentence_fra_batch(
        model, sae, text, 
        layer=layer, head=head,
        max_length=128, top_k=30,
        verbose=False
    )
    
    if fra_result is None or fra_result['total_interactions'] == 0:
        return None
    
    fra_sparse = fra_result['fra_tensor_sparse']
    indices = fra_sparse.indices()
    values = fra_sparse.values()
    
    # Find max absolute value
    max_idx = values.abs().argmax()
    
    return {
        'fra_sparse': fra_sparse,
        'indices': indices,
        'values': values,
        'max_idx': max_idx,
        'max_value': values[max_idx].item(),
        'max_abs': values[max_idx].abs().item(),
        'query_pos': indices[0, max_idx].item(),
        'key_pos': indices[1, max_idx].item(),
        'feature_i': indices[2, max_idx].item(),
        'feature_j': indices[3, max_idx].item(),
        'seq_len': fra_result['seq_len']
    }

def create_attention_pattern_from_fra(
    fra_info: Dict,
    ablate_max: bool = False
) -> torch.Tensor:
    """Sum FRA over feature dimensions to get attention pattern."""
    
    seq_len = fra_info['seq_len']
    indices = fra_info['indices']
    values = fra_info['values']
    
    # Create attention pattern by summing over feature dimensions
    pattern = torch.zeros((seq_len, seq_len), device=DEVICE)
    
    for i in range(indices.shape[1]):
        if ablate_max and i == fra_info['max_idx']:
            continue  # Skip the max feature pair
        
        q_pos = indices[0, i].item()
        k_pos = indices[1, i].item()
        pattern[q_pos, k_pos] += values[i].item()
    
    return pattern

def hook_fn_replace_attn_scores(
    attn_scores: torch.Tensor,
    hook,
    head_idx: int,
    new_pattern: torch.Tensor
):
    """Hook function to replace attention scores before softmax."""
    # attn_scores shape: [batch, head, seq, seq]
    seq_len = new_pattern.shape[0]
    attn_scores[:, head_idx, :seq_len, :seq_len] = new_pattern.unsqueeze(0)
    return attn_scores

def run_with_modified_attention(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
    new_attn_pattern: torch.Tensor,
    generate_n: int = 10
) -> Dict:
    """Run model with modified attention pattern."""
    
    tokens = model.tokenizer.encode(text)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Create hook function with the new pattern
    hook_fn = partial(
        hook_fn_replace_attn_scores,
        head_idx=head,
        new_pattern=new_attn_pattern
    )
    
    # Generate tokens one by one
    generated_tokens = []
    all_logits = []
    
    for _ in range(generate_n):
        # Run with hook on attention scores (before softmax)
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(input_ids)
        
        # Get next token
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        
        generated_tokens.append(next_token)
        all_logits.append(next_token_logits.cpu())
        
        # Append for next iteration
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    generated_text = model.tokenizer.decode(generated_tokens)
    
    return {
        'tokens': generated_tokens,
        'text': generated_text,
        'logits': torch.stack(all_logits)
    }

def analyze_single_example(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    prompt: str,
    layer: int,
    head: int
):
    """Analyze a single example with ablation."""
    
    print(f"\nPrompt: '{prompt}'")
    print("-"*60)
    
    # Get FRA and find max
    fra_info = get_fra_and_find_max(model, sae, prompt, layer, head)
    
    if fra_info is None:
        print("No feature interactions found!")
        return
    
    print(f"\nMax feature interaction:")
    print(f"  Features: F{fra_info['feature_i']} → F{fra_info['feature_j']}")
    print(f"  Position: ({fra_info['query_pos']}, {fra_info['key_pos']})")
    print(f"  Strength: {fra_info['max_abs']:.4f}")
    
    # Create original and ablated attention patterns
    original_pattern = create_attention_pattern_from_fra(fra_info, ablate_max=False)
    ablated_pattern = create_attention_pattern_from_fra(fra_info, ablate_max=True)
    
    # Run model with original pattern
    print(f"\nGenerating with ORIGINAL attention...")
    original_result = run_with_modified_attention(
        model, prompt, layer, head, original_pattern, generate_n=10
    )
    
    # Run model with ablated pattern
    print(f"Generating with ABLATED attention...")
    ablated_result = run_with_modified_attention(
        model, prompt, layer, head, ablated_pattern, generate_n=10
    )
    
    # Compare outputs
    print(f"\n" + "="*40)
    print("RESULTS:")
    print("-"*40)
    print(f"Original: {prompt}{original_result['text']}")
    print(f"Ablated:  {prompt}{ablated_result['text']}")
    
    # Check for differences
    if original_result['text'] != ablated_result['text']:
        print(f"\n✅ Ablation changed the output!")
        
        # Find first difference
        for i, (o, a) in enumerate(zip(original_result['tokens'], ablated_result['tokens'])):
            if o != a:
                o_word = model.tokenizer.decode([o])
                a_word = model.tokenizer.decode([a])
                print(f"  First difference at position {i}: '{o_word}' → '{a_word}'")
                break
    else:
        print(f"\n❌ No change in output")
    
    # Calculate KL divergence
    orig_probs = F.softmax(original_result['logits'], dim=-1)
    abl_probs = F.softmax(ablated_result['logits'], dim=-1)
    
    kl_divs = []
    for i in range(len(orig_probs)):
        kl = F.kl_div(abl_probs[i].log(), orig_probs[i], reduction='sum').item()
        kl_divs.append(kl)
    
    print(f"\nKL divergence: mean={np.mean(kl_divs):.4f}, max={np.max(kl_divs):.4f}")
    
    # Show attention pattern difference
    attn_diff = (original_pattern - ablated_pattern).abs()
    print(f"Attention difference: mean={attn_diff.mean():.6f}, max={attn_diff.max():.4f}")
    
    return {
        'original': original_result,
        'ablated': ablated_result,
        'kl_divs': kl_divs,
        'changed': original_result['text'] != ablated_result['text']
    }

# Test cases
test_prompts = [
    "The cat sat on the mat. The cat",
    "She went to the store and bought milk. She",  
    "John gave Mary a book. John gave",
    "The weather today is sunny. The weather",
    "Once upon a time there was a princess. Once upon",
]

print("\n" + "="*60)
print("RUNNING ABLATION EXPERIMENTS")
print("="*60)

results = []
for idx, prompt in enumerate(test_prompts):
    print(f"\n{'='*60}")
    print(f"Test {idx + 1}/{len(test_prompts)}")
    
    result = analyze_single_example(model, sae, prompt, LAYER, HEAD)
    if result:
        results.append(result)

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

changed_count = sum(1 for r in results if r['changed'])
print(f"Ablation changed output in {changed_count}/{len(results)} cases")

if results:
    all_kl_divs = [kl for r in results for kl in r['kl_divs']]
    print(f"Overall KL divergence: mean={np.mean(all_kl_divs):.4f}, max={np.max(all_kl_divs):.4f}")

print("\n" + "="*60)
print("Experiment complete!")
print("="*60)