#!/usr/bin/env python
"""Test ablation with diverse prompts including IOI, questions, and factual completion."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import json
from pathlib import Path
from typing import Dict, List

torch.set_grad_enabled(False)

print("="*60)
print("Diverse Prompt Ablation Analysis with Loss Metrics")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def find_max_feature_pairs_all_heads(model, sae, text, layer):
    """Find max non-self feature pairs for all heads."""
    max_pairs = {}
    
    for head in range(N_HEADS):
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
        
        # Filter self-interactions
        q_feats = indices[2, :]
        k_feats = indices[3, :]
        mask = q_feats != k_feats
        
        if not mask.any():
            max_pairs[head] = None
            continue
        
        non_self_values = values[mask].abs()
        max_idx_in_filtered = non_self_values.argmax()
        non_self_indices = torch.where(mask)[0]
        max_idx = non_self_indices[max_idx_in_filtered].item()
        
        max_pairs[head] = {
            'head': head,
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

def create_ablated_patterns(max_pairs):
    """Create ablated attention patterns."""
    ablated_patterns = {}
    
    for head, pair_info in max_pairs.items():
        if pair_info is None:
            ablated_patterns[head] = None
            continue
        
        fra_sparse = pair_info['fra_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        seq_len = pair_info['seq_len']
        
        ablated = torch.zeros((seq_len, seq_len), device=DEVICE)
        
        # Add all interactions except the max non-self pair
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            
            # Skip if this is the max non-self pair for this head
            if (q_pos == pair_info['query_pos'] and 
                k_pos == pair_info['key_pos'] and
                indices[2, i].item() == pair_info['feature_i'] and
                indices[3, i].item() == pair_info['feature_j']):
                continue
                
            ablated[q_pos, k_pos] += values[i].item()
        
        ablated_patterns[head] = ablated
    
    return ablated_patterns

def analyze_with_loss(model, sae, prompt, layer, generate_n=10):
    """Analyze prompt with ablation and detailed loss metrics."""
    
    # Find max pairs
    max_pairs = find_max_feature_pairs_all_heads(model, sae, prompt, layer)
    
    # Tokenize
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Get normal generation and loss
    with torch.no_grad():
        normal_logits = model(input_ids)
    
    # Generate normally
    normal_tokens = []
    current = input_ids
    all_normal_logits = []
    
    for _ in range(generate_n):
        with torch.no_grad():
            logits = model(current)
        all_normal_logits.append(logits[0, -1, :].cpu())
        next_token = torch.argmax(logits[0, -1, :]).item()
        normal_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Create ablated patterns
    ablated_patterns = create_ablated_patterns(max_pairs)
    
    # Hook function for ablation
    def hook_fn(attn_scores, hook):
        for head, pattern in ablated_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    # Get ablated generation
    ablated_tokens = []
    current = input_ids
    all_ablated_logits = []
    
    for _ in range(generate_n):
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(current)
        all_ablated_logits.append(logits[0, -1, :].cpu())
        next_token = torch.argmax(logits[0, -1, :]).item()
        ablated_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Calculate loss on the generated sequences
    # We'll calculate the loss of predicting each generated token given the prompt
    
    # Normal sequence loss
    normal_sequence = torch.cat([input_ids, torch.tensor([normal_tokens]).to(DEVICE)], dim=1)
    with torch.no_grad():
        normal_full_logits = model(normal_sequence[:, :-1])
    normal_loss = F.cross_entropy(
        normal_full_logits[0, len(tokens)-1:, :].reshape(-1, normal_full_logits.shape[-1]),
        normal_sequence[0, len(tokens):].reshape(-1),
        reduction='mean'
    ).item()
    
    # Ablated sequence loss (with hook active)
    ablated_sequence = torch.cat([input_ids, torch.tensor([ablated_tokens]).to(DEVICE)], dim=1)
    with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
        ablated_full_logits = model(ablated_sequence[:, :-1])
    ablated_loss = F.cross_entropy(
        ablated_full_logits[0, len(tokens)-1:, :].reshape(-1, ablated_full_logits.shape[-1]),
        ablated_sequence[0, len(tokens):].reshape(-1),
        reduction='mean'
    ).item()
    
    # Also calculate the loss of normal sequence under ablation (cross-evaluation)
    with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
        cross_logits = model(normal_sequence[:, :-1])
    cross_loss = F.cross_entropy(
        cross_logits[0, len(tokens)-1:, :].reshape(-1, cross_logits.shape[-1]),
        normal_sequence[0, len(tokens):].reshape(-1),
        reduction='mean'
    ).item()
    
    # Calculate KL divergence for first generated token
    normal_probs = F.softmax(all_normal_logits[0], dim=-1)
    ablated_probs = F.softmax(all_ablated_logits[0], dim=-1)
    kl_div = F.kl_div(ablated_probs.log(), normal_probs, reduction='sum').item()
    
    # Decode
    normal_text = model.tokenizer.decode(normal_tokens)
    ablated_text = model.tokenizer.decode(ablated_tokens)
    
    # Count token differences
    differences = sum(1 for n, a in zip(normal_tokens, ablated_tokens) if n != a)
    
    # Calculate total ablation strength
    total_strength = sum(p['abs_value'] for p in max_pairs.values() if p is not None)
    
    return {
        'prompt': prompt,
        'normal_completion': normal_text,
        'ablated_completion': ablated_text,
        'normal_tokens': normal_tokens,
        'ablated_tokens': ablated_tokens,
        'differences': differences,
        'changed': differences > 0,
        'loss_metrics': {
            'normal_loss': normal_loss,
            'ablated_loss': ablated_loss,
            'cross_loss': cross_loss,  # Loss of normal sequence under ablation
            'loss_increase': cross_loss - normal_loss,  # How much worse normal sequence becomes
            'kl_divergence': kl_div,
            'normal_perplexity': np.exp(normal_loss),
            'ablated_perplexity': np.exp(ablated_loss),
            'cross_perplexity': np.exp(cross_loss)
        },
        'ablation_strength': total_strength,
        'max_pairs': max_pairs
    }

# Diverse test prompts
test_prompts = [
    # IOI (Indirect Object Identification) prompts
    "When John and Mary went to the store, John gave a drink to",  # Should complete "Mary"
    "After Sarah and Tom visited the museum, Tom handed the tickets to",  # Should complete "Sarah"
    
    # Question with definite answer
    "The capital of France is",  # Should complete "Paris"
    "Q: What is 2+2? A:",  # Should complete with "4" or "Four"
    
    # Factual completion
    "The Earth orbits around the",  # Should complete "Sun"
    "Water freezes at 0 degrees",  # Should complete "Celsius" or "Centigrade"
    
    # Story continuation (less definite)
    "Once upon a time, there lived a wise old wizard who",
    "The scientist discovered something amazing when she looked through the microscope and saw",
]

print(f"\nAnalyzing {len(test_prompts)} diverse prompts...")
print("="*60)

results = []
for i, prompt in enumerate(test_prompts):
    print(f"\nTest {i+1}/{len(test_prompts)}: {prompt[:50]}...")
    result = analyze_with_loss(model, sae, prompt, LAYER, generate_n=15)
    results.append(result)
    
    print(f"  Normal:  → {result['normal_completion'][:40]}")
    print(f"  Ablated: → {result['ablated_completion'][:40]}")
    print(f"  Changed: {result['changed']} ({result['differences']}/15 tokens)")
    print(f"  Loss increase: {result['loss_metrics']['loss_increase']:+.3f}")
    print(f"  Cross perplexity: {result['loss_metrics']['cross_perplexity']:.1f} (from {result['loss_metrics']['normal_perplexity']:.1f})")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

changed_count = sum(1 for r in results if r['changed'])
avg_loss_increase = np.mean([r['loss_metrics']['loss_increase'] for r in results])
avg_kl = np.mean([r['loss_metrics']['kl_divergence'] for r in results])
avg_differences = np.mean([r['differences'] for r in results])

print(f"\nOutputs changed: {changed_count}/{len(results)}")
print(f"Average token differences: {avg_differences:.1f}/15")
print(f"Average loss increase: {avg_loss_increase:+.3f}")
print(f"Average KL divergence: {avg_kl:.3f}")

# Breakdown by prompt type
print("\nBy prompt type:")
print("-"*40)

ioi_results = results[:2]
question_results = results[2:4]
factual_results = results[4:6]
story_results = results[6:]

for name, subset in [("IOI", ioi_results), ("Questions", question_results), 
                     ("Factual", factual_results), ("Story", story_results)]:
    if subset:
        avg_loss = np.mean([r['loss_metrics']['loss_increase'] for r in subset])
        changed = sum(1 for r in subset if r['changed'])
        print(f"  {name}: {changed}/{len(subset)} changed, loss increase: {avg_loss:+.3f}")

# Save results
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

json_data = {
    'layer': LAYER,
    'summary': {
        'changed_count': changed_count,
        'total_tests': len(results),
        'avg_loss_increase': avg_loss_increase,
        'avg_kl_divergence': avg_kl,
        'avg_token_differences': avg_differences
    },
    'results': [
        {
            'prompt': r['prompt'],
            'normal': r['normal_completion'],
            'ablated': r['ablated_completion'],
            'changed': r['changed'],
            'differences': r['differences'],
            'loss_metrics': r['loss_metrics'],
            'ablation_strength': r['ablation_strength']
        }
        for r in results
    ]
}

json_path = output_dir / "diverse_ablation_results.json"
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"\n✅ Results saved to: {json_path}")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)