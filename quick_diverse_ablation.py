#!/usr/bin/env python
"""Quick diverse prompt ablation test."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import json
from pathlib import Path

torch.set_grad_enabled(False)

print("="*60)
print("Quick Diverse Prompt Ablation Test")
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
            max_length=128, top_k=20,  # Reduced top_k
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

def analyze_prompt(model, sae, prompt, layer):
    """Analyze a single prompt with ablation."""
    
    # Find max pairs
    max_pairs = find_max_feature_pairs_all_heads(model, sae, prompt, layer)
    
    # Get normal completion
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    normal_tokens = []
    current = input_ids
    for _ in range(10):  # Reduced to 10 tokens
        with torch.no_grad():
            logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        normal_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Get ablated completion
    ablated_patterns = create_ablated_patterns(max_pairs)
    
    ablated_tokens = []
    current = input_ids
    for _ in range(10):  # Reduced to 10 tokens
        def hook_fn(attn_scores, hook):
            for head, pattern in ablated_patterns.items():
                if pattern is not None:
                    seq_len = pattern.shape[0]
                    attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
            return attn_scores
        
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(current)
        
        next_token = torch.argmax(logits[0, -1, :]).item()
        ablated_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Decode
    normal_text = model.tokenizer.decode(normal_tokens)
    ablated_text = model.tokenizer.decode(ablated_tokens)
    
    # Count differences
    differences = sum(1 for n, a in zip(normal_tokens, ablated_tokens) if n != a)
    
    # Calculate total ablation strength
    total_strength = sum(p['abs_value'] for p in max_pairs.values() if p is not None)
    
    # Get top features
    top_features = []
    for h, p in max_pairs.items():
        if p is not None:
            top_features.append((h, p['feature_i'], p['feature_j'], p['abs_value']))
    
    return {
        'prompt': prompt,
        'normal': normal_text,
        'ablated': ablated_text,
        'changed': differences > 0,
        'differences': differences,
        'total_strength': total_strength,
        'top_features': sorted(top_features, key=lambda x: x[3], reverse=True)[:5]
    }

# Test prompts - fewer for speed
test_prompts = [
    # IOI
    "When John and Mary went to the store, John gave a drink to",
    
    # Question
    "The capital of France is",
    
    # Factual
    "The Earth orbits around the",
    
    # Story
    "Once upon a time, there lived a wise old wizard who",
]

print("\nAnalyzing prompts...")
results = []

for i, prompt in enumerate(test_prompts):
    print(f"\n{i+1}. {prompt[:50]}...")
    result = analyze_prompt(model, sae, prompt, LAYER)
    results.append(result)
    
    print(f"   Normal:  → {result['normal'][:40]}")
    print(f"   Ablated: → {result['ablated'][:40]}")
    print(f"   Changed: {result['changed']} ({result['differences']}/10 tokens)")
    print(f"   Strength: {result['total_strength']:.2f}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

changed = sum(1 for r in results if r['changed'])
print(f"\nChanged: {changed}/{len(results)}")
print(f"Avg differences: {np.mean([r['differences'] for r in results]):.1f}/10")
print(f"Avg strength: {np.mean([r['total_strength'] for r in results]):.2f}")

# Save simple HTML
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Diverse Ablation Results</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .case {{ background: white; padding: 15px; margin: 20px 0; border: 1px solid #ddd; border-radius: 8px; }}
        .prompt {{ background: #f0f0f0; padding: 10px; margin: 10px 0; font-family: monospace; }}
        .completions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .normal {{ background: #e8f5e9; padding: 10px; border-radius: 4px; }}
        .ablated {{ background: #ffebee; padding: 10px; border-radius: 4px; }}
        .changed {{ color: green; font-weight: bold; }}
        .unchanged {{ color: red; }}
        .features {{ background: #f5f5f5; padding: 10px; margin-top: 10px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Diverse Prompt Ablation Results</h1>
    <p>Layer {LAYER}, {changed}/{len(results)} prompts changed</p>
"""

for i, r in enumerate(results):
    status = "changed" if r['changed'] else "unchanged"
    html += f"""
    <div class="case">
        <h3>Test {i+1}: <span class="{status}">{'✓ Changed' if r['changed'] else '✗ Unchanged'}</span></h3>
        <div class="prompt">{r['prompt']}</div>
        <div class="completions">
            <div class="normal">
                <strong>Normal:</strong><br>
                {r['normal']}
            </div>
            <div class="ablated">
                <strong>Ablated:</strong><br>
                {r['ablated']}
            </div>
        </div>
        <div class="features">
            <strong>Top ablated pairs:</strong><br>
"""
    for h, fi, fj, strength in r['top_features'][:3]:
        html += f"Head {h}: F{fi}→F{fj} (strength: {strength:.3f})<br>"
    
    html += f"""
            <br>Total strength: {r['total_strength']:.2f} | Differences: {r['differences']}/10
        </div>
    </div>
"""

html += """
</body>
</html>
"""

# Save
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

html_path = output_dir / "quick_diverse_ablation.html"
with open(html_path, 'w') as f:
    f.write(html)

print(f"\n✅ Results saved to: {html_path}")

print("="*60)