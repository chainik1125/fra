#!/usr/bin/env python
"""Compare targeted vs random feature ablation as a benchmark."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import json
from pathlib import Path
import random

torch.set_grad_enabled(False)

print("="*60)
print("Ablation Comparison: Targeted vs Random")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12
D_SAE = 49152  # SAE dimension

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
            max_length=128, top_k=20,
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

def find_random_feature_pairs_all_heads(model, sae, text, layer, max_pairs_reference):
    """Generate random feature pairs matching the structure of max pairs."""
    random_pairs = {}
    
    for head in range(N_HEADS):
        # If the max pairs had no valid pair for this head, skip
        if head not in max_pairs_reference or max_pairs_reference[head] is None:
            random_pairs[head] = None
            continue
            
        ref_pair = max_pairs_reference[head]
        
        # Get FRA result to have the right structure
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=20,
            verbose=False
        )
        
        if fra_result is None:
            random_pairs[head] = None
            continue
        
        # Generate random feature indices
        random_feat_i = random.randint(0, D_SAE - 1)
        random_feat_j = random.randint(0, D_SAE - 1)
        while random_feat_j == random_feat_i:  # Ensure non-self
            random_feat_j = random.randint(0, D_SAE - 1)
        
        # Use same positions as the max pair for consistency
        random_pairs[head] = {
            'value': ref_pair['abs_value'],  # Use same magnitude for fair comparison
            'abs_value': ref_pair['abs_value'],
            'query_pos': ref_pair['query_pos'],
            'key_pos': ref_pair['key_pos'],
            'feature_i': random_feat_i,
            'feature_j': random_feat_j,
            'fra_sparse': fra_result['fra_tensor_sparse'],
            'seq_len': fra_result['seq_len']
        }
    
    return random_pairs

def create_ablated_patterns(pairs_to_ablate, fra_reference):
    """Create ablated attention patterns."""
    ablated_patterns = {}
    
    for head, pair_info in pairs_to_ablate.items():
        if pair_info is None:
            ablated_patterns[head] = None
            continue
        
        # Use the original FRA sparse tensor from reference
        if head in fra_reference and fra_reference[head] is not None:
            fra_sparse = fra_reference[head]['fra_sparse']
        else:
            ablated_patterns[head] = None
            continue
            
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        seq_len = pair_info['seq_len']
        
        ablated = torch.zeros((seq_len, seq_len), device=DEVICE)
        
        # Add all interactions except the specified pair
        for i in range(indices.shape[1]):
            q_pos = indices[0, i].item()
            k_pos = indices[1, i].item()
            feat_i = indices[2, i].item()
            feat_j = indices[3, i].item()
            
            # Skip if this matches the pair to ablate
            if (q_pos == pair_info['query_pos'] and 
                k_pos == pair_info['key_pos'] and
                feat_i == pair_info['feature_i'] and
                feat_j == pair_info['feature_j']):
                continue
                
            ablated[q_pos, k_pos] += values[i].item()
        
        ablated_patterns[head] = ablated
    
    return ablated_patterns

def analyze_prompt_with_benchmark(model, sae, prompt, layer):
    """Analyze prompt with both targeted and random ablation."""
    
    # Find max feature pairs
    max_pairs = find_max_feature_pairs_all_heads(model, sae, prompt, layer)
    
    # Generate random pairs with same structure
    random_pairs = find_random_feature_pairs_all_heads(model, sae, prompt, layer, max_pairs)
    
    # Get normal completion
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    normal_tokens = []
    current = input_ids
    for _ in range(10):
        with torch.no_grad():
            logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        normal_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Get targeted ablation completion
    targeted_patterns = create_ablated_patterns(max_pairs, max_pairs)
    
    targeted_tokens = []
    current = input_ids
    for _ in range(10):
        def hook_fn(attn_scores, hook):
            for head, pattern in targeted_patterns.items():
                if pattern is not None:
                    seq_len = pattern.shape[0]
                    attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
            return attn_scores
        
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(current)
        
        next_token = torch.argmax(logits[0, -1, :]).item()
        targeted_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Get random ablation completion
    random_patterns = create_ablated_patterns(random_pairs, max_pairs)
    
    random_ablation_tokens = []
    current = input_ids
    for _ in range(10):
        def hook_fn(attn_scores, hook):
            for head, pattern in random_patterns.items():
                if pattern is not None:
                    seq_len = pattern.shape[0]
                    attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
            return attn_scores
        
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(current)
        
        next_token = torch.argmax(logits[0, -1, :]).item()
        random_ablation_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Decode
    normal_text = model.tokenizer.decode(normal_tokens)
    targeted_text = model.tokenizer.decode(targeted_tokens)
    random_text = model.tokenizer.decode(random_ablation_tokens)
    
    # Count differences
    targeted_diffs = sum(1 for n, t in zip(normal_tokens, targeted_tokens) if n != t)
    random_diffs = sum(1 for n, r in zip(normal_tokens, random_ablation_tokens) if n != r)
    
    # Calculate ablation strength
    total_strength = sum(p['abs_value'] for p in max_pairs.values() if p is not None)
    
    # Get top feature info
    top_targeted = []
    top_random = []
    
    for h, p in max_pairs.items():
        if p is not None:
            top_targeted.append((h, p['feature_i'], p['feature_j'], p['abs_value']))
    
    for h, p in random_pairs.items():
        if p is not None:
            top_random.append((h, p['feature_i'], p['feature_j'], p['abs_value']))
    
    return {
        'prompt': prompt,
        'normal': normal_text,
        'targeted': targeted_text,
        'random': random_text,
        'targeted_changed': targeted_diffs > 0,
        'random_changed': random_diffs > 0,
        'targeted_diffs': targeted_diffs,
        'random_diffs': random_diffs,
        'total_strength': total_strength,
        'top_targeted': sorted(top_targeted, key=lambda x: x[3], reverse=True)[:5],
        'top_random': sorted(top_random, key=lambda x: x[3], reverse=True)[:5]
    }

# Test prompts
test_prompts = [
    # IOI
    "When John and Mary went to the store, John gave a drink to",
    
    # Question
    "The capital of France is",
    
    # Factual
    "The Earth orbits around the",
    
    # Story
    "Once upon a time, there lived a wise old wizard who",
    
    # Another IOI
    "Alice and Bob went to the park. Alice threw the ball to",
    
    # Another factual
    "Water is made of hydrogen and",
]

print("\nAnalyzing prompts with benchmark...")
results = []

for i, prompt in enumerate(test_prompts):
    print(f"\n{i+1}. {prompt[:50]}...")
    result = analyze_prompt_with_benchmark(model, sae, prompt, LAYER)
    results.append(result)
    
    print(f"   Normal:   → {result['normal'][:30]}...")
    print(f"   Targeted: → {result['targeted'][:30]}... ({result['targeted_diffs']}/10 changed)")
    print(f"   Random:   → {result['random'][:30]}... ({result['random_diffs']}/10 changed)")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

targeted_changed = sum(1 for r in results if r['targeted_changed'])
random_changed = sum(1 for r in results if r['random_changed'])

print(f"\nTargeted ablation: {targeted_changed}/{len(results)} changed")
print(f"Random ablation:   {random_changed}/{len(results)} changed")
print(f"\nAvg token differences:")
print(f"  Targeted: {np.mean([r['targeted_diffs'] for r in results]):.1f}/10")
print(f"  Random:   {np.mean([r['random_diffs'] for r in results]):.1f}/10")

# Generate HTML dashboard
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ablation Benchmark: Targeted vs Random</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .case {{ 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .prompt {{ 
            background: #ecf0f1; 
            padding: 12px; 
            margin: 15px 0; 
            font-family: 'Courier New', monospace;
            border-radius: 4px;
            font-size: 14px;
        }}
        .completions {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr 1fr; 
            gap: 15px;
            margin: 20px 0;
        }}
        .completion {{
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
        }}
        .normal {{ 
            background: #e8f5e9;
            border-color: #4caf50;
        }}
        .targeted {{ 
            background: #ffebee;
            border-color: #f44336;
        }}
        .random {{ 
            background: #e3f2fd;
            border-color: #2196f3;
        }}
        .completion-header {{
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .completion-text {{
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .changed {{ 
            color: #d32f2f;
            font-weight: bold;
        }}
        .unchanged {{ 
            color: #388e3c;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 4px;
        }}
        .features {{ 
            background: #fafafa; 
            padding: 12px; 
            margin-top: 15px; 
            font-size: 12px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        .feature-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 10px;
        }}
        .feature-list {{
            font-family: 'Courier New', monospace;
            font-size: 11px;
            line-height: 1.4;
        }}
        .effectiveness {{
            background: #fff9c4;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>🔬 Ablation Benchmark: Targeted vs Random Feature Pairs</h1>
    
    <div class="summary">
        <h2>Overall Results</h2>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{targeted_changed}/{len(results)}</div>
                <div class="stat-label">Targeted Changed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{random_changed}/{len(results)}</div>
                <div class="stat-label">Random Changed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{np.mean([r['targeted_diffs'] for r in results]):.1f}</div>
                <div class="stat-label">Avg Targeted Diffs</div>
            </div>
            <div class="stat">
                <div class="stat-value">{np.mean([r['random_diffs'] for r in results]):.1f}</div>
                <div class="stat-label">Avg Random Diffs</div>
            </div>
        </div>
    </div>
"""

prompt_types = ["IOI", "Question", "Factual", "Story", "IOI", "Factual"]

for i, (r, ptype) in enumerate(zip(results, prompt_types)):
    targeted_status = "changed" if r['targeted_changed'] else "unchanged"
    random_status = "changed" if r['random_changed'] else "unchanged"
    
    # Calculate effectiveness
    effectiveness_ratio = r['targeted_diffs'] / max(r['random_diffs'], 0.1)
    
    html += f"""
    <div class="case">
        <h3>Test {i+1}: {ptype}</h3>
        <div class="prompt">{r['prompt']}</div>
        
        <div class="completions">
            <div class="completion normal">
                <div class="completion-header">✓ Normal (Baseline)</div>
                <div class="completion-text">{r['normal']}</div>
            </div>
            <div class="completion targeted">
                <div class="completion-header">
                    🎯 Targeted Ablation
                    <span class="{targeted_status}">
                        ({r['targeted_diffs']}/10 changed)
                    </span>
                </div>
                <div class="completion-text">{r['targeted']}</div>
            </div>
            <div class="completion random">
                <div class="completion-header">
                    🎲 Random Ablation
                    <span class="{random_status}">
                        ({r['random_diffs']}/10 changed)
                    </span>
                </div>
                <div class="completion-text">{r['random']}</div>
            </div>
        </div>
        
        <div class="effectiveness">
            <strong>Effectiveness:</strong> 
            Targeted ablation is {effectiveness_ratio:.1f}x more effective than random
            (Total ablation strength: {r['total_strength']:.2f})
        </div>
        
        <div class="features">
            <strong>Ablated Features:</strong>
            <div class="feature-grid">
                <div>
                    <strong>Targeted (Max FRA pairs):</strong>
                    <div class="feature-list">
"""
    
    for h, fi, fj, strength in r['top_targeted'][:3]:
        html += f"Head {h:2}: F{fi:5}→F{fj:5} (str: {strength:.3f})<br>"
    
    html += f"""
                    </div>
                </div>
                <div>
                    <strong>Random control:</strong>
                    <div class="feature-list">
"""
    
    for h, fi, fj, strength in r['top_random'][:3]:
        html += f"Head {h:2}: F{fi:5}→F{fj:5} (str: {strength:.3f})<br>"
    
    html += f"""
                    </div>
                </div>
            </div>
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

html_path = output_dir / "ablation_benchmark_dashboard.html"
with open(html_path, 'w') as f:
    f.write(html)

# Save JSON
json_data = {
    'summary': {
        'targeted_changed': targeted_changed,
        'random_changed': random_changed,
        'total_tests': len(results),
        'avg_targeted_diffs': float(np.mean([r['targeted_diffs'] for r in results])),
        'avg_random_diffs': float(np.mean([r['random_diffs'] for r in results]))
    },
    'results': [
        {
            'prompt': r['prompt'],
            'prompt_type': ptype,
            'normal': r['normal'],
            'targeted': r['targeted'],
            'random': r['random'],
            'targeted_diffs': r['targeted_diffs'],
            'random_diffs': r['random_diffs'],
            'effectiveness_ratio': r['targeted_diffs'] / max(r['random_diffs'], 0.1)
        }
        for r, ptype in zip(results, prompt_types)
    ]
}

json_path = output_dir / "ablation_benchmark.json"
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"\n✅ Dashboard saved to: {html_path}")
print(f"✅ JSON saved to: {json_path}")

print("="*60)