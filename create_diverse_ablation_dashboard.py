#!/usr/bin/env python
"""Create dashboard showing multi-head ablation effects on diverse prompts."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import json
import requests
from typing import Dict, List, Tuple
from pathlib import Path

torch.set_grad_enabled(False)

print("="*60)
print("Creating Diverse Prompt Ablation Dashboard")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def get_feature_description(feature_id: int, layer: int) -> str:
    """Fetch feature description from Neuronpedia."""
    try:
        url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-res-jb/{feature_id}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('description', f'Feature {feature_id}')
        else:
            return f"Feature {feature_id}"
    except:
        return f"Feature {feature_id}"

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

def run_with_ablation(model, prompt, layer, ablated_patterns, generate_n=15):
    """Generate tokens with ablation."""
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    generated_tokens = []
    current_input = input_ids
    
    for _ in range(generate_n):
        def hook_fn(attn_scores, hook):
            for head, pattern in ablated_patterns.items():
                if pattern is not None:
                    seq_len = pattern.shape[0]
                    attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
            return attn_scores
        
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            logits = model(current_input)
        
        next_token = torch.argmax(logits[0, -1, :]).item()
        generated_tokens.append(next_token)
        current_input = torch.cat([current_input, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    return generated_tokens

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
    for _ in range(15):
        with torch.no_grad():
            logits = model(current)
        next_token = torch.argmax(logits[0, -1, :]).item()
        normal_tokens.append(next_token)
        current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)
    
    # Get ablated completion
    ablated_patterns = create_ablated_patterns(max_pairs)
    ablated_tokens = run_with_ablation(model, prompt, layer, ablated_patterns, 15)
    
    # Decode
    normal_text = model.tokenizer.decode(normal_tokens)
    ablated_text = model.tokenizer.decode(ablated_tokens)
    
    # Find differences
    differences = []
    for i, (n, a) in enumerate(zip(normal_tokens, ablated_tokens)):
        if n != a:
            differences.append({
                'position': i,
                'normal': model.tokenizer.decode([n]),
                'ablated': model.tokenizer.decode([a])
            })
    
    # Calculate total ablation strength
    total_strength = sum(p['abs_value'] for p in max_pairs.values() if p is not None)
    
    return {
        'prompt': prompt,
        'normal_completion': normal_text,
        'ablated_completion': ablated_text,
        'normal_tokens': normal_tokens,
        'ablated_tokens': ablated_tokens,
        'differences': differences,
        'max_pairs': max_pairs,
        'changed': len(differences) > 0,
        'total_strength': total_strength
    }

# Diverse test prompts
test_prompts = [
    # IOI (Indirect Object Identification) prompts
    "When John and Mary went to the store, John gave a drink to",
    "After Sarah and Tom visited the museum, Tom handed the tickets to",
    
    # Question with definite answer
    "The capital of France is",
    "Q: What is 2+2? A:",
    
    # Factual completion
    "The Earth orbits around the",
    "Water freezes at 0 degrees",
    
    # Story continuation (less definite)
    "Once upon a time, there lived a wise old wizard who",
    "The scientist discovered something amazing when she looked through the microscope and saw",
]

print("\nAnalyzing test prompts...")
results = []

for i, prompt in enumerate(test_prompts):
    print(f"  {i+1}/{len(test_prompts)}: {prompt[:50]}...")
    result = analyze_prompt(model, sae, prompt, LAYER)
    results.append(result)

# Get feature descriptions
print("\nFetching feature descriptions from Neuronpedia...")
all_features = set()
for result in results:
    for head, pair in result['max_pairs'].items():
        if pair is not None:
            all_features.add(pair['feature_i'])
            all_features.add(pair['feature_j'])

feature_descriptions = {}
for feat_id in sorted(all_features)[:30]:  # Limit to 30 to avoid timeout
    desc = get_feature_description(feat_id, LAYER)
    feature_descriptions[feat_id] = desc
    print(f"  F{feat_id}: {desc[:50]}...")

# Generate dashboard HTML
print("\nGenerating dashboard...")

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diverse Prompt Multi-Head Ablation Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .test-case {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .test-header {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }
        .prompt-type {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
            font-weight: normal;
        }
        .type-ioi { background: #e3f2fd; color: #1976d2; }
        .type-question { background: #f3e5f5; color: #7b1fa2; }
        .type-factual { background: #e8f5e9; color: #388e3c; }
        .type-story { background: #fff3e0; color: #f57c00; }
        .prompt {
            background: #ecf0f1;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-family: monospace;
        }
        .completions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .completion {
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .normal { background: #e8f5e9; }
        .ablated { background: #ffebee; }
        .completion-text {
            font-family: monospace;
            margin-top: 10px;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .features {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
        }
        .feature-pairs {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .feature-pair {
            background: white;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
            font-size: 12px;
        }
        .feature-id {
            font-weight: bold;
            color: #3498db;
        }
        .changed { 
            background: #ffcdd2;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .unchanged {
            background: #c8e6c9;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        .token-diff {
            background: #fffde7;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 13px;
        }
        .diff-item {
            margin: 3px 0;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <h1>🔬 Diverse Prompt Multi-Head Ablation Dashboard - Layer """ + str(LAYER) + """</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="stats">
"""

# Calculate summary statistics
changed_count = sum(1 for r in results if r['changed'])
avg_differences = np.mean([len(r['differences']) for r in results])
avg_strength = np.mean([r['total_strength'] for r in results])

# By category
ioi_changed = sum(1 for r in results[:2] if r['changed'])
question_changed = sum(1 for r in results[2:4] if r['changed'])
factual_changed = sum(1 for r in results[4:6] if r['changed'])
story_changed = sum(1 for r in results[6:] if r['changed'])

html_content += f"""
            <div class="stat-card">
                <div class="stat-value">{changed_count}/{len(results)}</div>
                <div class="stat-label">Total Changed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_differences:.1f}</div>
                <div class="stat-label">Avg Token Differences</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_strength:.2f}</div>
                <div class="stat-label">Avg Ablation Strength</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{ioi_changed}/2</div>
                <div class="stat-label">IOI Changed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{question_changed}/2</div>
                <div class="stat-label">Questions Changed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{factual_changed}/2</div>
                <div class="stat-label">Factual Changed</div>
            </div>
        </div>
    </div>
"""

# Determine prompt types
prompt_types = ["IOI", "IOI", "Question", "Question", "Factual", "Factual", "Story", "Story"]
type_classes = ["type-ioi", "type-ioi", "type-question", "type-question", 
                "type-factual", "type-factual", "type-story", "type-story"]

# Add test cases
for i, (result, ptype, tclass) in enumerate(zip(results, prompt_types, type_classes)):
    status = "✅ CHANGED" if result['changed'] else "❌ UNCHANGED"
    
    html_content += f"""
    <div class="test-case">
        <div class="test-header">
            Test {i+1}: {status}
            <span class="prompt-type {tclass}">{ptype}</span>
        </div>
        <div class="prompt">Prompt: {result['prompt']}</div>
        
        <div class="completions">
            <div class="completion normal">
                <strong>Normal Completion:</strong>
                <div class="completion-text">{result['normal_completion']}</div>
            </div>
            <div class="completion ablated">
                <strong>Ablated Completion:</strong>
                <div class="completion-text">{result['ablated_completion']}</div>
            </div>
        </div>
"""
    
    # Add token differences if any
    if result['differences']:
        html_content += """
        <div class="token-diff">
            <strong>Token-level differences:</strong>
"""
        for diff in result['differences'][:5]:  # Show first 5 differences
            html_content += f"""
            <div class="diff-item">Position {diff['position']}: "{diff['normal']}" → "{diff['ablated']}"</div>
"""
        if len(result['differences']) > 5:
            html_content += f"""
            <div class="diff-item">... and {len(result['differences']) - 5} more differences</div>
"""
        html_content += """
        </div>
"""
    
    # Add ablated features
    html_content += """
        <div class="features">
            <strong>Ablated Feature Pairs (max non-self per head):</strong>
            <div class="feature-pairs">
"""
    
    # Count valid pairs
    valid_pairs = [(h, p) for h, p in result['max_pairs'].items() if p is not None]
    
    for head, pair in valid_pairs[:12]:  # Show up to 12
        feat_i = pair['feature_i']
        feat_j = pair['feature_j']
        strength = pair['abs_value']
        desc_i = feature_descriptions.get(feat_i, f"Feature {feat_i}")[:30]
        desc_j = feature_descriptions.get(feat_j, f"Feature {feat_j}")[:30]
        
        html_content += f"""
                <div class="feature-pair">
                    <strong>Head {head}:</strong> 
                    <span class="feature-id">F{feat_i}</span> → <span class="feature-id">F{feat_j}</span>
                    (strength: {strength:.3f})<br>
                    <small>{desc_i}... → {desc_j}...</small>
                </div>
"""
    
    html_content += f"""
            </div>
            <div style="margin-top: 10px; font-size: 13px; color: #666;">
                Total ablated strength: {result['total_strength']:.3f} | 
                Valid pairs: {len(valid_pairs)}/{N_HEADS} heads
            </div>
        </div>
    </div>
"""

html_content += """
</body>
</html>
"""

# Save results
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

html_path = output_dir / "diverse_ablation_dashboard.html"
with open(html_path, 'w') as f:
    f.write(html_content)

# Save JSON data
json_data = {
    'layer': LAYER,
    'summary': {
        'changed_count': changed_count,
        'total_tests': len(results),
        'avg_differences': avg_differences,
        'avg_strength': avg_strength,
        'by_category': {
            'ioi': {'changed': ioi_changed, 'total': 2},
            'question': {'changed': question_changed, 'total': 2},
            'factual': {'changed': factual_changed, 'total': 2},
            'story': {'changed': story_changed, 'total': 2}
        }
    },
    'results': [
        {
            'prompt': r['prompt'],
            'prompt_type': ptype,
            'normal': r['normal_completion'],
            'ablated': r['ablated_completion'],
            'changed': r['changed'],
            'num_differences': len(r['differences']),
            'total_strength': r['total_strength'],
            'ablated_pairs': [
                {
                    'head': h,
                    'feature_i': p['feature_i'],
                    'feature_j': p['feature_j'],
                    'strength': p['abs_value']
                }
                for h, p in r['max_pairs'].items() if p is not None
            ]
        }
        for r, ptype in zip(results, prompt_types)
    ],
    'feature_descriptions': feature_descriptions
}

json_path = output_dir / "diverse_ablation_results.json"
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"\n✅ Dashboard saved to: {html_path}")
print(f"✅ JSON data saved to: {json_path}")

print("\n" + "="*60)
print("Dashboard creation complete!")
print("="*60)