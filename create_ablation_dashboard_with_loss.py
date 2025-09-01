#!/usr/bin/env python
"""Create dashboard showing multi-head ablation effects with loss metrics."""

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
print("Creating Multi-Head Ablation Dashboard with Loss Metrics")
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
        
        # Add all except the max pair
        for i in range(indices.shape[1]):
            if i != max_pairs[head].get('max_idx', i):
                q_pos = indices[0, i].item()
                k_pos = indices[1, i].item()
                ablated[q_pos, k_pos] += values[i].item()
        
        ablated_patterns[head] = ablated
    
    return ablated_patterns

def calculate_loss_metrics(model, prompt, layer, ablated_patterns):
    """Calculate loss increase from ablation."""
    
    tokens = model.tokenizer.encode(prompt)
    if len(tokens) > 128:
        tokens = tokens[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    # Get normal logits and loss
    with torch.no_grad():
        normal_logits = model(input_ids)
    
    # Calculate normal loss (using teacher forcing)
    if len(tokens) > 1:
        normal_loss = F.cross_entropy(
            normal_logits[0, :-1, :].reshape(-1, normal_logits.shape[-1]),
            input_ids[0, 1:].reshape(-1),
            reduction='mean'
        ).item()
    else:
        normal_loss = 0.0
    
    # Get ablated logits
    def hook_fn(attn_scores, hook):
        for head, pattern in ablated_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    with torch.no_grad():
        with model.hooks([(f"blocks.{layer}.attn.hook_attn_scores", hook_fn)]):
            ablated_logits = model(input_ids)
    
    # Calculate ablated loss
    if len(tokens) > 1:
        ablated_loss = F.cross_entropy(
            ablated_logits[0, :-1, :].reshape(-1, ablated_logits.shape[-1]),
            input_ids[0, 1:].reshape(-1),
            reduction='mean'
        ).item()
    else:
        ablated_loss = 0.0
    
    # Calculate KL divergence for next token prediction
    normal_probs = F.softmax(normal_logits[0, -1, :], dim=-1)
    ablated_probs = F.softmax(ablated_logits[0, -1, :], dim=-1)
    kl_div = F.kl_div(ablated_probs.log(), normal_probs, reduction='sum').item()
    
    # Calculate perplexity
    normal_perplexity = np.exp(normal_loss) if normal_loss > 0 else 1.0
    ablated_perplexity = np.exp(ablated_loss) if ablated_loss > 0 else 1.0
    
    return {
        'normal_loss': normal_loss,
        'ablated_loss': ablated_loss,
        'loss_increase': ablated_loss - normal_loss,
        'loss_ratio': ablated_loss / normal_loss if normal_loss > 0 else float('inf'),
        'normal_perplexity': normal_perplexity,
        'ablated_perplexity': ablated_perplexity,
        'perplexity_increase': ablated_perplexity - normal_perplexity,
        'kl_divergence': kl_div
    }

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
    """Analyze a single prompt with ablation and loss metrics."""
    
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
    
    # Calculate loss metrics
    loss_metrics = calculate_loss_metrics(model, prompt, layer, ablated_patterns)
    
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
    
    return {
        'prompt': prompt,
        'normal_completion': normal_text,
        'ablated_completion': ablated_text,
        'normal_tokens': normal_tokens,
        'ablated_tokens': ablated_tokens,
        'differences': differences,
        'max_pairs': max_pairs,
        'changed': len(differences) > 0,
        'loss_metrics': loss_metrics
    }

# Test cases
test_prompts = [
    "The cat sat on the mat. The cat",
    "She went to the store and bought milk. She",
    "John gave Mary a book. John gave",
    "The weather today is sunny. The weather",
    "Once upon a time there was a princess. Once upon",
]

print("\nAnalyzing test prompts...")
results = []

for i, prompt in enumerate(test_prompts):
    print(f"  {i+1}/{len(test_prompts)}: {prompt[:30]}...")
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
for feat_id in all_features:
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
    <title>Multi-Head Feature Ablation Dashboard with Loss Metrics</title>
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
        .loss-metrics {
            background: #fff3e0;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 15px;
        }
        .loss-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        .metric {
            background: white;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        .loss-increase { color: #e74c3c; }
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
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
    </style>
</head>
<body>
    <h1>🔬 Multi-Head Feature Ablation Dashboard - Layer """ + str(LAYER) + """</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="stats">
"""

# Calculate summary statistics
changed_count = sum(1 for r in results if r['changed'])
avg_loss_increase = np.mean([r['loss_metrics']['loss_increase'] for r in results])
avg_perplexity_increase = np.mean([r['loss_metrics']['perplexity_increase'] for r in results])
max_loss_increase = max(r['loss_metrics']['loss_increase'] for r in results)
avg_kl = np.mean([r['loss_metrics']['kl_divergence'] for r in results])

html_content += f"""
            <div class="stat-card">
                <div class="stat-value">{changed_count}/{len(results)}</div>
                <div class="stat-label">Outputs Changed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value loss-increase">+{avg_loss_increase:.3f}</div>
                <div class="stat-label">Avg Loss Increase</div>
            </div>
            <div class="stat-card">
                <div class="stat-value loss-increase">+{avg_perplexity_increase:.1f}</div>
                <div class="stat-label">Avg Perplexity Increase</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_kl:.3f}</div>
                <div class="stat-label">Avg KL Divergence</div>
            </div>
        </div>
    </div>
"""

# Add test cases
for i, result in enumerate(results):
    status = "✅ CHANGED" if result['changed'] else "❌ UNCHANGED"
    
    html_content += f"""
    <div class="test-case">
        <div class="test-header">Test {i+1}: {status}</div>
        <div class="prompt">Prompt: {result['prompt']}</div>
        
        <div class="loss-metrics">
            <strong>Loss Metrics:</strong>
            <div class="loss-grid">
                <div class="metric">
                    <div class="metric-label">Normal Loss</div>
                    <div class="metric-value">{result['loss_metrics']['normal_loss']:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Ablated Loss</div>
                    <div class="metric-value loss-increase">{result['loss_metrics']['ablated_loss']:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Loss Increase</div>
                    <div class="metric-value loss-increase">+{result['loss_metrics']['loss_increase']:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Loss Ratio</div>
                    <div class="metric-value">{result['loss_metrics']['loss_ratio']:.2f}x</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Normal Perplexity</div>
                    <div class="metric-value">{result['loss_metrics']['normal_perplexity']:.1f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Ablated Perplexity</div>
                    <div class="metric-value loss-increase">{result['loss_metrics']['ablated_perplexity']:.1f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">KL Divergence</div>
                    <div class="metric-value">{result['loss_metrics']['kl_divergence']:.3f}</div>
                </div>
            </div>
        </div>
        
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
    
    # Add ablated features
    html_content += """
        <div class="features">
            <strong>Ablated Feature Pairs (max non-self per head):</strong>
            <div class="feature-pairs">
"""
    
    for head in range(N_HEADS):
        if result['max_pairs'][head] is not None:
            pair = result['max_pairs'][head]
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
    
    html_content += """
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

html_path = output_dir / "multi_head_ablation_dashboard_with_loss.html"
with open(html_path, 'w') as f:
    f.write(html_content)

# Save JSON data
json_data = {
    'layer': LAYER,
    'summary': {
        'changed_count': changed_count,
        'total_tests': len(results),
        'avg_loss_increase': avg_loss_increase,
        'avg_perplexity_increase': avg_perplexity_increase,
        'max_loss_increase': max_loss_increase,
        'avg_kl_divergence': avg_kl
    },
    'results': [
        {
            'prompt': r['prompt'],
            'normal': r['normal_completion'],
            'ablated': r['ablated_completion'],
            'changed': r['changed'],
            'num_differences': len(r['differences']),
            'loss_metrics': r['loss_metrics'],
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
        for r in results
    ],
    'feature_descriptions': feature_descriptions
}

json_path = output_dir / "ablation_results_with_loss.json"
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"\n✅ Dashboard saved to: {html_path}")
print(f"✅ JSON data saved to: {json_path}")

print("\n" + "="*60)
print("Dashboard creation complete!")
print("="*60)