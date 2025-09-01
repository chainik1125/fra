#!/usr/bin/env python
"""Compare ablating ALL non-self pairs vs ONE random pair for a single sample."""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import random
from pathlib import Path

torch.set_grad_enabled(False)

print("="*60)
print("Single Sample: ALL Non-Self vs ONE Random Pair")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def find_all_nonself_and_one_random(model, sae, text, layer):
    """Find ALL non-self pairs and select ONE random pair for comparison."""
    all_nonself_pairs = {}
    one_random_pair = {}
    
    # Collect all available indices across all heads for random selection
    all_available_indices = []
    
    for head in range(N_HEADS):
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=30,
            verbose=False
        )
        
        if fra_result is None or fra_result['total_interactions'] == 0:
            all_nonself_pairs[head] = None
            one_random_pair[head] = None
            continue
        
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        
        if len(values) == 0:
            all_nonself_pairs[head] = None
            one_random_pair[head] = None
            continue
        
        # Find ALL non-self interactions
        q_feats = indices[2, :]
        k_feats = indices[3, :]
        mask = q_feats != k_feats
        non_self_indices = torch.where(mask)[0]
        
        if not mask.any():
            all_nonself_pairs[head] = None
            one_random_pair[head] = None
            continue
        
        # Store all non-self indices for this head
        all_nonself_pairs[head] = {
            'indices_to_remove': non_self_indices.tolist(),
            'fra_sparse': fra_sparse,
            'seq_len': fra_result['seq_len'],
            'num_removed': len(non_self_indices),
            'total_strength': values[mask].abs().sum().item()
        }
        
        # Collect indices for random selection (head, index_in_sparse, value)
        for idx in range(len(values)):
            all_available_indices.append((head, idx, values[idx].abs().item()))
        
        # Initialize empty for one_random_pair (will fill later)
        one_random_pair[head] = {
            'indices_to_remove': [],
            'fra_sparse': fra_sparse,
            'seq_len': fra_result['seq_len'],
            'num_removed': 0,
            'total_strength': 0
        }
    
    # Now select ONE random pair from all available pairs across all heads
    if all_available_indices:
        selected_head, selected_idx, selected_value = random.choice(all_available_indices)
        
        # Update the one_random_pair for the selected head
        one_random_pair[selected_head]['indices_to_remove'] = [selected_idx]
        one_random_pair[selected_head]['num_removed'] = 1
        one_random_pair[selected_head]['total_strength'] = selected_value
        
        # Get details about the selected pair
        fra_sparse = one_random_pair[selected_head]['fra_sparse']
        indices = fra_sparse.indices()
        selected_details = {
            'head': selected_head,
            'position': (indices[0, selected_idx].item(), indices[1, selected_idx].item()),
            'features': (indices[2, selected_idx].item(), indices[3, selected_idx].item()),
            'value': selected_value
        }
    else:
        selected_details = None
    
    return all_nonself_pairs, one_random_pair, selected_details

def create_ablated_patterns(pairs_info):
    """Create ablated patterns by removing specified pairs."""
    ablated_patterns = {}
    
    for head, info in pairs_info.items():
        if info is None:
            ablated_patterns[head] = None
            continue
        
        fra_sparse = info['fra_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        seq_len = info['seq_len']
        
        ablated = torch.zeros((seq_len, seq_len), device=DEVICE)
        
        # Add all interactions except those to remove
        indices_to_remove = set(info['indices_to_remove'])
        
        for i in range(indices.shape[1]):
            if i not in indices_to_remove:
                q_pos = indices[0, i].item()
                k_pos = indices[1, i].item()
                ablated[q_pos, k_pos] += values[i].item()
        
        ablated_patterns[head] = ablated
    
    return ablated_patterns

# Test on a single prompt
prompt = "When John and Mary went to the store, John gave a drink to"

print(f"\nPrompt: '{prompt}'")
print("-"*60)

# Find ALL non-self pairs and ONE random pair
all_nonself_info, one_random_info, random_details = find_all_nonself_and_one_random(model, sae, prompt, LAYER)

# Calculate statistics
total_nonself_removed = sum(info['num_removed'] for info in all_nonself_info.values() if info)
total_nonself_strength = sum(info['total_strength'] for info in all_nonself_info.values() if info)

print(f"\nAblation Statistics:")
print(f"  ALL non-self: {total_nonself_removed} pairs, total strength {total_nonself_strength:.3f}")
if random_details:
    print(f"  ONE random:   1 pair from Head {random_details['head']}, "
          f"F{random_details['features'][0]}→F{random_details['features'][1]}, "
          f"strength {random_details['value']:.4f}")

# Get normal completion
tokens = model.tokenizer.encode(prompt)
if len(tokens) > 128:
    tokens = tokens[:128]
input_ids = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

print("\nGenerating completions...")

# Normal generation
normal_tokens = []
current = input_ids
for _ in range(15):
    with torch.no_grad():
        logits = model(current)
    next_token = torch.argmax(logits[0, -1, :]).item()
    normal_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

normal_text = model.tokenizer.decode(normal_tokens)

# ALL non-self ablation
all_nonself_patterns = create_ablated_patterns(all_nonself_info)

all_nonself_tokens = []
current = input_ids
for _ in range(15):
    def hook_fn(attn_scores, hook):
        for head, pattern in all_nonself_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", hook_fn)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    all_nonself_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

all_nonself_text = model.tokenizer.decode(all_nonself_tokens)

# ONE random pair ablation
one_random_patterns = create_ablated_patterns(one_random_info)

one_random_tokens = []
current = input_ids
for _ in range(15):
    def hook_fn(attn_scores, hook):
        for head, pattern in one_random_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", hook_fn)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    one_random_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

one_random_text = model.tokenizer.decode(one_random_tokens)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n1. NORMAL (No ablation):")
print(f"   '{normal_text}'")

print(f"\n2. ALL NON-SELF ABLATED ({total_nonself_removed} pairs):")
print(f"   '{all_nonself_text}'")

print(f"\n3. ONE RANDOM PAIR ABLATED:")
print(f"   '{one_random_text}'")
if random_details:
    print(f"   (Ablated: Head {random_details['head']}, pos {random_details['position']}, "
          f"F{random_details['features'][0]}→F{random_details['features'][1]})")

# Count differences
nonself_diffs = sum(1 for n, a in zip(normal_tokens, all_nonself_tokens) if n != a)
random_diffs = sum(1 for n, r in zip(normal_tokens, one_random_tokens) if n != r)

print("\n" + "-"*60)
print("Token differences from normal:")
print(f"  ALL non-self: {nonself_diffs}/15 tokens changed")
print(f"  ONE random:   {random_diffs}/15 tokens changed")

if nonself_diffs > random_diffs:
    effectiveness = nonself_diffs / max(random_diffs, 0.1)
    print(f"\n✅ Ablating ALL non-self pairs is {effectiveness:.1f}x more disruptive than ONE random pair!")
elif nonself_diffs == random_diffs and nonself_diffs > 0:
    print(f"\n⚠️ Both ablations changed {nonself_diffs} tokens (unexpected)")
elif random_diffs > nonself_diffs:
    print(f"\n❌ ONE random pair more disruptive than ALL non-self? ({random_diffs} > {nonself_diffs})")
else:
    print(f"\n❌ Neither ablation had any effect")

# Save detailed results
print("\n" + "-"*60)
print("Summary:")
print(f"  Ablating {total_nonself_removed:,} non-self pairs → {nonself_diffs} token changes")
print(f"  Ablating 1 random pair → {random_diffs} token changes")
print(f"  Effectiveness ratio: {nonself_diffs / max(random_diffs, 0.1):.1f}x")

# Create simple HTML visualization
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ablation Comparison: ALL Non-Self vs ONE Random</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .prompt {{ background: #2c3e50; color: white; padding: 20px; margin: 20px 0; font-family: monospace; border-radius: 8px; }}
        .results {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 30px 0; }}
        .result {{ padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .normal {{ background: #e8f5e9; border: 2px solid #4caf50; }}
        .all-nonself {{ background: #ffebee; border: 2px solid #f44336; }}
        .one-random {{ background: #e3f2fd; border: 2px solid #2196f3; }}
        .text {{ font-family: monospace; margin: 15px 0; font-size: 14px; line-height: 1.5; }}
        .stats {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h2 {{ margin-top: 0; }}
        .metric {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }}
        .highlight {{ background: #ffd54f; padding: 2px 6px; border-radius: 3px; font-weight: bold; }}
        .big-number {{ font-size: 24px; font-weight: bold; color: #f44336; }}
        .small-number {{ font-size: 24px; font-weight: bold; color: #2196f3; }}
    </style>
</head>
<body>
    <h1>🔬 Ablation Comparison: ALL Non-Self vs ONE Random Pair</h1>
    
    <div class="prompt">
        <strong>Prompt:</strong> {prompt}
    </div>
    
    <div class="stats">
        <h2>Ablation Scale</h2>
        <div class="metric">
            ALL non-self pairs: <span class="big-number">{total_nonself_removed:,}</span> pairs removed
            (strength: {total_nonself_strength:.1f})
        </div>
        <div class="metric">
            ONE random pair: <span class="small-number">1</span> pair removed
            {f"(Head {random_details['head']}, F{random_details['features'][0]}→F{random_details['features'][1]}, strength: {random_details['value']:.4f})" if random_details else ""}
        </div>
    </div>
    
    <div class="results">
        <div class="result normal">
            <h2>✓ Normal</h2>
            <div style="font-size: 12px; color: #666;">No ablation (baseline)</div>
            <div class="text">{normal_text}</div>
        </div>
        
        <div class="result all-nonself">
            <h2>🔥 ALL Non-Self Ablated</h2>
            <div style="font-size: 12px; color: #666;">{total_nonself_removed:,} pairs removed</div>
            <div class="text">{all_nonself_text}</div>
            <div style="margin-top: 10px;">
                <span class="highlight">{nonself_diffs}/15 tokens changed</span>
            </div>
        </div>
        
        <div class="result one-random">
            <h2>🎲 ONE Random Ablated</h2>
            <div style="font-size: 12px; color: #666;">1 pair removed</div>
            <div class="text">{one_random_text}</div>
            <div style="margin-top: 10px;">
                <span class="highlight">{random_diffs}/15 tokens changed</span>
            </div>
        </div>
    </div>
    
    <div class="stats">
        <h2>Effectiveness Analysis</h2>
        <div class="metric" style="text-align: center; font-size: 18px;">
            Ablating <strong>{total_nonself_removed:,}</strong> non-self pairs is 
            <span class="highlight" style="font-size: 28px;">{nonself_diffs / max(random_diffs, 0.1):.1f}x</span> 
            more disruptive than ablating <strong>1</strong> random pair
        </div>
    </div>
</body>
</html>
"""

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
html_path = output_dir / "correct_ablation_comparison.html"
with open(html_path, 'w') as f:
    f.write(html)

print(f"\n✅ Results saved to: {html_path}")
print("="*60)