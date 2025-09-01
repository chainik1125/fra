#!/usr/bin/env python
"""Compare ablating ALL non-self pairs vs random baseline for a single sample."""

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
print("Single Sample: ALL Non-Self vs Random Ablation")
print("="*60)

LAYER = 5
DEVICE = 'cuda'
N_HEADS = 12

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device=DEVICE)

def find_all_nonself_and_random_pairs(model, sae, text, layer):
    """Find ALL non-self pairs and select random pairs for comparison."""
    all_nonself_pairs = {}
    random_pairs = {}
    
    for head in range(N_HEADS):
        fra_result = get_sentence_fra_batch(
            model, sae, text, 
            layer=layer, head=head,
            max_length=128, top_k=30,
            verbose=False
        )
        
        if fra_result is None or fra_result['total_interactions'] == 0:
            all_nonself_pairs[head] = None
            random_pairs[head] = None
            continue
        
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        
        if len(values) == 0:
            all_nonself_pairs[head] = None
            random_pairs[head] = None
            continue
        
        # Find ALL non-self interactions
        q_feats = indices[2, :]
        k_feats = indices[3, :]
        mask = q_feats != k_feats
        non_self_indices = torch.where(mask)[0]
        
        if not mask.any():
            all_nonself_pairs[head] = None
            random_pairs[head] = None
            continue
        
        # Store all non-self indices for this head
        all_nonself_pairs[head] = {
            'indices_to_remove': non_self_indices.tolist(),
            'fra_sparse': fra_sparse,
            'seq_len': fra_result['seq_len'],
            'num_removed': len(non_self_indices),
            'total_strength': values[mask].abs().sum().item()
        }
        
        # For random baseline: select same NUMBER of random indices to remove
        num_to_remove = len(non_self_indices)
        all_indices = list(range(len(values)))
        random_indices_to_remove = random.sample(all_indices, min(num_to_remove, len(all_indices)))
        
        random_pairs[head] = {
            'indices_to_remove': random_indices_to_remove,
            'fra_sparse': fra_sparse,
            'seq_len': fra_result['seq_len'],
            'num_removed': len(random_indices_to_remove),
            'total_strength': values[random_indices_to_remove].abs().sum().item() if random_indices_to_remove else 0
        }
    
    return all_nonself_pairs, random_pairs

def create_ablated_patterns(pairs_info, remove_all=False):
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

# Find ALL non-self pairs and random pairs
all_nonself_info, random_info = find_all_nonself_and_random_pairs(model, sae, prompt, LAYER)

# Calculate statistics
total_nonself_removed = sum(info['num_removed'] for info in all_nonself_info.values() if info)
total_nonself_strength = sum(info['total_strength'] for info in all_nonself_info.values() if info)
total_random_removed = sum(info['num_removed'] for info in random_info.values() if info)
total_random_strength = sum(info['total_strength'] for info in random_info.values() if info)

print(f"\nAblation Statistics:")
print(f"  ALL non-self: {total_nonself_removed} pairs, total strength {total_nonself_strength:.3f}")
print(f"  Random:       {total_random_removed} pairs, total strength {total_random_strength:.3f}")

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

# Random ablation
random_patterns = create_ablated_patterns(random_info)

random_tokens = []
current = input_ids
for _ in range(15):
    def hook_fn(attn_scores, hook):
        for head, pattern in random_patterns.items():
            if pattern is not None:
                seq_len = pattern.shape[0]
                attn_scores[:, head, :seq_len, :seq_len] = pattern.unsqueeze(0)
        return attn_scores
    
    with model.hooks([(f"blocks.{LAYER}.attn.hook_attn_scores", hook_fn)]):
        logits = model(current)
    
    next_token = torch.argmax(logits[0, -1, :]).item()
    random_tokens.append(next_token)
    current = torch.cat([current, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

random_text = model.tokenizer.decode(random_tokens)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n1. NORMAL (No ablation):")
print(f"   '{normal_text}'")

print(f"\n2. ALL NON-SELF ABLATED ({total_nonself_removed} pairs):")
print(f"   '{all_nonself_text}'")

print(f"\n3. RANDOM ABLATED ({total_random_removed} pairs):")
print(f"   '{random_text}'")

# Count differences
nonself_diffs = sum(1 for n, a in zip(normal_tokens, all_nonself_tokens) if n != a)
random_diffs = sum(1 for n, r in zip(normal_tokens, random_tokens) if n != r)

print("\n" + "-"*60)
print("Token differences from normal:")
print(f"  ALL non-self: {nonself_diffs}/15 tokens changed")
print(f"  Random:       {random_diffs}/15 tokens changed")

if nonself_diffs > random_diffs:
    print(f"\n✅ Non-self ablation is {nonself_diffs/max(random_diffs, 0.1):.1f}x more effective than random!")
elif nonself_diffs == random_diffs:
    print(f"\n⚠️ Both ablations have the same effect ({nonself_diffs} changes)")
else:
    print(f"\n❌ Random ablation more effective? ({random_diffs} > {nonself_diffs})")

# Save detailed results
print("\n" + "-"*60)
print("Detailed head-by-head breakdown:")
for head in range(N_HEADS):
    if all_nonself_info[head]:
        print(f"  Head {head:2}: Removed {all_nonself_info[head]['num_removed']:4} non-self pairs "
              f"(strength: {all_nonself_info[head]['total_strength']:.3f})")

# Create simple HTML visualization
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Single Sample Ablation Comparison</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        .prompt {{ background: #f0f0f0; padding: 15px; margin: 20px 0; font-family: monospace; }}
        .results {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
        .result {{ padding: 15px; border-radius: 8px; }}
        .normal {{ background: #e8f5e9; }}
        .nonself {{ background: #ffebee; }}
        .random {{ background: #e3f2fd; }}
        .text {{ font-family: monospace; margin-top: 10px; font-size: 14px; }}
        .stats {{ background: white; padding: 15px; margin: 20px 0; border: 1px solid #ddd; }}
        h2 {{ color: #333; }}
        .metric {{ margin: 5px 0; }}
        .highlight {{ background: yellow; padding: 2px 4px; }}
    </style>
</head>
<body>
    <h1>Single Sample: ALL Non-Self vs Random Ablation</h1>
    
    <div class="prompt">
        <strong>Prompt:</strong> {prompt}
    </div>
    
    <div class="stats">
        <h3>Ablation Statistics</h3>
        <div class="metric">ALL non-self: <strong>{total_nonself_removed}</strong> pairs removed, total strength <strong>{total_nonself_strength:.3f}</strong></div>
        <div class="metric">Random baseline: <strong>{total_random_removed}</strong> pairs removed, total strength <strong>{total_random_strength:.3f}</strong></div>
    </div>
    
    <div class="results">
        <div class="result normal">
            <h3>Normal (No ablation)</h3>
            <div class="text">{normal_text}</div>
            <div style="margin-top: 10px; font-size: 12px;">Baseline completion</div>
        </div>
        
        <div class="result nonself">
            <h3>ALL Non-Self Ablated</h3>
            <div class="text">{all_nonself_text}</div>
            <div style="margin-top: 10px; font-size: 12px;">
                <span class="highlight">{nonself_diffs}/15 tokens changed</span>
            </div>
        </div>
        
        <div class="result random">
            <h3>Random Ablated</h3>
            <div class="text">{random_text}</div>
            <div style="margin-top: 10px; font-size: 12px;">
                <span class="highlight">{random_diffs}/15 tokens changed</span>
            </div>
        </div>
    </div>
    
    <div class="stats">
        <h3>Effectiveness</h3>
        <div class="metric">
            Non-self ablation effectiveness: <strong>{nonself_diffs/max(random_diffs, 0.1):.1f}x</strong> compared to random
        </div>
    </div>
</body>
</html>
"""

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
html_path = output_dir / "single_sample_ablation.html"
with open(html_path, 'w') as f:
    f.write(html)

print(f"\n✅ Results saved to: {html_path}")
print("="*60)