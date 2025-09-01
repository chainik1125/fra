#!/usr/bin/env python
"""Test if specific feature channels (FRA[:,:,i,j]) act as induction heads."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
from tqdm import tqdm
import json

torch.set_grad_enabled(False)

print("="*60)
print("Testing Feature Channels for Induction Behavior")
print("="*60)

# Configuration
LAYER = 5
HEAD = 0

# Load model and SAE
print("\nLoading model and SAE...")
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', f'blocks.{LAYER}.hook_z', device='cuda')
print(f"Model and SAE loaded for Layer {LAYER}")

def create_induction_prompt(seq_len=50, vocab_size=1000):
    """Create a sequence with repeated patterns to test induction."""
    first_half = torch.randint(50, vocab_size, (seq_len//2,))
    repeated_seq = torch.cat([first_half, first_half])
    return repeated_seq

def get_feature_channel_induction_score(fra_tensor_sparse, feature_i, feature_j, verbose=False):
    """
    Calculate induction score for a specific feature channel FRA[:,:,i,j].
    
    Args:
        fra_tensor_sparse: Sparse 4D FRA tensor [seq_len, seq_len, d_sae, d_sae]
        feature_i: Query feature index
        feature_j: Key feature index
        
    Returns:
        Induction score for this feature channel
    """
    # Extract the specific feature channel as a dense 2D attention pattern
    # This is FRA[:, :, feature_i, feature_j] - a seq_len x seq_len matrix
    
    indices = fra_tensor_sparse.indices()  # [4, nnz]
    values = fra_tensor_sparse.values()    # [nnz]
    
    # Filter for specific feature pair
    mask = (indices[2] == feature_i) & (indices[3] == feature_j)
    
    if not mask.any():
        return 0.0, np.zeros((fra_tensor_sparse.shape[0], fra_tensor_sparse.shape[0]))  # No interactions for this feature pair
    
    # Get positions and values for this feature channel
    query_positions = indices[0, mask].cpu().numpy()
    key_positions = indices[1, mask].cpu().numpy()
    attention_values = values[mask].cpu().numpy()
    
    # Build attention pattern for this feature channel
    seq_len = fra_tensor_sparse.shape[0]
    attention_pattern = np.zeros((seq_len, seq_len))
    
    for q_pos, k_pos, val in zip(query_positions, key_positions, attention_values):
        attention_pattern[q_pos, k_pos] = val
    
    # Calculate induction score
    half_len = seq_len // 2
    induction_scores = []
    
    for i in range(half_len, seq_len):
        # Position i should attend to position (i - half_len) for induction
        expected_pos = i - half_len
        attention_score = attention_pattern[i, expected_pos]
        induction_scores.append(attention_score)
    
    avg_induction_score = np.mean(induction_scores) if induction_scores else 0.0
    
    if verbose:
        print(f"  Feature channel ({feature_i}, {feature_j}): {avg_induction_score:.4f}")
    
    return avg_induction_score, attention_pattern

# Generate test sequences with repetition
print("\nGenerating test sequences with repetition...")
num_prompts = 5
seq_len = 100  # 50 tokens repeated twice
test_texts = []

for i in range(num_prompts):
    # Create repeated token sequence
    prompt = create_induction_prompt(seq_len=seq_len)
    # Decode to text
    text = model.tokenizer.decode(prompt.tolist())
    test_texts.append(text)
    if i == 0:
        print(f"Example sequence (first 100 chars): {text[:100]}...")

# Process each sequence and accumulate FRA tensors
print(f"\nProcessing {num_prompts} sequences...")
all_fra_results = []

for i, text in enumerate(tqdm(test_texts, desc="Computing FRA")):
    fra_result = get_sentence_fra_batch(
        model, sae, text, 
        layer=LAYER, head=HEAD,
        max_length=seq_len, 
        top_k=30,  # Keep more features for better coverage
        verbose=False
    )
    all_fra_results.append(fra_result)

# Load top feature pairs from previous analysis
print("\nLoading top feature pairs from previous analysis...")
from pathlib import Path
json_path = Path("/root/fra/results/top_pairs.json")
if json_path.exists():
    with open(json_path, 'r') as f:
        data = json.load(f)
    top_pairs = data['top_pairs'][:50]  # Test top 50 pairs
else:
    print("No previous results found. Testing self-interactions of random features...")
    # Generate some random feature pairs to test
    top_pairs = []
    for i in range(20):
        feat_id = np.random.randint(0, 49152)
        top_pairs.append({
            'query_feature': feat_id,
            'key_feature': feat_id,  # Self-interaction
            'average': 0
        })

# Test each feature channel for induction behavior
print(f"\nTesting {len(top_pairs)} feature channels for induction...")
print("-"*40)

feature_channel_scores = []

for pair in tqdm(top_pairs, desc="Testing channels"):
    feature_i = pair['query_feature']
    feature_j = pair['key_feature']
    
    # Calculate induction score across all test sequences
    scores = []
    for fra_result in all_fra_results:
        if fra_result and 'fra_tensor_sparse' in fra_result:
            score, _ = get_feature_channel_induction_score(
                fra_result['fra_tensor_sparse'],
                feature_i, feature_j,
                verbose=False
            )
            scores.append(score)
    
    avg_score = np.mean(scores) if scores else 0
    
    feature_channel_scores.append({
        'feature_i': feature_i,
        'feature_j': feature_j,
        'induction_score': avg_score,
        'is_self': feature_i == feature_j,
        'fra_average': pair.get('average', 0)
    })

# Sort by induction score
feature_channel_scores.sort(key=lambda x: x['induction_score'], reverse=True)

# Display results
print("\n" + "="*60)
print("RESULTS: Top Feature Channels by Induction Score")
print("="*60)

print("\nTop 20 Induction Feature Channels:")
print("-"*40)
print(f"{'Rank':<5} {'Feature Pair':<20} {'Induction Score':<15} {'Type':<10}")
print("-"*40)

for i, channel in enumerate(feature_channel_scores[:20]):
    pair_str = f"F{channel['feature_i']} → F{channel['feature_j']}"
    type_str = "SELF" if channel['is_self'] else "CROSS"
    print(f"{i+1:<5} {pair_str:<20} {channel['induction_score']:<15.4f} {type_str:<10}")

# Analyze self vs cross interactions
self_channels = [c for c in feature_channel_scores if c['is_self']]
cross_channels = [c for c in feature_channel_scores if not c['is_self']]

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if self_channels:
    self_scores = [c['induction_score'] for c in self_channels]
    print(f"\nSelf-interaction channels (diagonal):")
    print(f"  Count: {len(self_channels)}")
    print(f"  Mean induction score: {np.mean(self_scores):.4f}")
    print(f"  Max induction score: {np.max(self_scores):.4f}")
    print(f"  % with score > 0.01: {sum(s > 0.01 for s in self_scores) / len(self_scores) * 100:.1f}%")

if cross_channels:
    cross_scores = [c['induction_score'] for c in cross_channels]
    print(f"\nCross-interaction channels:")
    print(f"  Count: {len(cross_channels)}")
    print(f"  Mean induction score: {np.mean(cross_scores):.4f}")
    print(f"  Max induction score: {np.max(cross_scores):.4f}")
    print(f"  % with score > 0.01: {sum(s > 0.01 for s in cross_scores) / len(cross_scores) * 100:.1f}%")

# Compare with token-level induction
print("\n" + "="*60)
print("COMPARISON WITH TOKEN-LEVEL")
print("="*60)
print(f"\nToken-level induction score for L{LAYER}H{HEAD}: 0.0216")
print(f"Best feature channel induction score: {feature_channel_scores[0]['induction_score']:.4f}")

if feature_channel_scores[0]['induction_score'] > 0.0216:
    print("✅ Found feature channels with STRONGER induction than token-level!")
else:
    print("❌ Feature channels show weaker induction than token-level")

# Visualize best induction channel
if feature_channel_scores[0]['induction_score'] > 0.01:
    print("\n" + "="*60)
    print("VISUALIZING BEST FEATURE CHANNEL")
    print("="*60)
    
    best_channel = feature_channel_scores[0]
    print(f"\nBest channel: F{best_channel['feature_i']} → F{best_channel['feature_j']}")
    print(f"Induction score: {best_channel['induction_score']:.4f}")
    
    # Get attention pattern for visualization
    if all_fra_results[0]:
        score, pattern = get_feature_channel_induction_score(
            all_fra_results[0]['fra_tensor_sparse'],
            best_channel['feature_i'],
            best_channel['feature_j'],
            verbose=False
        )
        
        # Calculate diagonal strength
        half_len = seq_len // 2
        diagonal_vals = [pattern[i + half_len, i] for i in range(half_len)]
        
        print(f"\nDiagonal statistics (induction positions):")
        print(f"  Mean: {np.mean(diagonal_vals):.4f}")
        print(f"  Max: {np.max(diagonal_vals):.4f}")
        print(f"  % non-zero: {sum(v > 0 for v in diagonal_vals) / len(diagonal_vals) * 100:.1f}%")

# Save results
results_path = "/root/fra/results/feature_channel_induction_results.json"
with open(results_path, 'w') as f:
    json.dump({
        'layer': LAYER,
        'head': HEAD,
        'num_sequences': num_prompts,
        'feature_channels_tested': len(feature_channel_scores),
        'top_channels': feature_channel_scores[:50],
        'summary': {
            'best_score': feature_channel_scores[0]['induction_score'] if feature_channel_scores else 0,
            'self_mean': np.mean(self_scores) if self_channels else 0,
            'cross_mean': np.mean(cross_scores) if cross_channels else 0,
            'token_level_score': 0.0216
        }
    }, f, indent=2)

print(f"\nResults saved to: {results_path}")
print("\n" + "="*60)
print("Analysis complete!")
print("="*60)