"""
Check what share of the L1 norm is captured by top-k features.
"""

import torch
from fra.utils import load_model, load_sae, load_dataset_hf
from fra.sae_wrapper import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations

torch.set_grad_enabled(False)

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

# Get a sample from dataset
print("Loading dataset...")
dataset = load_dataset_hf("Elriggs/openwebtext-100k", split="train", streaming=True, seed=42)
sample = next(iter(dataset))
text = sample['text']

# Truncate to 128 tokens
tokens = model.tokenizer.encode(text)[:128]
text = model.tokenizer.decode(tokens)

print(f"\nSample text ({len(tokens)} tokens):")
print(text[:200] + "..." if len(text) > 200 else text)

# Get activations and encode
activations = get_attention_activations(model, text, layer=5, max_length=128)
features = sae.encode(activations)

print(f"\nFeature shape: {features.shape}")
seq_len = features.shape[0]

# Analyze L1 norm share for each position
top_k_values = [10, 20, 50, 100, 200, 500, 1000]
avg_shares = {k: [] for k in top_k_values}

print(f"\nL1 norm share by top-k features:")
print(f"{'Position':<10} " + " ".join([f"Top-{k:<4d}" for k in top_k_values]))
print("-" * (10 + 11 * len(top_k_values)))

for pos in range(min(10, seq_len)):  # First 10 positions
    feat = features[pos]
    
    # Sort by absolute value
    sorted_vals, sorted_idx = torch.sort(feat.abs(), descending=True)
    
    # Calculate L1 norm shares
    total_l1 = sorted_vals.sum().item()
    
    row_str = f"Pos {pos:<5d} "
    for k in top_k_values:
        top_k_l1 = sorted_vals[:k].sum().item()
        share = top_k_l1 / total_l1 if total_l1 > 0 else 0
        avg_shares[k].append(share)
        row_str += f" {share:>8.1%} "
    
    print(row_str)

# Calculate averages across all positions
print(f"\n{'Average':<10} " + " ".join([f"{sum(avg_shares[k])/len(avg_shares[k]):>8.1%} " for k in top_k_values]))

# Now calculate for ALL positions (not just first 10)
print(f"\nCalculating across all {seq_len} positions...")
all_shares = {k: [] for k in top_k_values}

for pos in range(seq_len):
    feat = features[pos]
    sorted_vals, _ = torch.sort(feat.abs(), descending=True)
    total_l1 = sorted_vals.sum().item()
    
    for k in top_k_values:
        top_k_l1 = sorted_vals[:k].sum().item()
        share = top_k_l1 / total_l1 if total_l1 > 0 else 0
        all_shares[k].append(share)

print(f"\nL1 norm captured by top-k features (averaged over {seq_len} positions):")
for k in top_k_values:
    avg = sum(all_shares[k]) / len(all_shares[k])
    print(f"  Top-{k:<4d}: {avg:>6.2%} of total L1 norm")

# Special focus on top-20
print(f"\n=== Top-20 Analysis ===")
top_20_shares = all_shares[20]
print(f"Average L1 share: {sum(top_20_shares)/len(top_20_shares):.2%}")
print(f"Min L1 share: {min(top_20_shares):.2%}")
print(f"Max L1 share: {max(top_20_shares):.2%}")

# Check what fraction of features have >0.01 activation
print(f"\n=== Feature magnitude distribution ===")
thresholds = [0.01, 0.05, 0.1, 0.5, 1.0]
for thresh in thresholds:
    count = (features.abs() > thresh).sum(1).float().mean().item()
    print(f"  Features > {thresh:>4.2f}: {count:>7.1f} ({count/features.shape[1]:>6.2%})")

import os
os._exit(0)