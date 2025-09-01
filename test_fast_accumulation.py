#!/usr/bin/env python
"""Test faster accumulation approach - vectorized operations."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.utils import load_dataset_hf
import time
from tqdm import tqdm

torch.set_grad_enabled(False)

def fast_fra_accumulation(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    texts,
    layer: int = 5,
    head: int = 0,
    max_length: int = 128,
    top_k: int = 10
):
    """Faster FRA accumulation using vectorized operations."""
    
    d_sae = sae.sae.W_dec.shape[0]
    device = next(model.parameters()).device
    
    # Get attention weights once
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    W_dec = sae.sae.W_dec
    
    # Precompute projection matrices for all features
    print("Precomputing feature projections...")
    t0 = time.time()
    # W_dec @ W_Q: [d_sae, d_model] @ [d_model, d_head] = [d_sae, d_head]
    all_q_proj = torch.matmul(W_dec, W_Q)  # [d_sae, d_head]
    all_k_proj = torch.matmul(W_dec, W_K)  # [d_sae, d_head]
    
    # Precompute all pairwise attention scores between features
    # This is data-independent and can be reused
    print(f"Precomputed projections in {time.time()-t0:.1f}s")
    
    # Initialize accumulator
    interaction_dict = {}
    
    for text_idx, text in enumerate(tqdm(texts, desc="Processing samples")):
        # Get tokens
        tokens = model.tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        hook_name = f"blocks.{layer}.attn.hook_z"
        
        # Get activations
        _, cache = model.run_with_cache(tokens_tensor, names_filter=[hook_name])
        z = cache[hook_name].squeeze(0)  # [seq_len, n_heads, d_head]
        seq_len = z.shape[0]
        z_flat = z.flatten(-2, -1)  # [seq_len, 768]
        
        # Encode to SAE features
        feature_acts = sae.sae.encode(z_flat)  # [seq_len, d_sae]
        
        # Get top-k features per position
        topk_vals, topk_idx = torch.topk(feature_acts.abs(), k=top_k, dim=-1)  # [seq_len, top_k]
        
        # Process position pairs
        for key_pos in range(seq_len):
            for query_pos in range(key_pos, seq_len):
                # Get active features
                q_feats = topk_idx[query_pos]  # [top_k]
                k_feats = topk_idx[key_pos]    # [top_k]
                q_vals = feature_acts[query_pos, q_feats]  # [top_k]
                k_vals = feature_acts[key_pos, k_feats]    # [top_k]
                
                # Get projections for these features
                q_proj = all_q_proj[q_feats]  # [top_k, d_head]
                k_proj = all_k_proj[k_feats]  # [top_k, d_head]
                
                # Compute attention scores: [top_k, top_k]
                att_scores = torch.matmul(q_proj, k_proj.T)
                
                # Scale by feature activations
                scaled_scores = att_scores * q_vals.unsqueeze(1) * k_vals.unsqueeze(0)
                
                # Add to accumulator (only significant interactions)
                mask = scaled_scores.abs() > 1e-6
                if mask.any():
                    q_global = q_feats[mask.any(dim=1)]
                    k_global = k_feats[mask.any(dim=0)]
                    
                    for i, q_f in enumerate(q_global):
                        for j, k_f in enumerate(k_global):
                            if mask[i, j]:
                                key = (q_f.item(), k_f.item())
                                val = scaled_scores[i, j].item()
                                
                                if key not in interaction_dict:
                                    interaction_dict[key] = {'sum': 0, 'count': 0}
                                
                                interaction_dict[key]['sum'] += abs(val)
                                interaction_dict[key]['count'] += 1
        
        # Clear cache
        del cache, z, z_flat, feature_acts
        torch.cuda.empty_cache()
    
    # Convert to sorted list
    results = []
    for (q_f, k_f), stats in interaction_dict.items():
        avg = stats['sum'] / stats['count']
        results.append({
            'query_feature': q_f,
            'key_feature': k_f,
            'average': avg,
            'count': stats['count'],
            'sum': stats['sum']
        })
    
    results.sort(key=lambda x: x['average'], reverse=True)
    
    return results


# Main test
print("="*60)
print("Testing FAST FRA accumulation")
print("="*60)

# Load model and SAE
print("\nLoading model...")
t0 = time.time()
model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
print(f"Model loaded in {time.time()-t0:.1f}s")

print("\nLoading SAE...")
t0 = time.time()
sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', 'blocks.5.hook_z', device='cuda')
print(f"SAE loaded in {time.time()-t0:.1f}s")

# Load samples
print("\nLoading 10 samples...")
dataset = load_dataset_hf(streaming=True)
texts = []
for i, item in enumerate(dataset):
    if i >= 10:
        break
    text = item.get('text', '') if isinstance(item, dict) else str(item)
    if len(text.split()) > 10:
        texts.append(text[:500])

print(f"Loaded {len(texts)} texts")

# Run fast accumulation
print("\n" + "="*60)
print("Running FAST accumulation...")
print("="*60)
t0 = time.time()

results = fast_fra_accumulation(
    model, sae, texts,
    layer=5, head=0,
    max_length=128, top_k=10
)

t1 = time.time()
print(f"\n✅ Complete in {t1-t0:.1f}s")
print(f"Per sample: {(t1-t0)/len(texts):.1f}s")

# Show top results
print(f"\nFound {len(results)} unique feature pairs")
print("\nTop 10 interactions:")
for i, r in enumerate(results[:10]):
    print(f"{i+1}. F{r['query_feature']} → F{r['key_feature']}: {r['average']:.4f} (count: {r['count']})")