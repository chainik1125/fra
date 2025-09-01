"""
Test to identify performance bottlenecks in FRA computation.
"""

import torch
import time
from fra.utils import load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations

def main():
    print("Loading model and SAE...")
    t0 = time.time()
    model = load_model('gpt2-small', 'cuda')
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    t0 = time.time()
    sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
    sae = SimpleAttentionSAE(sae_data)
    print(f"SAE loaded in {time.time() - t0:.2f}s")
    
    text = "The cat sat."
    layer = 5
    head = 0
    
    # Time activation extraction
    print(f"\nTiming activation extraction for: '{text}'")
    t0 = time.time()
    activations = get_attention_activations(model, text, layer=layer, max_length=10)
    print(f"  Activations extracted in {time.time() - t0:.3f}s")
    print(f"  Shape: {activations.shape}")
    
    # Time SAE encoding
    t0 = time.time()
    features = sae.encode(activations)
    print(f"  SAE encoding in {time.time() - t0:.3f}s")
    print(f"  Shape: {features.shape}")
    print(f"  Sparsity: {(features == 0).sum().item() / features.numel():.2%}")
    
    # Time single position pair analysis
    print(f"\nTiming single position pair:")
    
    # Get active features for positions 0 and 1
    query_features = features[1]
    key_features = features[0]
    
    query_active = torch.where(query_features != 0)[0]
    key_active = torch.where(key_features != 0)[0]
    
    print(f"  Query active features: {len(query_active)}")
    print(f"  Key active features: {len(key_active)}")
    
    if len(query_active) > 0 and len(key_active) > 0:
        t0 = time.time()
        
        # Get decoder vectors
        query_vecs = sae.W_dec[query_active]
        key_vecs = sae.W_dec[key_active]
        
        # Get attention weights
        W_Q = model.blocks[layer].attn.W_Q[head]
        W_K = model.blocks[layer].attn.W_K[head]
        
        # Compute attention
        q = torch.einsum('da,nd->na', W_Q, query_vecs)
        k = torch.einsum('da,nd->na', W_K, key_vecs)
        int_matrix = torch.einsum('qa,ka->qk', q, k)
        
        print(f"  Position pair computed in {time.time() - t0:.3f}s")
        print(f"  Interaction matrix shape: {int_matrix.shape}")
    
    # Estimate time for full sentence
    seq_len = activations.shape[0]
    n_pairs = seq_len * (seq_len + 1) // 2
    print(f"\nFor full sentence:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of position pairs: {n_pairs}")


if __name__ == "__main__":
    import os
    try:
        main()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)