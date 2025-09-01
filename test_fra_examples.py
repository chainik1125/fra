"""
Test FRA on example sentences to find interesting feature interactions.
Using the optimized implementation.
"""

import torch
import numpy as np
from pathlib import Path
from fra.utils import load_config, load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis_fast import get_sentence_fra_sparse_fast
from fra.fra_analysis import get_top_feature_interactions
from fra.log import logger


def analyze_text(model, sae, text, layer, head):
    """Analyze a text and print results."""
    print(f"\n{'='*70}")
    print(f"Text: '{text}'")
    print(f"Head: {head}")
    print(f"{'='*70}")
    
    # Compute FRA
    fra_result = get_sentence_fra_sparse_fast(
        model, sae, text, layer, head,
        max_length=50, verbose=False
    )
    
    # Get top interactions
    top_int = get_top_feature_interactions(
        fra_result['data_dep_int_matrix'], top_k=10
    )
    
    top_loc = get_top_feature_interactions(
        fra_result['data_dep_localization_matrix'], top_k=10
    )
    
    print(f"Sequence length: {fra_result['seq_len']} tokens")
    print(f"Non-zero interactions: {fra_result['nnz']:,}")
    print(f"Density: {fra_result['density']:.8%}")
    
    print(f"\nTop feature interactions:")
    for i, (q, k, v) in enumerate(top_int[:5]):
        print(f"  {i+1}. F{q:5d} → F{k:5d}: {v:8.4f}")
    
    print(f"\nMost localized interactions:")
    for i, (q, k, v) in enumerate(top_loc[:5]):
        print(f"  {i+1}. F{q:5d} → F{k:5d}: dist={v:5.2f}")
    
    return fra_result


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Load model and SAE
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = load_model(config["model"]["name"], str(device))
    sae_data = load_sae(config["sae"]["repo"], config["sae"]["layer"], str(device))
    sae = SimpleAttentionSAE(sae_data)
    
    print(f"\n🔬 Feature-Resolved Attention Analysis")
    print(f"Model: {config['model']['name']}, Layer: {sae.layer}")
    print(f"SAE: d_sae={sae.d_sae} (dict_mult=64)")
    
    # Test sentences designed to show different attention patterns
    test_cases = [
        # Induction pattern - repeated token
        ("The cat sat. The cat ran.", [5, 10, 11]),  # Heads often showing induction
        
        # Repeated name pattern
        ("John went home. John was tired.", [5, 10, 11]),
        
        # Code-like pattern
        ("x = 1; y = 2; z = 3;", [0, 1, 7]),
        
        # List completion pattern
        ("A, B, C, D, E, F,", [0, 4, 7]),
        
        # Subject-verb agreement
        ("The dogs run. The dog runs.", [2, 9, 10]),
    ]
    
    results = []
    for text, heads in test_cases:
        for head in heads:
            try:
                result = analyze_text(model, sae, text, sae.layer, head)
                results.append({
                    'text': text,
                    'head': head,
                    'nnz': result['nnz'],
                    'top_interaction': get_top_feature_interactions(
                        result['data_dep_int_matrix'], top_k=1
                    )[0] if result['nnz'] > 0 else None
                })
            except Exception as e:
                print(f"Error analyzing head {head}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Strongest Feature Interactions Found")
    print(f"{'='*70}")
    
    # Sort by strongest interaction
    strong_results = [r for r in results if r['top_interaction'] is not None]
    strong_results.sort(key=lambda x: abs(x['top_interaction'][2]), reverse=True)
    
    print(f"\nTop 5 strongest feature interactions across all tests:")
    for i, r in enumerate(strong_results[:5]):
        q, k, v = r['top_interaction']
        print(f"{i+1}. Head {r['head']:2d}: F{q:5d} → F{k:5d} = {v:8.4f}")
        print(f"   Text: '{r['text'][:30]}...'")
    
    # Look for patterns
    print(f"\nHeads with most non-zero interactions:")
    head_nnz = {}
    for r in results:
        if r['head'] not in head_nnz:
            head_nnz[r['head']] = []
        head_nnz[r['head']].append(r['nnz'])
    
    for head in sorted(head_nnz.keys()):
        avg_nnz = np.mean(head_nnz[head])
        print(f"  Head {head:2d}: avg {avg_nnz:7.1f} non-zeros")


if __name__ == "__main__":
    import os
    import sys
    
    try:
        main()
        print(f"\n✅ Analysis complete!")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)