"""
Simple test of Feature-Resolved Attention on a few sentences.
"""

import torch
import numpy as np
from pathlib import Path

from fra.utils import load_config, load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis import get_sentence_fra_sparse, get_top_feature_interactions
from fra.log import logger


def main():
    """
    Test FRA on a simple example.
    """
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Load model and SAE
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = load_model(config["model"]["name"], str(device))
    sae_data = load_sae(config["sae"]["repo"], config["sae"]["layer"], str(device))
    sae = SimpleAttentionSAE(sae_data)
    
    print(f"\n🔬 Testing Feature-Resolved Attention")
    print(f"Model: {config['model']['name']}")
    print(f"SAE: Layer {sae.layer}, d_sae={sae.d_sae}")
    
    # Test sentences with clear patterns
    test_sentences = [
        # Simple repetition (should show induction)
        "The cat sat on the mat. The cat was happy.",
        
        # Name pattern (repeated context)
        "John Smith went to the store. John Smith bought milk.",
        
        # List pattern
        "Apple is red. Banana is yellow. Orange is orange."
    ]
    
    # Test on head 10 (often an induction head in GPT-2 layer 5)
    head = 10
    
    print(f"\n🎯 Testing Head {head} in Layer {sae.layer}")
    
    for text in test_sentences:
        print(f"\n{'='*80}")
        print(f"Text: '{text}'")
        print(f"{'='*80}")
        
        # Compute sparse FRA
        fra_result = get_sentence_fra_sparse(
            model, sae, text, 
            layer=sae.layer, 
            head=head, 
            max_length=128, 
            verbose=True
        )
        
        # Get top feature interactions
        top_interactions = get_top_feature_interactions(
            fra_result['data_dep_int_matrix'], 
            top_k=15
        )
        
        # Get top localized interactions
        top_localized = get_top_feature_interactions(
            fra_result['data_dep_localization_matrix'],
            top_k=15
        )
        
        print(f"\n📊 Statistics:")
        print(f"  • Sequence length: {fra_result['seq_len']} tokens")
        print(f"  • Position pairs analyzed: {fra_result['total_pairs']}")
        print(f"  • Non-zero feature interactions: {fra_result['nnz']:,}")
        print(f"  • Feature interaction density: {fra_result['density']:.6%}")
        print(f"  • Memory saved vs dense: {(1 - fra_result['density']) * 100:.2f}%")
        
        print(f"\n🔝 Top Feature Interactions (by average strength):")
        for i, (q_feat, k_feat, value) in enumerate(top_interactions[:10]):
            print(f"  {i+1:2d}. Feature {q_feat:5d} → Feature {k_feat:5d}: {value:10.6f}")
        
        print(f"\n📍 Top Localized Interactions (distance-weighted):")
        for i, (q_feat, k_feat, value) in enumerate(top_localized[:10]):
            print(f"  {i+1:2d}. Feature {q_feat:5d} → Feature {k_feat:5d}: avg_dist={value:6.2f}")
        
        # Look for repeated feature interactions (potential induction patterns)
        from collections import Counter
        feature_pairs = [(q, k) for q, k, _ in top_interactions]
        
        # Check if any feature pairs appear multiple times
        print(f"\n🔄 Checking for repeated feature patterns:")
        interaction_dict = {}
        for q_feat, k_feat, value in top_interactions:
            pair = (q_feat, k_feat)
            if pair not in interaction_dict:
                interaction_dict[pair] = []
            interaction_dict[pair].append(value)
        
        # Find features that interact strongly
        strong_features = set()
        for q_feat, k_feat, value in top_interactions:
            if abs(value) > 0.01:  # Threshold for "strong"
                strong_features.add(q_feat)
                strong_features.add(k_feat)
        
        print(f"  • Found {len(strong_features)} features with strong interactions")
        print(f"  • Feature IDs: {sorted(list(strong_features))[:10]}...")
        
        # Quick check: Do repeated words activate similar features?
        tokens = model.tokenizer.encode(text)
        print(f"\n🔤 Token analysis:")
        print(f"  • Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        
        # Find repeated tokens
        from collections import Counter
        token_counts = Counter(tokens)
        repeated_tokens = {t: c for t, c in token_counts.items() if c > 1}
        if repeated_tokens:
            print(f"  • Repeated tokens: {repeated_tokens}")
            for token_id, count in repeated_tokens.items():
                token_str = model.tokenizer.decode([token_id])
                print(f"    - '{token_str}' appears {count} times")


if __name__ == "__main__":
    import os
    import sys
    
    try:
        main()
        print(f"\n✅ Analysis complete!")
    finally:
        # Clean up CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)