"""
Test Feature-Resolved Attention on example sentences to find interesting feature interactions.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

from fra.utils import load_config, load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis import get_sentence_fra_sparse, get_top_feature_interactions
from fra.activation_utils import get_attention_pattern
from fra.log import logger


def analyze_sentence(
    model, 
    sae, 
    text: str, 
    layer: int, 
    head: int,
    top_k: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Analyze a sentence with FRA and extract interesting patterns.
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Analyzing: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"{'='*80}")
    
    # Compute sparse FRA
    fra_result = get_sentence_fra_sparse(
        model, sae, text, layer, head, 
        max_length=128, verbose=verbose
    )
    
    # Get top feature interactions
    top_interactions = get_top_feature_interactions(
        fra_result['data_dep_int_matrix'], 
        top_k=top_k
    )
    
    # Get top localized interactions (features that interact at specific distances)
    top_localized = get_top_feature_interactions(
        fra_result['data_dep_localization_matrix'],
        top_k=top_k
    )
    
    # Get standard attention pattern for comparison
    attn_pattern = get_attention_pattern(model, text, layer=layer, head=head)
    
    # Calculate some statistics
    stats = {
        'seq_len': fra_result['seq_len'],
        'total_pairs': fra_result['total_pairs'],
        'nnz': fra_result['nnz'],
        'density': fra_result['density'],
        'memory_saved': (1 - fra_result['density']) * 100,
        'top_interactions': top_interactions,
        'top_localized': top_localized,
        'attn_entropy': -np.sum(attn_pattern * np.log(attn_pattern + 1e-10)),
    }
    
    return stats


def print_analysis_results(text: str, stats: Dict, sae: SimpleAttentionSAE):
    """
    Pretty print the analysis results.
    """
    print(f"\n📊 Statistics:")
    print(f"  • Sequence length: {stats['seq_len']} tokens")
    print(f"  • Position pairs analyzed: {stats['total_pairs']}")
    print(f"  • Non-zero feature interactions: {stats['nnz']:,}")
    print(f"  • Feature interaction density: {stats['density']:.6%}")
    print(f"  • Memory saved vs dense: {stats['memory_saved']:.2f}%")
    print(f"  • Token attention entropy: {stats['attn_entropy']:.2f}")
    
    print(f"\n🔝 Top Feature Interactions (by average strength):")
    for i, (q_feat, k_feat, value) in enumerate(stats['top_interactions'][:10]):
        print(f"  {i+1:2d}. Feature {q_feat:5d} → Feature {k_feat:5d}: {value:8.4f}")
    
    print(f"\n📍 Top Localized Interactions (distance-weighted):")
    for i, (q_feat, k_feat, value) in enumerate(stats['top_localized'][:10]):
        print(f"  {i+1:2d}. Feature {q_feat:5d} → Feature {k_feat:5d}: {value:8.4f}")


def main():
    """
    Test FRA on various example sentences.
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
    
    # Test sentences with different patterns
    test_sentences = [
        # Simple repetition (should show induction)
        "The cat sat on the mat. The cat was happy.",
        
        # Code-like pattern
        "def foo(x): return x + 1. def bar(y): return y + 2.",
        
        # List pattern
        "Alice likes apples. Bob likes bananas. Charlie likes cherries.",
        
        # Repeated structure with different content
        "The red car is fast. The blue car is slow. The green car is medium.",
        
        # Question-answer pattern
        "What is the capital of France? The capital of France is Paris.",
        
        # Mathematical pattern
        "2 + 2 = 4. 3 + 3 = 6. 4 + 4 = 8.",
        
        # Name pattern (repeated context)
        "John Smith went to the store. John Smith bought milk.",
        
        # Pronoun resolution
        "Sarah went to the market because she needed food.",
        
        # Nested structure
        "The book that the student who studied hard read was interesting.",
        
        # Causal chain
        "It rained, so the ground got wet, which made the flowers grow."
    ]
    
    # Analyze each sentence
    all_results = []
    
    # Test on multiple heads to find interesting patterns
    heads_to_test = [0, 1, 4, 7, 10, 11]  # Heads that are often induction heads in GPT-2
    
    for head in heads_to_test:
        print(f"\n{'='*80}")
        print(f"🎯 Testing Head {head} in Layer {sae.layer}")
        print(f"{'='*80}")
        
        for text in test_sentences[:3]:  # Start with first 3 sentences
            stats = analyze_sentence(
                model, sae, text, 
                layer=sae.layer, 
                head=head, 
                verbose=False
            )
            print_analysis_results(text, stats, sae)
            all_results.append({
                'text': text,
                'head': head,
                'stats': stats
            })
    
    # Find the most interesting patterns
    print(f"\n{'='*80}")
    print(f"🏆 Most Interesting Patterns Found")
    print(f"{'='*80}")
    
    # Sort by number of non-zero interactions
    sorted_by_nnz = sorted(all_results, key=lambda x: x['stats']['nnz'], reverse=True)
    print(f"\n📈 Highest feature interaction counts:")
    for i, result in enumerate(sorted_by_nnz[:3]):
        print(f"{i+1}. Head {result['head']}: {result['stats']['nnz']:,} interactions")
        print(f"   Text: '{result['text'][:50]}...'")
    
    # Find patterns with strongest individual interactions
    strongest_interactions = []
    for result in all_results:
        if result['stats']['top_interactions']:
            max_strength = abs(result['stats']['top_interactions'][0][2])
            strongest_interactions.append((result, max_strength))
    
    sorted_by_strength = sorted(strongest_interactions, key=lambda x: x[1], reverse=True)
    print(f"\n💪 Strongest individual feature interactions:")
    for i, (result, strength) in enumerate(sorted_by_strength[:3]):
        top_int = result['stats']['top_interactions'][0]
        print(f"{i+1}. Head {result['head']}: Feature {top_int[0]} → {top_int[1]} = {top_int[2]:.4f}")
        print(f"   Text: '{result['text'][:50]}...'")
    
    return all_results


if __name__ == "__main__":
    import os
    import sys
    
    try:
        results = main()
        print(f"\n✅ Analysis complete. Analyzed {len(results)} sentence-head combinations.")
    finally:
        # Clean up CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)