"""
Minimal test of FRA on very short text.
"""

import torch
import numpy as np
from pathlib import Path

from fra.utils import load_config, load_model, load_sae
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis import get_sentence_fra_sparse, get_top_feature_interactions
from fra.log import logger


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Load model and SAE
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    model = load_model(config["model"]["name"], str(device))
    sae_data = load_sae(config["sae"]["repo"], config["sae"]["layer"], str(device))
    sae = SimpleAttentionSAE(sae_data)
    
    print(f"Testing FRA on minimal example")
    print(f"Model: {config['model']['name']}, SAE Layer: {sae.layer}")
    
    # Very short test - just 5 tokens
    text = "The cat sat."
    head = 10
    
    print(f"\nText: '{text}'")
    print(f"Testing head {head}")
    
    # Compute sparse FRA with very short max_length
    fra_result = get_sentence_fra_sparse(
        model, sae, text, 
        layer=sae.layer, 
        head=head, 
        max_length=10,  # Very short
        verbose=True
    )
    
    print(f"\nResults:")
    print(f"  Sequence length: {fra_result['seq_len']} tokens")
    print(f"  Position pairs: {fra_result['total_pairs']}")
    print(f"  Non-zero interactions: {fra_result['nnz']:,}")
    print(f"  Density: {fra_result['density']:.8%}")
    
    # Get top interactions
    top_interactions = get_top_feature_interactions(
        fra_result['data_dep_int_matrix'], 
        top_k=10
    )
    
    print(f"\nTop feature interactions:")
    for i, (q_feat, k_feat, value) in enumerate(top_interactions):
        print(f"  {i+1}. Feature {q_feat} → Feature {k_feat}: {value:.6f}")


if __name__ == "__main__":
    import os
    
    try:
        main()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)