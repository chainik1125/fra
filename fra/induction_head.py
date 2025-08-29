"""
Feature-Resolved Attention (FRA) for Induction Heads
Main script for demonstrating FRA gives better mechanistic insights than token-level attention.
"""

import torch
from pathlib import Path
import argparse
from typing import Dict, Any
from fra.fra_func import analyze_feature_attention_interactions
from fra.utils import load_config, load_model, load_sae, load_dataset_hf
from fra.sae_wrapper import SimpleAttentionSAE
from fra.activation_utils import get_attention_activations, get_attention_pattern
from fra.log import logger




def loader(config_path: str = None):
    """Main entry point for Feature-Resolved Attention experiments.
    
    Args:
        config_path: Path to configuration file
    """
    # Determine config path
    if config_path is None:
        # Try to find config.yaml in project root
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            config_path = Path.cwd() / "config.yaml"
    
    # Load configuration
    config = load_config(str(config_path))
    
    # Set device
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config["experiment"]["seed"])
    
    # Create output directory
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load model
    model = load_model(
        model_name=config["model"]["name"],
        device=str(device)
    )
    
    # Step 2: Load SAE for specified layer
    sae_data = load_sae(
        repo=config["sae"]["repo"],
        layer=config["sae"]["layer"],
        device=str(device)
    )
    
    # Step 3: Load dataset (with streaming for memory efficiency)
    dataset = load_dataset_hf(
        dataset_name=config["dataset"]["name"],
        split=config["dataset"]["split"],
        streaming=config["dataset"].get("streaming", True),  # Default to streaming
        seed=config["experiment"]["seed"]
    )
    
    logger.info("Successfully loaded all components!")
    logger.info(f"Model: {config['model']['name']} with {model.cfg.n_layers} layers")
    logger.info(f"SAE: Layer {config['sae']['layer']}, dict_mult={sae_data['config'].get('dict_mult', 'unknown')}")
    logger.info(f"Dataset: {config['dataset']['name']} loaded")
    
    # TODO: Implement Feature-Resolved Attention analysis
    # This will be implemented in the next steps
    
    return model, sae_data, dataset


def main(config_path: str = None):
    """Main entry point for Feature-Resolved Attention experiments.
    
    Args:
        config_path: Path to configuration file
    """
    model, sae_data, dataset = loader(config_path)
    
    # Wrap SAE for easier use
    sae = SimpleAttentionSAE(sae_data)
    
    # Access samples from the dataset
    # For streaming datasets, use iteration
    if hasattr(dataset, '__iter__'):  # Streaming dataset
        sample = next(iter(dataset))
        print(f"First sample (streaming): {sample['text'][:100]}...")
    else:  # Regular dataset - can index directly
        sample = dataset[0]
        print(f"First sample: {sample['text'][:100]}...")
    
    print(f"\nModel: {model.cfg.model_name}, {model.cfg.n_layers} layers")
    print(f"SAE: Layer {sae.layer}, d_in={sae.d_in}, d_sae={sae.d_sae}")
    
    # Example: Get activations for a sample text
    
    #test_text = "The cat sat on the mat. The cat was happy."
    test_text = sample['text']

    
    # Get attention output activations for the SAE's layer
    activations = get_attention_activations(model, test_text, layer=sae.layer)
    print(f"\nActivations for test text:")
    print(f"  Text: '{test_text}'")
    print(f"  Activations shape: {activations.shape}")  # [seq_len, hidden_dim]
    
    # Encode to SAE features
    features = sae.encode(activations)
    print(f"  SAE features shape: {features.shape}")  # [seq_len, d_sae]
    print(f"  Feature sparsity: {sae.feature_sparsity(features):.2%}")
    
    # Get attention pattern for a specific head
    attn_pattern = get_attention_pattern(model, test_text, layer=sae.layer, head=0)
    print(f"  Attention pattern shape (head 0): {attn_pattern.shape}")  # [seq_len, seq_len]
    
    #Now need to implement feature resolved attention for this sample

    test_fra=analyze_feature_attention_interactions(model, sae, layer=sae.layer, head=0, 
            input_text=test_text, query_position=0, key_position=1)
    
    print(f'  FRA test shape: {test_fra["interaction_matrix"].shape}')
    print(f'  Query active features: {len(test_fra["query_active_features"])}')
    print(f'  Key active features: {len(test_fra["key_active_features"])}')
    
    return model, sae, dataset


if __name__ == "__main__":
    import os
    import sys
    
    parser = argparse.ArgumentParser(description="Feature-Resolved Attention Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    try:
        model, sae, dataset = main(args.config)
    finally:
        # Clean up CUDA resources to prevent core dump
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force exit to avoid cleanup issues
        os._exit(0)
    
    
    
    

    