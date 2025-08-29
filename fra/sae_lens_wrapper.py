"""
SAE Lens wrapper for attention SAEs that provides the same interface as SimpleAttentionSAE.
"""

import torch
import torch.nn.functional as F
from typing import Any
from sae_lens import SAE


class SAELensAttentionSAE:
    """Wrapper for SAE Lens attention SAEs with encode/decode methods."""
    
    def __init__(self, release: str, sae_id: str, device: str = "cuda"):
        """Initialize SAE from SAE Lens.
        
        Args:
            release: SAE Lens release name (e.g., "gpt2-small-hook-z-kk")
            sae_id: SAE ID (e.g., "blocks.5.hook_z")
            device: Device to load SAE on
        """
        self.release = release
        self.sae_id = sae_id
        self.device = device
        
        # Load the SAE from SAE Lens
        self.sae = SAE.from_pretrained(release, sae_id, device=device)
        
        # Turn off hook_z reshaping to have manual control
        if hasattr(self.sae, 'turn_off_forward_pass_hook_z_reshaping'):
            self.sae.turn_off_forward_pass_hook_z_reshaping()
        
        # Get dimensions
        self.d_in = self.sae.cfg.d_in  # Should be 768 for GPT-2 small attention
        self.d_sae = self.sae.cfg.d_sae  # Should be 49152
        
        # Extract weights for compatibility with FRA code
        self.W_dec = self.sae.W_dec  # [d_sae, d_in]
        self.W_enc = self.sae.W_enc  # [d_in, d_sae]
        self.b_enc = self.sae.b_enc  # [d_sae]
        self.b_dec = self.sae.b_dec  # [d_in]
        
        # Extract layer number from sae_id (e.g., "blocks.5.hook_z" -> 5)
        parts = sae_id.split('.')
        self.layer = int(parts[1]) if len(parts) > 1 else 0
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations to SAE features.
        
        Args:
            x: Input tensor of shape [..., d_in] where d_in=768 for concatenated attention heads
            
        Returns:
            SAE features of shape [..., d_sae]
        """
        original_shape = x.shape
        
        # If input is attention activations from hook_z, it might be [seq_len, n_heads, d_head]
        # We need to flatten the heads dimension
        if len(original_shape) == 3 and original_shape[-2:] == (12, 64):  # GPT-2 small has 12 heads, 64 dim each
            x = x.flatten(-2, -1)  # [seq_len, 768]
        
        # Ensure input is 2D for SAE
        if len(x.shape) > 2:
            batch_shape = x.shape[:-1]
            x = x.reshape(-1, self.d_in)
            features = self.sae.encode(x)
            features = features.reshape(*batch_shape, self.d_sae)
        else:
            features = self.sae.encode(x)
        
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to activations.
        
        Args:
            features: SAE features of shape [..., d_sae]
            
        Returns:
            Reconstructed activations of shape [..., d_in]
        """
        # Use SAE's decode method directly
        original_shape = features.shape
        if len(features.shape) > 2:
            batch_shape = features.shape[:-1]
            features = features.reshape(-1, self.d_sae)
            reconstructed = self.sae.decode(features)
            reconstructed = reconstructed.reshape(*batch_shape, self.d_in)
        else:
            reconstructed = self.sae.decode(features)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass through SAE.
        
        Args:
            x: Input tensor of shape [..., d_in]
            
        Returns:
            Tuple of (features, reconstructed) tensors
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return features, reconstructed
    
    def get_active_features(self, features: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Get indices of active features.
        
        Args:
            features: SAE features tensor
            threshold: Minimum activation to consider a feature active
            
        Returns:
            Boolean mask of active features
        """
        return features > threshold
    
    def feature_sparsity(self, features: torch.Tensor) -> float:
        """Calculate sparsity of features.
        
        Args:
            features: SAE features tensor
            
        Returns:
            Fraction of features that are zero
        """
        return (features == 0).float().mean().item()


def get_attention_activations_for_sae_lens(
    model: Any,
    input_text: str,
    layer: int,
    max_length: int = 128
) -> torch.Tensor:
    """
    Get attention activations in the format expected by SAE Lens SAEs.
    
    SAE Lens attention SAEs are trained on hook_z (the concatenated attention values),
    not hook_attn_out (the output after projection).
    
    Args:
        model: The HookedTransformer model
        input_text: Input text to analyze
        layer: Which layer to get activations from
        max_length: Maximum sequence length
        
    Returns:
        Tensor of shape (sequence_length, n_heads * d_head) = (seq_len, 768)
    """
    # Tokenize
    tokens = model.tokenizer.encode(input_text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    device = next(model.parameters()).device
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Get hook_z activations
    hook_name = f"blocks.{layer}.attn.hook_z"
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    
    # Shape: [batch=1, seq_len, n_heads=12, d_head=64]
    z = cache[hook_name].squeeze(0)  # Remove batch dimension
    
    # Flatten heads dimension: [seq_len, 12*64=768]
    z_flat = z.flatten(-2, -1)
    
    return z_flat