"""
Test SAE loading and L0 calculation using the exact code from the Colab notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from fra.utils import load_model, load_dataset_hf

# AutoEncoder class from the Colab
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = torch.float32  # Assuming float32
        torch.manual_seed(cfg.get("seed", 0))
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.dtype = dtype
        self.device = cfg["device"]

        self.version = 0
        self.to(cfg["device"])

    def forward(self, x, per_token=False):
        x_cent = x - self.b_dec  # KEY: Subtract b_dec first!
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)  # [batch_size, d_hidden]
        x_reconstruct = acts @ self.W_dec + self.b_dec  # [batch_size, act_size]
        if per_token:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1)  # [batch_size]
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1))  # [batch_size]
            loss = l2_loss + l1_loss  # [batch_size]
        else:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)  # []
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1).mean(dim=0))  # []
            loss = l2_loss + l1_loss  # []
        return loss, x_reconstruct, acts, l2_loss, l1_loss


def main():
    print("Loading model...")
    model = load_model('gpt2-small', 'cuda')
    
    print("Loading dataset...")
    dataset = load_dataset_hf("Elriggs/openwebtext-100k", split="train", streaming=True, seed=42)
    sample = next(iter(dataset))
    text = sample['text']
    
    # Truncate to reasonable length
    tokens = model.tokenizer.encode(text)[:50]
    text = model.tokenizer.decode(tokens)
    print(f"\nSample text ({len(tokens)} tokens):")
    print(text[:200] + "..." if len(text) > 200 else text)
    
    # Get attention activations for layer 5
    print("\nGetting attention activations...")
    from fra.activation_utils import get_attention_activations
    activations = get_attention_activations(model, text, layer=5, max_length=128)
    print(f"Activations shape: {activations.shape}")
    
    # Load the SAE weights directly
    print("\nLoading SAE from HuggingFace...")
    from fra.utils import load_sae
    sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
    
    # Create config based on what we know
    cfg = {
        "dict_size": 49152,  # 64x multiplier
        "act_size": 768,     # GPT-2 small hidden size
        "l1_coeff": sae_data['config'].get('l1_coeff', 1.0),
        "device": "cuda",
        "seed": 0
    }
    
    # Create SAE and load weights
    print("\nCreating SAE with Colab code...")
    sae = AutoEncoder(cfg)
    
    # Load the state dict
    sae.W_enc.data = sae_data['state_dict']['W_enc'].cuda()
    sae.W_dec.data = sae_data['state_dict']['W_dec'].cuda()
    sae.b_enc.data = sae_data['state_dict']['b_enc'].cuda()
    sae.b_dec.data = sae_data['state_dict']['b_dec'].cuda()
    
    print(f"SAE loaded: d_hidden={sae.d_hidden}, act_size={cfg['act_size']}")
    
    # Run forward pass
    print("\nRunning SAE forward pass...")
    with torch.no_grad():
        loss, x_reconstruct, acts, l2_loss, l1_loss = sae(activations, per_token=True)
    
    # Calculate L0 (number of active features)
    l0_per_token = (acts > 0).sum(dim=-1).float()
    avg_l0 = l0_per_token.mean().item()
    
    print(f"\n📊 Results:")
    print(f"  Features shape: {acts.shape}")
    print(f"  Average L0 (active features): {avg_l0:.1f}")
    print(f"  Min L0: {l0_per_token.min().item():.0f}")
    print(f"  Max L0: {l0_per_token.max().item():.0f}")
    
    # Check L1 norm
    l1_per_token = acts.abs().sum(dim=-1)
    avg_l1 = l1_per_token.mean().item()
    print(f"  Average L1 norm: {avg_l1:.2f}")
    
    # Check sparsity
    sparsity = 1.0 - (avg_l0 / acts.shape[1])
    print(f"  Sparsity: {sparsity:.2%}")
    
    # Per-position breakdown
    print(f"\nPer-position L0:")
    for i in range(min(10, acts.shape[0])):
        l0 = (acts[i] > 0).sum().item()
        l1 = acts[i].abs().sum().item()
        max_feat = acts[i].max().item()
        print(f"  Position {i:2d}: L0={l0:4d}, L1={l1:7.2f}, max_activation={max_feat:.3f}")
    
    print(f"\n{'='*50}")
    print(f"Expected L0 ~16 according to paper")
    print(f"Actual L0: {avg_l0:.1f}")
    
    # Check if we're using the right hook point
    print(f"\nNote: We're using attention OUTPUT (hook_attn_out) activations")
    print(f"This should be the concatenated outputs from all heads")


if __name__ == "__main__":
    import os
    
    try:
        main()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)