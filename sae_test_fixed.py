"""
Test SAE using the exact code from colab.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# AutoEncoder class from colab.py (with DTYPES fixed)
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        # Fix: Use float32 as default since DTYPES isn't defined
        dtype = torch.float32
        torch.manual_seed(cfg["seed"])
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
        x_cent = x - self.b_dec  # This is the key line!
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


print("Loading model and dataset...", flush=True)
from fra.utils import load_model, load_dataset_hf

model = load_model('gpt2-small', 'cuda')
dataset = load_dataset_hf("Elriggs/openwebtext-100k", split="train", streaming=True, seed=42)
sample = next(iter(dataset))

# Get a short text sample
tokens = model.tokenizer.encode(sample['text'])[:30]
text = model.tokenizer.decode(tokens)
print(f"Text ({len(tokens)} tokens): {text[:100]}...", flush=True)

# Get activations
print("Getting activations...", flush=True)
from fra.activation_utils import get_attention_activations
activations = get_attention_activations(model, text, layer=5, max_length=128)
print(f"Activations shape: {activations.shape}", flush=True)

# Load SAE weights
print("Loading SAE weights...", flush=True)
from fra.utils import load_sae
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')

# Create SAE with proper config
cfg = {
    "dict_size": 49152,
    "act_size": 768,
    "l1_coeff": 1.0,
    "device": "cuda",
    "seed": 0,
    "enc_dtype": "float32"
}

sae = AutoEncoder(cfg)

# Load the actual weights
sae.W_enc.data = sae_data['state_dict']['W_enc'].cuda()
sae.W_dec.data = sae_data['state_dict']['W_dec'].cuda()
sae.b_enc.data = sae_data['state_dict']['b_enc'].cuda()
sae.b_dec.data = sae_data['state_dict']['b_dec'].cuda()

print("Computing features...", flush=True)
with torch.no_grad():
    loss, x_reconstruct, acts, l2_loss, l1_loss = sae(activations, per_token=True)

# Calculate L0
l0_per_position = (acts > 0).sum(dim=-1).float()
avg_l0 = l0_per_position.mean().item()

print(f"\n{'='*50}", flush=True)
print(f"RESULTS:", flush=True)
print(f"  Average L0: {avg_l0:.1f} active features", flush=True)
print(f"  Expected: ~16 (from paper)", flush=True)
print(f"  Sparsity: {1.0 - avg_l0/49152:.2%}", flush=True)

print(f"\nPer-position L0:", flush=True)
for i in range(min(10, acts.shape[0])):
    l0 = l0_per_position[i].item()
    print(f"  Pos {i:2d}: {l0:4.0f} active features", flush=True)

import os
os._exit(0)