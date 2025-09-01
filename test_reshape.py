"""
Test if we need to reshape the attention activations.
"""

import torch
import torch.nn.functional as F
from fra.utils import load_model, load_sae

torch.set_grad_enabled(False)

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')

# Get components
W_enc = sae_data['state_dict']['W_enc'].cuda()
b_enc = sae_data['state_dict']['b_enc'].cuda()
b_dec = sae_data['state_dict']['b_dec'].cuda()

print(f"W_enc shape: {W_enc.shape}")
print(f"Expected input dim: {W_enc.shape[0]}")

text = "The cat sat."

# Try different hook points
hook_points = [
    "blocks.5.hook_attn_out",  # What we've been using
    "blocks.5.attn.hook_z",    # Raw attention output (before projection)
    "blocks.5.hook_z",         # Alternative name?
]

for hook_name in hook_points:
    print(f"\n{'='*50}")
    print(f"Testing hook: {hook_name}")
    
    try:
        # Get activations
        tokens = model.tokenizer.encode(text)
        tokens_t = torch.tensor([tokens]).cuda()
        
        _, cache = model.run_with_cache(tokens_t, names_filter=[hook_name])
        
        if hook_name in cache:
            act = cache[hook_name][0]  # Remove batch dim
            print(f"  Raw shape: {act.shape}")
            
            # Check if it needs reshaping
            if len(act.shape) == 3:  # [seq, n_heads, d_head]
                n_heads = act.shape[1]
                d_head = act.shape[2]
                print(f"  Detected n_heads={n_heads}, d_head={d_head}")
                
                # Reshape to concatenated form
                import einops
                act_concat = einops.rearrange(act, "seq n_heads d_head -> seq (n_heads d_head)")
                print(f"  Reshaped to: {act_concat.shape}")
                
                # Test encoding with reshaped
                x_cent = act_concat - b_dec
                features = F.relu(x_cent @ W_enc + b_enc)
                l0 = (features > 0).sum(1).float().mean().item()
                print(f"  L0 with reshape: {l0:.1f}")
                
            elif len(act.shape) == 2:  # [seq, d_model]
                print(f"  Already concatenated: {act.shape}")
                
                # Test encoding
                x_cent = act - b_dec
                features = F.relu(x_cent @ W_enc + b_enc)
                l0 = (features > 0).sum(1).float().mean().item()
                print(f"  L0: {l0:.1f}")
        else:
            print(f"  Hook not found")
            
    except Exception as e:
        print(f"  Error: {e}")

# Also check what config says
print(f"\n{'='*50}")
print("SAE Config:")
for key, val in sae_data['config'].items():
    print(f"  {key}: {val}")

import os
os._exit(0)