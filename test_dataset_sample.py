"""
Test FRA on a sample from the actual dataset.
"""

import torch
from fra.utils import load_model, load_sae, load_dataset_hf
from fra.sae_wrapper import SimpleAttentionSAE
from fra.fra_analysis_topk import get_sentence_fra_topk, get_top_interactions_torch
import time

torch.set_grad_enabled(False)

print("Loading model and SAE...")
model = load_model('gpt2-small', 'cuda')
sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
sae = SimpleAttentionSAE(sae_data)

print("Loading dataset...")
dataset = load_dataset_hf(
    dataset_name="Elriggs/openwebtext-100k",
    split="train",
    streaming=True,
    seed=42
)

# Get first sample from dataset
print("\nGetting sample from dataset...")
sample = next(iter(dataset))
text = sample['text']

# Truncate to reasonable length for tokenization
tokens = model.tokenizer.encode(text)
if len(tokens) > 128:
    tokens = tokens[:128]
    text = model.tokenizer.decode(tokens)

print(f"\nSample text (truncated to {len(tokens)} tokens):")
print("="*80)
print(text[:500] + "..." if len(text) > 500 else text)
print("="*80)

# Test different heads that are often induction heads in GPT-2
heads_to_test = [0, 5, 10, 11]  # Common induction head positions in layer 5

print(f"\nTesting FRA on layer 5, different heads with top-20 features:")

results = []
for head in heads_to_test:
    print(f"\n--- Head {head} ---")
    t0 = time.time()
    
    fra_result = get_sentence_fra_topk(
        model, sae, text,
        layer=5, head=head,
        max_length=128,
        top_k=20,
        verbose=False
    )
    
    elapsed = time.time() - t0
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Sequence length: {fra_result['seq_len']} tokens")
    print(f"  Non-zero interactions: {fra_result['nnz']:,}")
    print(f"  Density: {fra_result['density']:.8%}")
    
    # Get top interactions
    top_interactions = get_top_interactions_torch(fra_result['data_dep_int_matrix'], top_k=10)
    
    if len(top_interactions) > 0:
        print(f"  Top 5 feature interactions:")
        for i, (q_feat, k_feat, value) in enumerate(top_interactions[:5]):
            print(f"    {i+1}. Feature {q_feat:5d} → Feature {k_feat:5d}: {value:8.4f}")
        
        # Check for self-interactions (potential induction)
        self_interactions = [(q, k, v) for q, k, v in top_interactions if q == k]
        if self_interactions:
            print(f"  Self-interactions found (potential induction):")
            for q, k, v in self_interactions[:3]:
                print(f"    Feature {q} → Feature {k}: {v:.4f}")
    else:
        print(f"  No interactions found")
    
    results.append({
        'head': head,
        'nnz': fra_result['nnz'],
        'top_interaction': top_interactions[0] if top_interactions else None
    })

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

# Find head with most interactions
best_head = max(results, key=lambda x: x['nnz'])
print(f"\nHead with most feature interactions: Head {best_head['head']} ({best_head['nnz']:,} interactions)")

# Find strongest interaction
strongest = None
for r in results:
    if r['top_interaction']:
        if strongest is None or abs(r['top_interaction'][2]) > abs(strongest[2]):
            strongest = r['top_interaction']
            strongest_head = r['head']

if strongest:
    q, k, v = strongest
    print(f"Strongest interaction: Feature {q} → Feature {k} = {v:.4f} (Head {strongest_head})")

# Look for patterns
print(f"\nPattern analysis:")
all_self_interactions = []
for r in results:
    if r['top_interaction'] and r['top_interaction'][0] == r['top_interaction'][1]:
        all_self_interactions.append((r['head'], r['top_interaction']))

if all_self_interactions:
    print(f"  Heads with self-interactions (potential induction): {[h for h, _ in all_self_interactions]}")
else:
    print(f"  No clear self-interaction patterns found")

import os
if torch.cuda.is_available():
    torch.cuda.empty_cache()
os._exit(0)