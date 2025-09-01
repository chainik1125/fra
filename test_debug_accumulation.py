#!/usr/bin/env python
"""Debug accumulation timing."""

import torch
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
import time

torch.set_grad_enabled(False)

print("Debug accumulation timing...")

# Load model and SAE
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

text = "The cat sat on the mat. The cat was happy."

# Time the FRA computation alone
print("\n1. FRA computation alone:")
t0 = time.time()
fra_result = get_sentence_fra_batch(
    model, sae, text, layer=5, head=0,
    max_length=128, top_k=20, verbose=False
)
t1 = time.time()
print(f"   FRA computation: {t1-t0:.1f}s")

# Get the sparse tensor
fra_sparse = fra_result['fra_tensor_sparse']
indices = fra_sparse.indices()
values = fra_sparse.values()
print(f"   Sparse tensor nnz: {fra_sparse._nnz()}")

# Time the accumulation loop
print("\n2. Accumulation timing:")
interaction_sum = {}
interaction_count = {}
top_examples = {}

t0 = time.time()
for i in range(min(100, indices.shape[1])):  # Just first 100 for test
    q_pos, k_pos, q_feat, k_feat = indices[:, i].tolist()
    strength = values[i].item()
    
    # Skip self-interactions
    if q_feat == k_feat:
        continue
    
    # Update running statistics
    pair_key = (q_feat, k_feat)
    interaction_sum[pair_key] = interaction_sum.get(pair_key, 0.0) + abs(strength)
    interaction_count[pair_key] = interaction_count.get(pair_key, 0) + 1
    
t1 = time.time()
print(f"   Accumulation (100 interactions): {t1-t0:.3f}s")

# Extrapolate
total_nnz = fra_sparse._nnz()
estimated_time = (t1-t0) * (total_nnz / 100)
print(f"   Estimated for all {total_nnz} interactions: {estimated_time:.1f}s")

# Now test with example creation and list management
print("\n3. With example creation:")
from fra.dataset_search import InteractionExample

top_examples = {}
tokens = model.tokenizer.encode(text, truncation=True, max_length=128)
token_strs = [model.tokenizer.decode(t) for t in tokens]

t0 = time.time()
for i in range(min(100, indices.shape[1])):
    q_pos, k_pos, q_feat, k_feat = indices[:, i].tolist()
    strength = values[i].item()
    
    if q_feat == k_feat:
        continue
    
    # Create example
    example = InteractionExample(
        sample_idx=0,
        text=text[:200],
        query_pos=q_pos, key_pos=k_pos,
        strength=strength,
        query_token=token_strs[q_pos] if q_pos < len(token_strs) else "",
        key_token=token_strs[k_pos] if k_pos < len(token_strs) else ""
    )
    
    # Update top examples
    pair_key = (q_feat, k_feat)
    if pair_key not in top_examples:
        top_examples[pair_key] = []
    
    examples_list = top_examples[pair_key]
    
    # This is where the slowdown might be!
    if len(examples_list) < 3:
        examples_list.append(example)
    else:
        # The min() call here could be slow with many examples
        min_strength = min(abs(ex.strength) for ex in examples_list)
        if abs(strength) > min_strength:
            examples_list.append(example)
            if len(examples_list) > 6:  # 2 * top_k
                examples_list.sort(key=lambda x: abs(x.strength), reverse=True)
                top_examples[pair_key] = examples_list[:3]

t1 = time.time()
print(f"   With examples (100 interactions): {t1-t0:.3f}s")
estimated_time = (t1-t0) * (total_nnz / 100)
print(f"   Estimated for all {total_nnz} interactions: {estimated_time:.1f}s")

print(f"\n⚠️ If estimated time >> 3s, the accumulation is the bottleneck!")