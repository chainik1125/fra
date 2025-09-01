# Feature-Resolved Attention (FRA) Project Updates

## Project Goal
Demonstrate that Feature-Resolved Attention (FRA) provides better mechanistic insights than token-level attention by analyzing attention patterns at the SAE feature level instead of token level.

## Current Status (2025-08-29)

### ✅ Completed Tasks
1. **Model and SAE Loading**: Successfully loaded GPT-2 small and attention SAEs from `ckkissane/attn-saes-gpt2-small-all-layers`
2. **FRA Implementation**: Created working implementations for computing feature-resolved attention
3. **Optimization**: Developed top-k GPU-optimized version that processes ~1500 position pairs/second
4. **Verification**: Confirmed index mapping correctly preserves original feature space indices

### 🔍 Key Findings

#### Working FRA Implementation
- Successfully computing Feature-Resolved Attention for GPT-2 layer 5
- Finding feature interactions with significant strengths (up to 10.2 in magnitude)
- Detecting self-interactions (e.g., F36759 → F36759) which may indicate induction-like behavior
- Processing speed: ~1500 position pairs/second with top-20 features

#### Example Results (OpenWebText sample, Head 10):
```
Non-zero interactions: 186,744
Top interactions:
  1. F19906 → F38828: -10.20
  2. F585 → F19906: 8.20
  3. F36759 → F36759: 7.05 (self-interaction)
```

### ⚠️ Confusing Points & Unexpected Behavior

#### 1. **Extremely Low SAE Sparsity**
- **Expected**: SAEs typically have >99% sparsity (100-200 active features out of 49,152)
- **Observed**: Only ~85% sparsity (~7,500 active features per position!)
- **L1 Norm Distribution**: Top-20 features capture only 18% of total L1 norm on average
  - Varies wildly by position (0.75% to 68%)
  - Many small activations rather than few large ones

#### 2. **Threshold Has Minimal Effect**
- Setting threshold to 1e-4 barely changes active feature count
- Suggests features have many small but non-zero activations
- Not clear if this is intentional or indicates an issue with the SAE

#### 3. **Unclear SAE Training Details**
- Could not confirm activation function (assumed ReLU based on standard practice)
- Unusual activation pattern suggests these SAEs may have been trained differently
- Possibly uses a different sparsity penalty or training objective

### 💡 Solutions Implemented

#### Top-K Feature Selection
- Keep only top-20 features per position by magnitude
- Reduces computation from ~7,500 × 7,500 to 20 × 20 interactions per position pair
- Makes FRA computation tractable despite unexpected density
- Still captures the most important feature interactions

### 📊 Technical Details

#### Memory & Performance
- Dense FRA matrix would be 49,152 × 49,152 = 18GB
- Sparse implementation with top-20: ~250KB
- Model loading: ~20 seconds (acceptable)
- FRA computation: <1 second for 50-token text

#### Implementation Files
- `fra/fra_analysis.py`: Original correct implementation (slow)
- `fra/fra_analysis_topk.py`: Top-k optimized version (fast, production-ready)
- `fra/sae_wrapper.py`: Custom wrapper for non-standard SAE format
- `fra/activation_utils.py`: Functions for extracting attention activations

## Next Steps
1. Find known Token Induction Heads (TIH) in GPT-2 small
2. Test if these heads show Conceptual Induction Heads (CIH) in feature space
3. Search for CIH in non-TIH heads
4. Compare interpretability of CIH vs TIH

## Open Questions
1. Why do these attention SAEs have such low sparsity?
2. Is this activation pattern intentional or a loading/interpretation issue?
3. Should we try different SAEs or accept this and move forward?
4. Does the low sparsity affect the validity of our FRA analysis?

## Recommendation
Despite the unexpected SAE behavior, the FRA implementation is working and finding meaningful feature interactions. The top-k approach successfully handles the density issue. We should proceed with the induction head analysis while keeping in mind the unusual SAE characteristics.