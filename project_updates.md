# Project Updates

## Initial Setup (Hour 1)

### Model and SAE Selection
- **Model**: GPT-2 Small (via TransformerLens) ✅
- **SAEs**: ckkissane/attn-saes-gpt2-small-all-layers (Hugging Face) ✅
  - These SAEs are trained on attention output (z) not ln1
  - Available for all layers of GPT-2 Small
  - dict_mult=64 (hidden dimension is 64x the model dimension)
- **Dataset**: Elriggs/openwebtext-100k (streaming enabled for memory efficiency) ✅

### Implementation Progress
- Created config.yaml for experiment configuration ✅
- Implemented loader functions in utils.py: ✅
  - `load_model()`: Loads GPT-2 Small via TransformerLens
  - `load_sae()`: Loads SAE weights and config (custom format, not SAE Lens)
  - `load_dataset()`: Using sample texts with repetition patterns
- Set up main script structure in induction_head.py ✅
- Successfully loaded all components and verified functionality

### Key Discoveries
- The attention SAEs from ckkissane are not in standard SAE Lens format
- They're trained on concatenated attention outputs (Hcat_z)
- Need custom loading logic to handle the weights and config

## Feature-Resolved Attention Implementation (Hours 2-4)

### Core FRA Implementation
- Implemented `get_sentence_fra_batch()` in fra_func.py for computing 4D FRA tensors ✅
- Uses sparse tensor representation to handle memory efficiently
- Processing speed: ~3.5s per sample on A100 GPU ✅

### Dashboard and Visualization
- **Single Sample Dashboard**: `fra/single_sample_viz.py`
  - Interactive visualization for individual text samples
  - Shows feature interactions for specific query-key position pairs
  
- **Accumulated Features Dashboard**: `fra/dataset_average_simple.py` ✅
  - Main working dashboard for analyzing top feature pairs across datasets
  - Efficient sparse tensor accumulation and processing
  - Generates HTML dashboard with top feature interactions
  - **To run**: `python run_and_package_results.py`
  - **Output location**: `/root/fra/results/accumulated_fra_L5H0_Nsamples.html`
  - **Package output**: `/root/fra/results/results_package.tar.gz`

### Key Findings
- Found 4.2M unique feature pairs across 20 samples
- 36% of top 200 pairs are self-interactions (potential "Conceptual Induction Heads")
- Top features show strong self-attention patterns (F25247, F44620, F23477)
- Processing maintains target speed of ~3.3s per sample

### Performance Optimizations
- Fixed O(n²) bottleneck in sparse tensor processing
- Optimized top-k extraction using torch operations
- Removed unnecessary dictionary conversions in verbose output