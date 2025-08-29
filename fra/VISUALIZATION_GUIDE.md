# FRA Visualization Guide

## Overview

The FRA project provides two ways to visualize Feature-Resolved Attention patterns:

1. **HTML Dashboard** - Static interactive HTML file
2. **Streamlit App** - Dynamic web application (if configured)

## HTML Dashboard

### Quick Start

```bash
# Generate dashboard for default text
python -m fra.single_sample_viz

# Or use the helper script
python fra/run_visualization.py
```

### Features

The HTML dashboard provides:

- **Interactive Feature Links**: Each feature ID is clickable and links directly to its Neuronpedia page
  - URL format: `https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_id}`
  - Example: https://www.neuronpedia.org/gpt2-small/5-att-kk/13

- **Load Description Button**: Click to load an embedded Neuronpedia view directly in the dashboard

- **Visual Indicators**:
  - Green arrows (→) for positive interactions
  - Red arrows (←) for negative interactions  
  - Yellow highlighting for self-interactions (potential induction heads)

- **Statistics Panel**: Shows sparsity, L0 norm, and total interactions

### Output Location

Dashboards are saved in `fra/results/` with the naming pattern:
- Without timestamp (default): `fra_dashboard_L{layer}H{head}.html`
- With timestamp: `fra_dashboard_L{layer}H{head}_{timestamp}.html`

### Customization

```python
from fra.single_sample_viz import create_fra_dashboard
from fra.induction_head import SAELensAttentionSAE
from transformer_lens import HookedTransformer

# Load model and SAE
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

# Custom text
text = "Your custom text here"

# Generate dashboard with custom settings
dashboard_path = create_fra_dashboard(
    model=model,
    sae=sae,
    text=text,
    layer=5,
    head=0,
    top_k_features=20,          # Number of top features per position
    top_k_interactions=30,       # Number of top interactions to show
    use_timestamp=False          # Set to True to add timestamp to filename
)
```

## Neuronpedia Integration

### Direct Links

Each feature in the dashboard links to its Neuronpedia page where you can:
- View detailed feature explanations
- See top activating examples
- Explore feature statistics
- Test the feature on custom text

### Embedded Views

Click the "Load Description" button to load an embedded Neuronpedia view directly in the dashboard. This provides:
- Feature explanation
- Activation plots
- Top examples

### Manual URL Construction

If you need to manually construct Neuronpedia URLs:

```
https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_id}
```

For embedded view (in iframe):
```
https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_id}?embed=true&embedexplanation=true&embedplots=true
```

## Streamlit App (Optional)

If you have a Streamlit app configured:

```bash
# Run Streamlit app
python fra/run_visualization.py --streamlit

# Or directly
streamlit run fra/streamlit_app.py
```

The Streamlit app provides dynamic interaction and real-time updates.

## Viewing the Dashboard

### Local Machine

1. Open the HTML file directly in your browser:
   ```
   file:///path/to/fra/results/fra_dashboard_L5H0.html
   ```

2. Or serve it with Python:
   ```bash
   cd fra/results
   python -m http.server 8000
   # Then open http://localhost:8000 in your browser
   ```

### Remote Server

If running on a remote server:

1. **Download the HTML file** to your local machine
2. Open it in any modern web browser

Or use SSH port forwarding:
```bash
ssh -L 8000:localhost:8000 user@remote-server
# Then on the server: cd fra/results && python -m http.server 8000
# Open http://localhost:8000 in your local browser
```

## Troubleshooting

### Features Not Loading

The dashboard creates direct links to Neuronpedia. If features don't load:
1. Check your internet connection
2. Try clicking the feature ID link to open Neuronpedia in a new tab
3. Neuronpedia may be temporarily unavailable

### Dashboard Not Generating

1. Ensure all dependencies are installed:
   ```bash
   pip install torch transformer-lens sae-lens
   ```

2. Check CUDA availability if using GPU:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. Check the results folder exists:
   ```bash
   ls fra/results/
   ```

## File Organization

```
fra/
├── results/                 # Generated dashboards (gitignored)
│   └── fra_dashboard_*.html
├── single_sample_viz.py     # Dashboard generation code
├── run_visualization.py     # Helper script
└── neuron_html.html         # Template reference
```

## Tips

1. **Best Text Length**: 20-100 tokens work well for visualization
2. **Repetition**: Text with repetition helps identify induction heads
3. **Layer Selection**: Layers 5-7 often have interesting attention patterns
4. **Head Selection**: Try heads 0, 5, 10 for variety

## Example Texts

```python
# Induction pattern
text = "The cat sat on the mat. The cat was happy."

# Subject-verb agreement
text = "The students who studied hard passed. The student who studied hard passed."

# Coreference
text = "Alice went to the store. She bought milk."
```