"""
Interactive HTML dashboard for visualizing Feature-Resolved Attention on a single text sample.
Includes Neuronpedia feature descriptions for interpretability.
"""

import torch
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
import html


def get_neuronpedia_url(layer: int, feature_idx: int, embed: bool = False) -> str:
    """
    Get Neuronpedia URL for a feature.
    
    Args:
        layer: Layer number
        feature_idx: Feature index
        embed: Whether to get embedded version
        
    Returns:
        Neuronpedia URL
    """
    base_url = f"https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_idx}"
    if embed:
        return f"{base_url}?embed=true&embedexplanation=true&embedplots=true&embedtest=false"
    return base_url


def create_fra_dashboard(
    model: Any,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    top_k_features: int = 20,
    top_k_interactions: int = 50,
    output_path: str = None,
    use_timestamp: bool = False
) -> str:
    """
    Create an interactive HTML dashboard for FRA visualization.
    
    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Layer number
        head: Head number
        top_k_features: Number of top features per position
        top_k_interactions: Number of top interactions to show
        output_path: Path to save HTML file
        
    Returns:
        Path to the generated HTML file
    """
    from fra.induction_head import compute_fra, get_top_feature_interactions, get_attention_activations
    from datetime import datetime
    
    # Set default output path in results folder
    if output_path is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"fra_dashboard_L{layer}H{head}_{timestamp}.html"
        else:
            output_path = results_dir / f"fra_dashboard_L{layer}H{head}.html"
    else:
        output_path = Path(output_path)
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute FRA
    print(f"Computing FRA for layer {layer}, head {head}...")
    fra_result = compute_fra(
        model, sae, text,
        layer=layer, head=head,
        top_k=top_k_features,
        verbose=True
    )
    
    # Get top interactions
    top_interactions = get_top_feature_interactions(
        fra_result['fra_matrix'],
        top_k=top_k_interactions
    )
    
    # Build feature URLs for Neuronpedia
    print("Building Neuronpedia links...")
    
    # Tokenize text for display
    tokens = model.tokenizer.encode(text)
    token_strings = [model.tokenizer.decode([t]) for t in tokens]
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature-Resolved Attention Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .stat-card h3 {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #495057;
        }}
        
        .section {{
            padding: 30px;
        }}
        
        .section h2 {{
            color: #495057;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .text-display {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            font-family: 'Courier New', monospace;
            line-height: 1.8;
        }}
        
        .token {{
            display: inline-block;
            padding: 2px 4px;
            margin: 2px;
            background: #e9ecef;
            border-radius: 3px;
            transition: all 0.3s;
        }}
        
        .token:hover {{
            background: #667eea;
            color: white;
            transform: scale(1.1);
        }}
        
        .interactions-grid {{
            display: grid;
            gap: 20px;
        }}
        
        .interaction-card {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s;
        }}
        
        .interaction-card:hover {{
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .interaction-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .interaction-strength {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .positive {{
            color: #28a745;
        }}
        
        .negative {{
            color: #dc3545;
        }}
        
        .feature-pair {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 20px;
            align-items: center;
        }}
        
        .feature-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .feature-box h4 {{
            color: #495057;
            margin-bottom: 8px;
        }}
        
        .feature-description {{
            color: #6c757d;
            font-size: 0.9em;
            line-height: 1.4;
        }}
        
        .arrow {{
            font-size: 2em;
            color: #667eea;
        }}
        
        .neuronpedia-link {{
            display: inline-block;
            margin-top: 10px;
            color: #667eea;
            text-decoration: none;
            font-size: 0.85em;
        }}
        
        .neuronpedia-link:hover {{
            text-decoration: underline;
        }}
        
        .self-interaction {{
            background: #fff3cd;
            border-color: #ffc107;
        }}
        
        .self-interaction .feature-box {{
            border-left-color: #ffc107;
        }}
        
        .load-btn {{
            margin-top: 10px;
            padding: 6px 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85em;
        }}
        
        .load-btn:hover {{
            background: #764ba2;
        }}
        
        .feature-desc-content {{
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
            font-size: 0.9em;
            display: none;
        }}
        
        .feature-desc-content.loaded {{
            display: block;
        }}
        
        .feature-box h4 a:hover {{
            text-decoration: underline !important;
        }}
        
        @media (max-width: 768px) {{
            .feature-pair {{
                grid-template-columns: 1fr;
            }}
            
            .arrow {{
                text-align: center;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Feature-Resolved Attention Dashboard</h1>
            <p>Layer {layer}, Head {head} | GPT-2 Small</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Sequence Length</h3>
                <div class="value">{fra_result['seq_len']}</div>
            </div>
            <div class="stat-card">
                <h3>Sparsity</h3>
                <div class="value">{fra_result['sparsity']*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Avg L0</h3>
                <div class="value">{fra_result['avg_l0']:.1f}</div>
            </div>
            <div class="stat-card">
                <h3>Non-zero Interactions</h3>
                <div class="value">{fra_result['nnz']:,}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📝 Input Text</h2>
            <div class="text-display">
                {''.join([f'<span class="token" title="Token {i}">{html.escape(t)}</span>' for i, t in enumerate(token_strings[:fra_result['seq_len']])])}
            </div>
        </div>
        
        <div class="section">
            <h2>🔥 Top Feature Interactions</h2>
            <div class="interactions-grid">
"""
    
    # Add interaction cards
    for i, (q_feat, k_feat, strength) in enumerate(top_interactions[:top_k_interactions]):
        q_url = get_neuronpedia_url(layer, q_feat)
        k_url = get_neuronpedia_url(layer, k_feat)
        
        is_self = (q_feat == k_feat)
        card_class = "interaction-card self-interaction" if is_self else "interaction-card"
        strength_class = "positive" if strength > 0 else "negative"
        arrow_symbol = "→" if strength > 0 else "←"
        
        html_content += f"""
                <div class="{card_class}">
                    <div class="interaction-header">
                        <h3>Interaction #{i+1}</h3>
                        <span class="interaction-strength {strength_class}">{strength:.4f}</span>
                    </div>
                    <div class="feature-pair">
                        <div class="feature-box">
                            <h4>Query: <a href="{q_url}" target="_blank" style="color: #667eea; text-decoration: none;">Feature {q_feat} ↗</a></h4>
                            <p class="feature-description">Click feature number above to view on Neuronpedia</p>
                            <button onclick="loadFeatureInfo({layer}, {q_feat}, 'q_{i}')" class="load-btn">Load Description</button>
                            <div id="q_{i}_desc" class="feature-desc-content"></div>
                        </div>
                        <div class="arrow">{arrow_symbol}</div>
                        <div class="feature-box">
                            <h4>Key: <a href="{k_url}" target="_blank" style="color: #667eea; text-decoration: none;">Feature {k_feat} ↗</a></h4>
                            <p class="feature-description">Click feature number above to view on Neuronpedia</p>
                            <button onclick="loadFeatureInfo({layer}, {k_feat}, 'k_{i}')" class="load-btn">Load Description</button>
                            <div id="k_{i}_desc" class="feature-desc-content"></div>
                        </div>
                    </div>
                    {f'<div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 5px; text-align: center;"><strong>⚡ Self-Interaction</strong> - Potential induction behavior</div>' if is_self else ''}
                </div>
"""
    
    html_content += """
            </div>
        </div>
    </div>
    
    <script>
        // Function to load feature info in iframe
        function loadFeatureInfo(layer, featureId, targetId) {
            const targetDiv = document.getElementById(targetId + '_desc');
            const embedUrl = `https://www.neuronpedia.org/gpt2-small/${layer}-att-kk/${featureId}?embed=true&embedexplanation=true&embedplots=true&embedtest=false`;
            
            // Toggle display
            if (targetDiv.classList.contains('loaded') && targetDiv.innerHTML !== '') {
                targetDiv.classList.remove('loaded');
                targetDiv.innerHTML = '';
            } else {
                targetDiv.innerHTML = `<iframe src="${embedUrl}" style="width:100%; height:400px; border:none; border-radius:5px;"></iframe>`;
                targetDiv.classList.add('loaded');
            }
        }
        
        // Add interactive features
        document.addEventListener('DOMContentLoaded', function() {
            // Animate stats on load
            const statValues = document.querySelectorAll('.stat-card .value');
            statValues.forEach((stat, index) => {
                stat.style.opacity = '0';
                stat.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    stat.style.transition = 'all 0.5s ease';
                    stat.style.opacity = '1';
                    stat.style.transform = 'translateY(0)';
                }, index * 100);
            });
            
            // Animate interaction cards on scroll
            const observerOptions = {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            };
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            }, observerOptions);
            
            const cards = document.querySelectorAll('.interaction-card');
            cards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(30px)';
                card.style.transition = `all 0.5s ease ${index * 0.05}s`;
                observer.observe(card);
            });
        });
    </script>
</body>
</html>
"""
    
    # Save HTML file
    output_path = Path(output_path)
    output_path.write_text(html_content)
    print(f"Dashboard saved to: {output_path}")
    
    return str(output_path)


def main():
    """Generate FRA dashboard for a sample text."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from fra.induction_head import SAELensAttentionSAE
    
    torch.set_grad_enabled(False)
    
    # Load model and SAE
    print("Loading model and SAE...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    
    RELEASE = "gpt2-small-hook-z-kk"
    SAE_ID = "blocks.5.hook_z"
    sae = SAELensAttentionSAE(RELEASE, SAE_ID, device="cuda")
    
    # Sample text with repetition (good for finding induction)
    text = "The cat sat on the mat. The cat was happy. The dog ran in the park. The dog was tired."
    
    # Generate dashboard (will automatically save to results folder)
    dashboard_path = create_fra_dashboard(
        model=model,
        sae=sae,
        text=text,
        layer=5,
        head=0,
        top_k_features=20,
        top_k_interactions=30
    )
    
    print(f"\n✅ Dashboard generated successfully!")
    print(f"📁 Open {dashboard_path} in your browser to view the interactive visualization.")
    
    return dashboard_path


if __name__ == "__main__":
    import os
    
    try:
        dashboard_path = main()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)