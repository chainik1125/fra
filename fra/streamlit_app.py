# streamlit_app.py
import streamlit as st
from urllib.parse import quote

def neuronpedia_iframe(
    layer: int,
    feature_idx: int,
    default_text: str = "",
    width: int = 860,
    height: int = 420,
    show_expl: bool = True,
    show_plots: bool = True,
    show_test: bool = False,
):
    base = f"https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_idx}"
    qs = [
        "embed=true",
        f"embedexplanation={'true' if show_expl else 'false'}",
        f"embedplots={'true' if show_plots else 'false'}",
        f"embedtest={'true' if show_test else 'false'}",
    ]
    if default_text:
        qs.append(f"defaulttesttext={quote(default_text)}")
    url = base + "?" + "&".join(qs)
    st.components.v1.iframe(url, width=width, height=height)

st.set_page_config(page_title="FRA — Feature Explainer", layout="wide")

st.title("Feature-Resolved Attention — Feature Explainer")

colL, colR = st.columns([1,2])

with colL:
    text = st.text_area("Sample text (optional for activation tester)", "The cat sat on the mat.")
    layer = st.number_input("Layer (0–11 for GPT-2 Small)", min_value=0, max_value=11, value=3, step=1)
    feat = st.number_input("Feature index (from your FRA / SAE)", min_value=0, value=1234, step=1)
    show_plots = st.checkbox("Show plots", value=True)
    show_test = st.checkbox("Show activation tester", value=False)

with colR:
    st.markdown("#### Neuronpedia Feature Dashboard")
    neuronpedia_iframe(layer, feat, default_text=text, show_plots=show_plots, show_test=show_test)
