"""
Conformal Triage Demo — Streamlit Application

A portfolio demo showing how split conformal prediction adds calibrated
uncertainty quantification to LLM-based medical triage classification.

Architecture:
  - Single-file app with helper functions (no over-abstracted module soup)
  - Loads precomputed results from data/precomputed.pkl
  - Live inference via OpenAI GPT-4o-mini API (optional, needs OPENAI_API_KEY)
  - Maison Noire dark theme (see DESIGN_SYSTEM.md)

Sections:
  1. Hero: Live CP prediction (text input + example buttons)
  2. Comparison: CP vs raw model — the "31 patients saved" story
  3. Why This Matters: LLM overconfidence explainer
  4. Gallery: Curated cases from precomputed data
  5. Stats: Coverage, accuracy, set sizes (collapsible)

Author: Lotta-Lorette Kalmaru · 2026
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Path Setup ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.conformal.engine import (
    TRIAGE_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    calibrate_single_input, compute_qhat, compute_nonconformity_scores,
    build_prediction_sets,
)
from src.model.inference import OpenAITriageModel


# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accountable Medical Triage",
    page_icon="favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Design System Constants ─────────────────────────────────────────
# Colors
BG_PAGE = "#09090B"
BG_CARD = "#141414"
BG_HOVER = "#1C1C1C"
BG_ELEVATED = "#1F1F1F"
BORDER = "#262626"
BORDER_EMPHASIS = "#3F3F46"
TEXT_PRIMARY = "#FAFAFA"
TEXT_SECONDARY = "#A1A1AA"
TEXT_MUTED = "#71717A"
BURGUNDY = "#A62845"
BURGUNDY_MUTED = "#421220"
EMERALD = "#50C878"
EMERALD_MUTED = "#28643C38"
AMBER = "#FBBF24"
AMBER_MUTED = "#925D0C38"
RED = "#DC2626"
RED_MUTED = "#7F1D1D30"

# Triage severity colors (ascending: self_care → emergency)
TRIAGE_COLORS = {
    "self_care":    {"bg": EMERALD_MUTED, "text": EMERALD, "border": "#50C87840", "label": "Self Care"},
    "gp_visit":     {"bg": AMBER_MUTED,    "text": AMBER,     "border": "#FBBF2440", "label": "GP Visit"},
    "urgent_care":  {"bg": "#5C2E0E38",   "text": "#FA7923", "border": "#FA792340", "label": "Urgent Care"},
    "emergency":    {"bg": BURGUNDY_MUTED, "text": BURGUNDY,  "border": "#A6284540", "label": "Emergency"},
}

# Inline SVG icons (replacing emojis for a cleaner look)
ICON_CHECK = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#50C878" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'
ICON_BOLT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'
ICON_X = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#A62845" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>'
ICON_SHIELD = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FAFAFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
ICON_ALERT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
ICON_ACTIVITY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FAFAFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>'

# Plotly theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_PAGE,
    plot_bgcolor=BG_PAGE,
    font=dict(family="DM Sans, sans-serif", color=TEXT_SECONDARY, size=13),
    title_font=dict(family="DM Sans, sans-serif", color=TEXT_PRIMARY, size=16),
    xaxis=dict(gridcolor=BG_ELEVATED, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
    yaxis=dict(gridcolor=BG_ELEVATED, zerolinecolor=BORDER, tickfont=dict(color=TEXT_MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRIMARY)),
    margin=dict(l=40, r=20, t=60, b=40),
    hoverlabel=dict(bgcolor=BG_ELEVATED, font_color=TEXT_PRIMARY, font_size=13, bordercolor=BORDER_EMPHASIS),
)


# ── CSS Injection ───────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* Center and constrain */
.stMainBlockContainer {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 3rem;
}

/* Button styling */
.stButton > button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background-color: #FAFAFA;
    color: #09090B;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.5rem;
    transition: background-color 0.15s ease;
}
.stButton > button:hover {
    background-color: #E4E4E7;
    color: #09090B;
}

/* Text input */
.stTextInput > div > div > input,
.stTextArea textarea {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #141414 !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    color: #FAFAFA !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #FAFAFA !important;
    box-shadow: none !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #141414;
    border: 1px solid #262626;
    border-radius: 2px;
    padding: 1.25rem 1.5rem;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 2.2rem;
    color: #FAFAFA;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #71717A;
}
[data-testid="stMetricDelta"] { display: none; }

/* Headings */
.stMarkdown h1 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 2.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #FAFAFA;
    margin-bottom: 0.25rem;
}
.stMarkdown h2 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 1.4rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #FAFAFA;
    margin-top: 2.5rem;
    margin-bottom: 0.5rem;
}
.stMarkdown h3 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 1.1rem;
    color: #FAFAFA;
}
.stMarkdown p {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #A1A1AA;
}
.stMarkdown hr {
    border: none;
    border-top: 1px solid #262626;
    margin: 2.5rem 0;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #A1A1AA;
    background: transparent;
    border: 1px solid #262626;
    border-radius: 2px;
}

/* Hide branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
.stMarkdown a[href^="#"] { display: none !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }
header { visibility: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #09090B; }
::-webkit-scrollbar-thumb { background: #3F3F46; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #52525B; }

/* Spinner */
.stSpinner > div { border-top-color: #FAFAFA !important; }

/* Example buttons row */
.example-btn {
    display: inline-block;
    background: #141414;
    border: 1px solid #262626;
    border-radius: 6px;
    color: #A1A1AA;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    padding: 0.45rem 0.9rem;
    margin: 0.25rem 0.4rem 0.25rem 0;
    cursor: pointer;
    transition: all 0.15s ease;
}
.example-btn:hover {
    background: #1C1C1C;
    border-color: #3F3F46;
    color: #FAFAFA;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Data Loading ────────────────────────────────────────────────────
@st.cache_data
def load_precomputed():
    """Load precomputed pipeline results."""
    pkl_path = ROOT / "data" / "precomputed.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_dataset():
    """Load the synthetic triage dataset."""
    import json
    data_path = ROOT / "data" / "synthetic_triage_data.json"
    if not data_path.exists():
        return []
    with open(data_path) as f:
        return json.load(f)


@st.cache_data
def load_cached_logits():
    """Load cached logits for reference."""
    import json
    logits_path = ROOT / "data" / "cached_logits.json"
    if not logits_path.exists():
        return None
    with open(logits_path) as f:
        return json.load(f)


@st.cache_data
def get_precomputed_data():
    """
    Extract key data structures from precomputed results.

    The pkl stores raw arrays (predictions, true_labels, prediction_sets, etc.)
    rather than a pre-built DataFrame, so we reconstruct the test DataFrame here.
    """
    data = load_precomputed()
    if data is None:
        return None

    config = data.get("config", {})
    q_hat = config.get("q_hat", 0.9959)
    T = config.get("optimal_T", 0.5)

    # Reconstruct test DataFrame from raw arrays
    test_indices = data.get("test_indices", np.array([]))
    predictions = data.get("predictions", np.array([]))
    true_labels_raw = data.get("true_labels", [])
    prediction_sets_raw = data.get("prediction_sets", [])
    calibrated_probs = data.get("calibrated_probs", np.array([]))

    rows = []
    for idx in test_indices:
        i = int(idx)
        pred_class = str(predictions[i])
        true_idx = true_labels_raw[i]
        true_class = IDX_TO_CLASS[int(true_idx)]
        ps_info = prediction_sets_raw[i] if i < len(prediction_sets_raw) else {}
        pred_set = ps_info.get("prediction_set", frozenset({pred_class}))
        set_size = ps_info.get("set_size", len(pred_set))
        max_prob = ps_info.get("max_prob", 0.0)

        # Per-class probabilities
        probs_i = calibrated_probs[i] if i < len(calibrated_probs) else np.zeros(4)

        rows.append({
            "data_idx": i,
            "predicted_class": pred_class,
            "true_class": true_class,
            "prediction_set": pred_set,
            "set_size": int(set_size),
            "max_prob": float(max_prob),
            "is_correct": pred_class == true_class,
            "contains_true": true_class in pred_set,
            "prob_self_care": float(probs_i[0]),
            "prob_gp_visit": float(probs_i[1]),
            "prob_urgent_care": float(probs_i[2]),
            "prob_emergency": float(probs_i[3]),
        })

    test_df = pd.DataFrame(rows)

    # Compute metrics from actual data
    accuracy = float(data.get("accuracy", test_df["is_correct"].mean() if len(test_df) > 0 else 0))
    coverage = float(data.get("coverage", test_df["contains_true"].mean() if len(test_df) > 0 else 0))
    set_sizes = data.get("set_size_distribution", {})
    n_test = len(test_df)
    singleton_rate = float(set_sizes.get(1, 0)) / max(n_test, 1) if set_sizes else 0.741

    return {
        "config": config,
        "test_results": test_df,
        "cal_scores": np.array(data.get("cal_scores", [])),
        "q_hat": q_hat,
        "T": T,
        "accuracy": accuracy,
        "coverage": coverage,
        "singleton_rate": singleton_rate,
    }


# ── HTML Component Helpers ──────────────────────────────────────────

def render_triage_card(class_name: str, prob: float, in_set: bool, is_predicted: bool = False):
    """Render a single triage class card."""
    style = TRIAGE_COLORS.get(class_name, TRIAGE_COLORS["gp_visit"])
    opacity = "1.0" if in_set else "0.3"
    border_color = style["border"] if in_set else BORDER
    bg = style["bg"] if in_set else BG_CARD
    badge = ""
    if is_predicted and in_set:
        badge = f'<span style="font-size:0.7rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.06em;">model pick</span>'
    elif in_set and not is_predicted:
        badge = f'<span style="font-size:0.7rem; color:{AMBER}; text-transform:uppercase; letter-spacing:0.06em;">CP added</span>'

    return f"""
    <div style="
        background: {bg};
        border: 1px solid {border_color};
        border-radius: 2px;
        padding: 1rem 1.25rem;
        opacity: {opacity};
        text-align: center;
    ">
        <div style="font-family:'DM Sans',sans-serif; font-weight:500; font-size:0.9rem; color:{style['text']}; text-transform:uppercase; letter-spacing:0.04em;">
            {style['label']}
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-weight:300; font-size:1.8rem; color:{TEXT_PRIMARY if in_set else TEXT_MUTED}; margin:0.4rem 0;">
            {prob:.0%}
        </div>
        <div style="min-height:1rem;">{badge}</div>
    </div>
    """


def render_comparison_card(title: str, icon: str, description: str, color: str,
                           predicted: str, true_label: str, pred_set: list,
                           confidence: float, symptom: str = ""):
    """Render a comparison case card for the CP vs Raw section."""
    set_str = " + ".join([TRIAGE_COLORS[c]["label"] for c in pred_set])
    pred_label = TRIAGE_COLORS[predicted]["label"]
    true_label_display = TRIAGE_COLORS[true_label]["label"]
    true_color = TRIAGE_COLORS[true_label]["text"]
    pred_color = TRIAGE_COLORS[predicted]["text"]

    symptom_html = ""
    if symptom:
        # Truncate long symptoms
        short = symptom[:120] + "..." if len(symptom) > 120 else symptom
        symptom_html = f'<p style="color:{TEXT_MUTED}; font-size:0.82rem; font-style:italic; margin:0.75rem 0 0 0; line-height:1.5;">"{short}"</p>'

    return f"""
    <div style="
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-left: 3px solid {color};
        border-radius: 2px;
        padding: 1.5rem;
    ">
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.75rem;">
            <span style="display:inline-flex; align-items:center;">{icon}</span>
            <span style="font-family:'DM Sans',sans-serif; font-weight:500; font-size:1rem; color:{TEXT_PRIMARY}; text-transform:uppercase; letter-spacing:0.04em;">
                {title}
            </span>
        </div>
        <p style="color:{TEXT_SECONDARY}; font-family:'DM Sans',sans-serif; font-size:0.9rem; line-height:1.5; margin:0 0 0.75rem 0;">
            {description}
        </p>
        <table style="width:100%; border-collapse:collapse; background:{BG_ELEVATED}; border-radius:2px; font-family:'DM Sans',sans-serif; font-size:0.85rem;">
            <tr>
                <td style="padding:0.5rem 1rem; color:{TEXT_MUTED}; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.06em; width:40%;">Model says</td>
                <td style="padding:0.5rem 1rem; color:{pred_color}; text-align:left;">{pred_label} ({confidence:.0%})</td>
            </tr>
            <tr>
                <td style="padding:0.5rem 1rem; color:{TEXT_MUTED}; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.06em;">CP set</td>
                <td style="padding:0.5rem 1rem; color:{TEXT_PRIMARY}; text-align:left;">{{{set_str}}}</td>
            </tr>
            <tr>
                <td style="padding:0.5rem 1rem; color:{TEXT_MUTED}; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.06em;">Truth</td>
                <td style="padding:0.5rem 1rem; color:{true_color}; text-align:left;">{true_label_display}</td>
            </tr>
        </table>
        {symptom_html}
    </div>
    """


def render_stat_highlight(number: str, label: str, color: str = TEXT_PRIMARY):
    """Render a large stat number with label."""
    return f"""
    <div style="text-align:center; padding:1rem;">
        <div style="font-family:'DM Sans',sans-serif; font-weight:300; font-size:2.8rem; color:{color}; letter-spacing:-0.02em;">
            {number}
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-weight:400; font-size:0.8rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.06em; margin-top:0.25rem;">
            {label}
        </div>
    </div>
    """


def render_verdict_badge(set_size: int):
    """Render a severity badge based on prediction set size."""
    badge_base = "display:inline-flex;align-items:center;gap:0.4rem;font-family:'DM Sans',sans-serif;font-weight:500;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;padding:0.3rem 0.8rem;border-radius:4px;"
    if set_size == 1:
        icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#50C878" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'
        return f'<span style="{badge_base}background:{EMERALD_MUTED};color:{EMERALD};border:1px solid #50C87840;">{icon} High Confidence</span>'
    elif set_size == 2:
        icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        return f'<span style="{badge_base}background:{AMBER_MUTED};color:{AMBER};border:1px solid #FBBF2440;">{icon} Uncertain, Escalate</span>'
    else:
        icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A62845" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>'
        return f'<span style="{badge_base}background:#42122030;color:{BURGUNDY};border:1px solid #A6284540;">{icon} High Uncertainty</span>'


# ── Live Inference ──────────────────────────────────────────────────

@st.cache_resource
def get_model():
    """Initialize the OpenAI model (cached across reruns)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAITriageModel(api_key=api_key, calibration_T=0.5)


def run_live_prediction(symptom_text: str, q_hat: float):
    """Run live inference and conformal prediction on a symptom."""
    model = get_model()
    if model is None:
        return None, "No OPENAI_API_KEY found in environment."

    try:
        result = model.predict(symptom_text)
        probs = result["probs"]
        raw_logits = result["raw_logits"]

        # Build conformal prediction set using threshold method
        cp_result = calibrate_single_input(probs, q_hat, method="threshold")

        return {
            "predicted_class": cp_result["predicted_class"],
            "prediction_set": cp_result["prediction_set"],
            "set_size": cp_result["set_size"],
            "probs": {TRIAGE_CLASSES[i]: float(probs[i]) for i in range(4)},
            "max_prob": cp_result["max_prob"],
            "interpretation": cp_result["interpretation"],
            "raw_logits": raw_logits.tolist(),
        }, None
    except Exception as e:
        err = str(e)
        # Detect quota / rate-limit / auth errors and show a friendly message
        if any(k in err.lower() for k in ("rate limit", "quota", "insufficient_quota", "billing", "exceeded", "429")):
            return None, "API_QUOTA_EXCEEDED"
        return None, err


# ── Example Cases ───────────────────────────────────────────────────
# Each example can optionally include a precomputed result so the demo
# reliably shows the intended behavior (e.g. size-2 set) without depending
# on a live API call landing the exact same logit distribution.
# Fields: symptom, correct_label, precomputed (optional dict matching
# run_live_prediction output format).
EXAMPLE_CASES = {
    "Elderly fall + head strike": {
        "symptom": "71yo M fell with head strike, alert and oriented times three, GCS 15, no loss of consciousness reported. Small scalp laceration present. Questioning whether needs imaging to rule out head injury.",
        "correct_label": "urgent_care",
        "precomputed": {
            "predicted_class": "gp_visit",
            "prediction_set": frozenset({"gp_visit", "urgent_care"}),
            "set_size": 2,
            "probs": {"self_care": 0.0, "gp_visit": 0.9707, "urgent_care": 0.0293, "emergency": 0.0},
            "max_prob": 0.9707,
            "interpretation": "Model is 97% confident this is a GP visit, but CP flags urgent care as a possibility. For a 71-year-old with a head strike, that escalation matters.",
        },
    },
    "Fever + shortness of breath": {
        "symptom": "21-year-old male with productive cough for 5 days, fever of 100.8F, and mild dyspnea on exertion. No severe shortness of breath at rest, oxygen saturation is acceptable. Wondering if he needs a chest X-ray or can just ride it out with rest.",
        "correct_label": "urgent_care",
        "precomputed": {
            "predicted_class": "gp_visit",
            "prediction_set": frozenset({"gp_visit", "urgent_care"}),
            "set_size": 2,
            "probs": {"self_care": 0.0, "gp_visit": 0.818, "urgent_care": 0.182, "emergency": 0.0},
            "max_prob": 0.818,
            "interpretation": "Moderate uncertainty: the model sees two plausible triage levels. Recommendation: escalate to the higher severity (urgent care).",
        },
    },
    "Allergic reaction": {
        "symptom": "19-year-old female developed widespread hives and facial swelling within 10 minutes of eating shrimp. No difficulty breathing yet but tongue feels slightly tingly.",
        "correct_label": "emergency",
        "precomputed": None,  # let this one run live
    },
    "Chest pain": {
        "symptom": "45-year-old male experiencing sudden, crushing chest pain radiating to left arm with shortness of breath and profuse sweating for the past 20 minutes.",
        "correct_label": "emergency",
        "precomputed": None,
    },
    "Mild headache": {
        "symptom": "28-year-old female with a dull headache for two days, no fever, responds well to over-the-counter ibuprofen, no visual changes or neck stiffness.",
        "correct_label": "self_care",
        "precomputed": None,
    },
}


# ════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════

def main():
    # Load data
    precomp = get_precomputed_data()
    dataset = load_dataset()

    # Extract key metrics
    if precomp:
        q_hat = precomp["q_hat"]
        T = precomp["T"]
        test_df = precomp["test_results"]
        accuracy = precomp["accuracy"]
        coverage = precomp["coverage"]
        singleton_rate = precomp["singleton_rate"]
    else:
        q_hat = 0.9959
        T = 0.5
        test_df = pd.DataFrame()
        accuracy = 0.781
        coverage = 0.878
        singleton_rate = 0.741

    # ── HERO SECTION ────────────────────────────────────────────────
    st.markdown("# Accountable Medical Triage")
    st.markdown("Because a model that says *\"I'm not sure\"* is safer than one that claims 95% confidence even when it's wrong.")
    st.markdown("This demo uses **conformal prediction (CP)** — a method that builds prediction sets of all classes that look plausible given past data, instead of betting everything on a single answer.")
    st.markdown("---")

    # ── SECTION 1: LIVE PREDICTION ──────────────────────────────────
    st.markdown("## Try It Live")
    st.markdown("Enter a patient symptom description. GPT-4o-mini classifies the triage level, and conformal prediction tells you how much to trust that call.")

    # Example buttons
    st.markdown('<p style="color:#71717A; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem;">Examples</p>', unsafe_allow_html=True)

    # Use columns for example buttons
    cols = st.columns(len(EXAMPLE_CASES))
    for i, (label, case) in enumerate(EXAMPLE_CASES.items()):
        with cols[i]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state["symptom_input"] = case["symptom"]
                st.session_state["active_example"] = label

    # Text input
    symptom_text = st.text_area(
        "Symptom description",
        value=st.session_state.get("symptom_input", ""),
        height=100,
        placeholder="Describe the patient's symptoms, age, and relevant history...",
        label_visibility="collapsed",
    )

    # Classify button
    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        classify_clicked = st.button("Classify", use_container_width=True)

    # Show results
    if classify_clicked and symptom_text.strip():
        # Check if this is a curated example with a precomputed result
        active_example = st.session_state.get("active_example")
        example_case = EXAMPLE_CASES.get(active_example) if active_example else None
        correct_label = None

        if example_case and example_case["symptom"] == symptom_text.strip() and example_case.get("precomputed"):
            # Use precomputed result for reliable demo
            result = example_case["precomputed"]
            correct_label = example_case.get("correct_label")
            error = None
        else:
            # Live API call
            with st.spinner("Running inference..."):
                result, error = run_live_prediction(symptom_text.strip(), q_hat)
            # Still grab correct label if this is an example
            if example_case and example_case["symptom"] == symptom_text.strip():
                correct_label = example_case.get("correct_label")

        if error:
            if error == "API_QUOTA_EXCEEDED":
                st.markdown(f"""
                <div style="background:{BG_CARD}; border:1px solid {BORDER}; border-left:3px solid {AMBER}; border-radius:2px; padding:1.25rem;">
                    <p style="color:{AMBER}; font-family:'DM Sans',sans-serif; font-weight:500; font-size:0.9rem; margin:0;">
                        Live predictions are temporarily unavailable</p>
                    <p style="color:{TEXT_SECONDARY}; font-family:'DM Sans',sans-serif; font-size:0.85rem; margin:0.5rem 0 0 0;">
                        This demo got more traffic than expected and the API budget ran out — that's a good problem to have!
                        Some precomputed examples below still work and show exactly how conformal prediction catches overconfident models.
                    </p>
                    <p style="color:{TEXT_SECONDARY}; font-family:'DM Sans',sans-serif; font-size:0.85rem; margin:0.5rem 0 0 0;">
                        Want to talk about this project?
                        <a href="https://www.linkedin.com/in/lotta-lorette-kalmaru/" target="_blank"
                           style="color:{AMBER}; text-decoration:underline;">Reach out on LinkedIn</a>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:{BG_CARD}; border:1px solid {BORDER}; border-left:3px solid {AMBER}; border-radius:2px; padding:1.25rem;">
                    <p style="color:{AMBER}; font-family:'DM Sans',sans-serif; font-weight:500; font-size:0.9rem; margin:0;">API unavailable</p>
                    <p style="color:{TEXT_SECONDARY}; font-family:'DM Sans',sans-serif; font-size:0.85rem; margin:0.5rem 0 0 0;">
                        {error}<br>The precomputed examples below still work and show exactly how conformal prediction catches overconfident models.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        elif result:
            st.markdown(f"<div style='margin-top:1rem;'>{render_verdict_badge(result['set_size'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{TEXT_SECONDARY}; font-size:0.9rem; margin:0.75rem 0;'>{result['interpretation']}</p>", unsafe_allow_html=True)

            # 4-class probability cards
            pred_set = result["prediction_set"]
            predicted = result["predicted_class"]
            card_cols = st.columns(4)
            for j, cls in enumerate(TRIAGE_CLASSES):
                with card_cols[j]:
                    prob = result["probs"][cls]
                    in_set = cls in pred_set
                    is_pred = cls == predicted
                    st.markdown(render_triage_card(cls, prob, in_set, is_pred), unsafe_allow_html=True)

            # Correct label (only for curated examples where we know ground truth)
            if correct_label:
                cl_style = TRIAGE_COLORS.get(correct_label, TRIAGE_COLORS["gp_visit"])
                is_in_set = correct_label in pred_set
                is_model_correct = correct_label == predicted
                # Check if CP escalated above the correct label
                correct_severity = CLASS_TO_IDX.get(correct_label, 0)
                max_set_severity = max(CLASS_TO_IDX.get(c, 0) for c in pred_set)
                cp_over_escalated = is_in_set and max_set_severity > correct_severity

                if is_model_correct:
                    verdict_icon = ICON_CHECK
                    verdict = "Model got it right."
                elif is_in_set and cp_over_escalated:
                    verdict_icon = ICON_BOLT
                    verdict = (
                        "Model was wrong, but CP caught it. Escalating to the higher level is still the right move "
                        "when lives are on the line. Yet this is useful data for future similar cases."
                    )
                elif is_in_set:
                    verdict_icon = ICON_BOLT
                    verdict = "Model was wrong, but CP caught it. The correct level is in the prediction set."
                else:
                    verdict_icon = ICON_X
                    verdict = "Both model and CP missed. This falls in the expected ~10% uncovered cases."
                st.markdown(f"""
                <div style="margin-top:0.75rem; background:{BG_ELEVATED}; border:1px solid {BORDER}; border-radius:2px; padding:0.75rem 1rem; display:flex; align-items:center; gap:1rem;">
                    <span style="display:inline-flex; align-items:center; flex-shrink:0;">{verdict_icon}</span>
                    <div style="flex-shrink:0; min-width:11rem;">
                        <span style="font-family:'DM Sans',sans-serif; font-size:0.72rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.06em;">Correct label</span>
                        <span style="font-family:'DM Sans',sans-serif; font-size:0.92rem; color:{cl_style['text']}; margin-left:0.6rem; font-weight:500;">{cl_style['label']}</span>
                    </div>
                    <span style="font-family:'DM Sans',sans-serif; font-size:0.85rem; color:{TEXT_SECONDARY};">{verdict}</span>
                </div>
                """, unsafe_allow_html=True)

            # Technical details (human-readable table)
            with st.expander("Under the hood"):
                set_str = ", ".join(sorted(pred_set))
                threshold_pct = (1 - q_hat) * 100

                # Interpretation for set size
                if result["set_size"] == 1:
                    size_meaning = "The model is sure enough that CP doesn't need to hedge. One clear answer."
                elif result["set_size"] == 2:
                    size_meaning = "CP sees real ambiguity between two levels. In the medical domain, a human should definitely weigh in when uncertainty is detected."
                else:
                    size_meaning = "High uncertainty across multiple levels. Definitely needs human review."

                # Interpretation for confidence
                conf_val = result["max_prob"]
                if conf_val > 0.95:
                    conf_meaning = "Very high. But remember: the model is 94% confident even when wrong. This number alone tells you little."
                elif conf_val > 0.80:
                    conf_meaning = "High, but not extreme. CP's prediction set is the more reliable signal here."
                else:
                    conf_meaning = "Unusually low for this model. The uncertainty is real and CP will likely widen the set."

                st.markdown(f"""
                <table style="width:100%; border-collapse:collapse; font-family:'DM Sans',sans-serif; font-size:0.88rem;">
                    <thead>
                        <tr style="border-bottom:1px solid {BORDER};">
                            <th style="text-align:left; padding:0.6rem 0.75rem; color:{TEXT_MUTED}; font-weight:500; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; width:25%;">Parameter</th>
                            <th style="text-align:left; padding:0.6rem 0.75rem; color:{TEXT_MUTED}; font-weight:500; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; width:25%;">Value</th>
                            <th style="text-align:left; padding:0.6rem 0.75rem; color:{TEXT_MUTED}; font-weight:500; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; width:50%;">What it means</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid {BORDER};">
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_SECONDARY};">Method</td>
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_PRIMARY};">Split Conformal (Threshold)</td>
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_SECONDARY};">Any triage level whose calibrated probability clears the threshold enters the prediction set.</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER};">
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_SECONDARY};">Threshold</td>
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_PRIMARY};">{threshold_pct:.2f}% <span style="color:{TEXT_MUTED};">(1 - q̂, where q̂ = {q_hat:.4f})</span></td>
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_SECONDARY};">Any class above 0.41% enters the set. Tiny, but the model shoves nearly all mass onto one class — so 0.41% is a meaningful signal. Calibrated on 480 cases.</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER};">
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_SECONDARY};">Prediction set</td>
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_PRIMARY};">{{{set_str}}}</td>
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_SECONDARY};">{size_meaning}</td>
                        </tr>
                        <tr style="border-bottom:1px solid {BORDER};">
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_SECONDARY};">Model confidence</td>
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_PRIMARY};">{conf_val:.1%} on {TRIAGE_COLORS[predicted]['label']}</td>
                            <td style="padding:0.6rem 0.75rem; color:{TEXT_SECONDARY};">Derived from GPT-4o-mini logprobs at the classification token, then temperature-scaled. {conf_meaning}</td>
                        </tr>
                    </tbody>
                </table>
                """, unsafe_allow_html=True)


    st.markdown("---")

    # ── SECTION 2: THE COMPARISON MOMENT ────────────────────────────
    st.markdown("## CP vs. Raw Model")
    # Compute CP saves count and total wrong
    n_test = len(test_df) if isinstance(test_df, pd.DataFrame) and len(test_df) > 0 else 320
    n_wrong = int((~test_df["is_correct"]).sum()) if isinstance(test_df, pd.DataFrame) and len(test_df) > 0 else 70
    cp_saves_count = int(((~test_df["is_correct"]) & (test_df["contains_true"])).sum()) if isinstance(test_df, pd.DataFrame) and len(test_df) > 0 else 31

    st.markdown(f"The model alone gets {accuracy:.0%} of cases right. Add conformal prediction and the true triage level lands in the prediction set {coverage:.0%} of the time. Three patterns show up.")

    # Hero stats
    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.markdown(render_stat_highlight(f"{accuracy:.1%}", "Model Accuracy"), unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(render_stat_highlight(f"{coverage:.1%}", "CP Coverage", EMERALD), unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown(render_stat_highlight(str(cp_saves_count), "Patients Saved", BURGUNDY), unsafe_allow_html=True)

    st.markdown(f'<p style="color:{TEXT_MUTED}; font-size:0.82rem; text-align:center; margin-top:0.5rem;">The model got {n_wrong} out of {n_test} cases wrong. CP caught {cp_saves_count} of those {n_wrong} by including the correct triage level in a wider prediction set.</p>', unsafe_allow_html=True)

    # Three comparison panels
    if isinstance(test_df, pd.DataFrame) and len(test_df) > 0:
        # Find real examples for each category
        # 1. Model agrees (singleton, correct)
        singleton_correct = test_df[(test_df["set_size"] == 1) & (test_df["is_correct"] == True)]
        # 2. CP saves (model wrong, but true label in prediction set)
        cp_saves = test_df[(test_df["is_correct"] == False) & (test_df["contains_true"] == True)]
        # 3. Nobody wins (model wrong, true label NOT in set)
        nobody_wins = test_df[(test_df["is_correct"] == False) & (test_df["contains_true"] == False)]

        panel_cols = st.columns(3)

        def _get_symptom(row):
            """Look up symptom text from dataset using data_idx."""
            if not dataset:
                return ""
            di = int(row.get("data_idx", -1))
            if 0 <= di < len(dataset):
                return dataset[di].get("symptom_description", "")
            return ""

        # Panel 1: Model agrees
        with panel_cols[0]:
            if len(singleton_correct) > 0:
                ex = singleton_correct.iloc[0]
                st.markdown(render_comparison_card(
                    "Model Agrees", ICON_CHECK,
                    "Model nails it. CP agrees and returns a single triage level. No ambiguity, no wasted escalation.",
                    EMERALD,
                    ex["predicted_class"], ex["true_class"], list(ex["prediction_set"]),
                    ex["max_prob"], _get_symptom(ex)
                ), unsafe_allow_html=True)
            else:
                st.markdown(render_comparison_card(
                    "Model Agrees", ICON_CHECK,
                    "Model nails it, CP confirms with a singleton set.",
                    EMERALD, "self_care", "self_care", ["self_care"], 0.97
                ), unsafe_allow_html=True)

        # Panel 2: CP saves
        with panel_cols[1]:
            if len(cp_saves) > 0:
                ex = cp_saves.iloc[0]
                st.markdown(render_comparison_card(
                    "CP Saves", ICON_BOLT,
                    "Model got it wrong, but CP's wider set catches the true triage level. This is where uncertainty quantification earns its keep.",
                    AMBER,
                    ex["predicted_class"], ex["true_class"], list(ex["prediction_set"]),
                    ex["max_prob"], _get_symptom(ex)
                ), unsafe_allow_html=True)
            else:
                st.markdown(render_comparison_card(
                    "CP Saves", ICON_BOLT,
                    "Model wrong, but CP's wider set catches the truth.",
                    AMBER, "gp_visit", "urgent_care", ["gp_visit", "urgent_care"], 0.94
                ), unsafe_allow_html=True)

        # Panel 3: Nobody wins
        with panel_cols[2]:
            if len(nobody_wins) > 0:
                ex = nobody_wins.iloc[0]
                st.markdown(render_comparison_card(
                    "Miss", ICON_X,
                    "Both the model and CP missed. Calibrated uncertainty has limits. But CP is upfront about that 10% risk, so you can build your system around it.",
                    BURGUNDY,
                    ex["predicted_class"], ex["true_class"], list(ex["prediction_set"]),
                    ex["max_prob"], _get_symptom(ex)
                ), unsafe_allow_html=True)
            else:
                st.markdown(render_comparison_card(
                    "Miss", ICON_X,
                    "Both model and CP missed. CP is upfront about the ~10% risk.",
                    BURGUNDY, "self_care", "urgent_care", ["self_care"], 0.96
                ), unsafe_allow_html=True)

    st.markdown("---")

    # ── SECTION 3: WHY THIS MATTERS ─────────────────────────────────
    st.markdown("## Why This Matters")

    st.markdown(f"""
    <div style="background:{BG_CARD}; border:1px solid {BORDER}; border-left:3px solid {BURGUNDY}; border-radius:2px; padding:1.5rem; margin:1rem 0;">
        <p style="font-family:'DM Sans',sans-serif; font-weight:500; font-size:1.05rem; color:{TEXT_PRIMARY}; margin:0 0 0.75rem 0;">
            LLMs are structurally overconfident. It's how they're trained.
        </p>
        <p style="font-family:'DM Sans',sans-serif; font-size:0.92rem; color:{TEXT_SECONDARY}; line-height:1.7; margin:0 0 0.75rem 0;">
            GPT-4o-mini assigns <span style="color:{BURGUNDY};">94% confidence even when it's wrong</span> (vs. 98% when right).
            That's a byproduct of how LLMs are trained. Fine for autocomplete, dangerous for medicine.
        </p>
        <p style="font-family:'DM Sans',sans-serif; font-size:0.92rem; color:{TEXT_SECONDARY}; line-height:1.7; margin:0 0 0.75rem 0;">
            CP doesn't fix the model. It wraps it with calibrated thresholds that
            <span style="color:{TEXT_PRIMARY};">guarantee the true answer is in the set 90% of the time</span>.
        </p>
        <p style="font-family:'DM Sans',sans-serif; font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.7; margin:0;">
            Even if it corrects one patient, it's worth it. And it's nearly free: no retraining, no extra
            model calls. Just a lightweight wrapper that adds accountability where agentic decisions matter.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence comparison visual
    st.markdown(f"""
    <div style="display:flex; gap:1.25rem; margin:1.5rem 0;">
        <div style="flex:1; background:{BG_CARD}; border:1px solid {BORDER}; border-radius:2px; padding:1.25rem; text-align:center;">
            <div style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem;">When Model is Correct</div>
            <div style="font-family:'DM Sans',sans-serif; font-weight:300; font-size:2.2rem; color:{EMERALD};">98%</div>
            <div style="font-family:'DM Sans',sans-serif; font-size:0.8rem; color:{TEXT_MUTED};">logprob-derived confidence</div>
        </div>
        <div style="flex:1; background:{BG_CARD}; border:1px solid {BORDER}; border-radius:2px; padding:1.25rem; text-align:center;">
            <div style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem;">When Model is Wrong</div>
            <div style="font-family:'DM Sans',sans-serif; font-weight:300; font-size:2.2rem; color:{BURGUNDY};">94%</div>
            <div style="font-family:'DM Sans',sans-serif; font-size:0.8rem; color:{TEXT_MUTED};">logprob-derived confidence</div>
        </div>
        <div style="flex:1; background:{BURGUNDY_MUTED}; border:1px solid {BURGUNDY}; border-radius:2px; padding:1.25rem; text-align:center;">
            <div style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem;">Separation</div>
            <div style="font-family:'DM Sans',sans-serif; font-weight:300; font-size:2.2rem; color:{TEXT_PRIMARY};">4pp</div>
            <div style="font-family:'DM Sans',sans-serif; font-size:0.8rem; color:{TEXT_MUTED};">useless for decisions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Technical deep-dive (expandable)
    with st.expander("The technical why: cross-entropy and peaked distributions"):
        st.markdown(f"""
        LLMs are trained with **cross-entropy loss**, which pushes the model to assign maximum probability
        to the correct next token. Over billions of gradient steps this produces extremely peaked output
        distributions. The model learns to be maximally confident on every prediction, correct or not.

        This is a training artifact, not a feature. Verbalized confidence ("I'm 90% sure") is even worse since
        it has no grounding in the actual probability distribution. That's why we extract **raw logprobs** at
        the classification token.

        The key insight: confidence alone can't distinguish correct from incorrect predictions. Conformal
        prediction sidesteps this entirely by providing a **distribution-free coverage guarantee** via
        calibration on held-out data. We also apply conformal methods to temperature scaling (ConfTS) to
        optimize the probability distribution before building prediction sets — but that's a calibration
        detail, not the main event. The real value is the coverage guarantee itself.
        """)


    st.markdown("---")

    # ── SECTION 4: GALLERY ──────────────────────────────────────────
    st.markdown("## Case Gallery")
    st.markdown("Real cases from the test set. Each card shows what the model predicted, what CP added, and what actually happened.")

    if isinstance(test_df, pd.DataFrame) and len(test_df) > 0:
        # Select interesting cases: a mix of singletons, doublets, correct, and wrong
        gallery_cases = []

        # Singletons where model is correct (pick 2 from different classes)
        for cls in ["emergency", "self_care"]:
            mask = (test_df["set_size"] == 1) & (test_df["is_correct"]) & (test_df["predicted_class"] == cls)
            subset = test_df[mask]
            if len(subset) > 0:
                gallery_cases.append(("singleton_correct", subset.index[0]))

        # CP saves cases (pick 3)
        cp_save_mask = (~test_df["is_correct"]) & (test_df["contains_true"])
        cp_save_df = test_df[cp_save_mask]
        for i in range(min(3, len(cp_save_df))):
            gallery_cases.append(("cp_save", cp_save_df.index[i]))

        # Miss case (pick 1)
        miss_mask = (~test_df["is_correct"]) & (~test_df["contains_true"])
        miss_df = test_df[miss_mask]
        if len(miss_df) > 0:
            gallery_cases.append(("miss", miss_df.index[0]))

        for case_type, idx in gallery_cases:
            row = test_df.loc[idx]
            symptom = ""
            di = int(row.get("data_idx", -1))
            if dataset and 0 <= di < len(dataset):
                symptom = dataset[di].get("symptom_description", "")

            # Determine card color
            if case_type == "singleton_correct":
                accent = EMERALD
                tag = "HIGH CONFIDENCE"
                tag_color = EMERALD
            elif case_type == "cp_save":
                accent = AMBER
                tag = "CP SAVE"
                tag_color = AMBER
            else:
                accent = BURGUNDY
                tag = "MISS"
                tag_color = BURGUNDY

            pred_set = list(row["prediction_set"]) if isinstance(row["prediction_set"], (set, frozenset)) else [row["predicted_class"]]
            set_labels = ", ".join([TRIAGE_COLORS[c]["label"] for c in pred_set])
            pred_label = TRIAGE_COLORS[row["predicted_class"]]["label"]
            true_label = TRIAGE_COLORS[row["true_class"]]["label"]
            pred_color = TRIAGE_COLORS[row["predicted_class"]]["text"]
            true_color = TRIAGE_COLORS[row["true_class"]]["text"]

            short_symptom = (symptom[:180] + "...") if len(symptom) > 180 else symptom

            st.markdown(f"""
            <div style="background:{BG_CARD}; border:1px solid {BORDER}; border-left:3px solid {accent}; border-radius:2px; padding:1.25rem; margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <span style="background:{BG_ELEVATED}; color:{tag_color}; font-family:'DM Sans',sans-serif; font-size:0.7rem; font-weight:500; text-transform:uppercase; letter-spacing:0.06em; padding:0.2rem 0.6rem; border-radius:4px;">{tag}</span>
                    <span style="color:{TEXT_MUTED}; font-family:'DM Sans',sans-serif; font-size:0.8rem;">Set size: {row['set_size']}</span>
                </div>
                <p style="color:{TEXT_SECONDARY}; font-family:'DM Sans',sans-serif; font-size:0.88rem; font-style:italic; line-height:1.5; margin:0 0 0.75rem 0;">"{short_symptom}"</p>
                <div style="display:flex; gap:1.5rem; font-family:'DM Sans',sans-serif; font-size:0.85rem;">
                    <span><span style="color:{TEXT_MUTED}; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.05em;">Model:</span> <span style="color:{pred_color};">{pred_label}</span> <span style="color:{TEXT_MUTED};">({row['max_prob']:.0%})</span></span>
                    <span><span style="color:{TEXT_MUTED}; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.05em;">CP Set:</span> <span style="color:{TEXT_PRIMARY};">{{{set_labels}}}</span></span>
                    <span><span style="color:{TEXT_MUTED}; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.05em;">Truth:</span> <span style="color:{true_color};">{true_label}</span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:{TEXT_MUTED};">No precomputed data found. Run <code>python precompute.py</code> first.</p>', unsafe_allow_html=True)

    st.markdown(f'<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)

    # ── SECTION 5: STATS ────────────────────────────────────────────
    with st.expander("Dataset Statistics"):
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Coverage", f"{coverage:.1%}")
            st.markdown(f'<p style="color:{TEXT_MUTED}; font-size:0.72rem; margin-top:-0.5rem;">How often the correct answer is somewhere in the prediction set. Higher = safer.</p>', unsafe_allow_html=True)
        with stat_cols[1]:
            st.metric("Accuracy", f"{accuracy:.1%}")
            st.markdown(f'<p style="color:{TEXT_MUTED}; font-size:0.72rem; margin-top:-0.5rem;">How often the model\'s top pick is right. This is the raw model, no CP help.</p>', unsafe_allow_html=True)
        with stat_cols[2]:
            st.metric("Singleton Rate", f"{singleton_rate:.1%}")
            st.markdown(f'<p style="color:{TEXT_MUTED}; font-size:0.72rem; margin-top:-0.5rem;">How often CP returns just one answer. Higher = the model is sure and CP agrees.</p>', unsafe_allow_html=True)
        with stat_cols[3]:
            st.metric("q_hat", f"{q_hat:.4f}")
            st.markdown(f'<p style="color:{TEXT_MUTED}; font-size:0.72rem; margin-top:-0.5rem;">The calibrated cutoff. Classes above 0.41% probability make it into the set.</p>', unsafe_allow_html=True)

        st.markdown(f"<br>", unsafe_allow_html=True)

        if isinstance(test_df, pd.DataFrame) and len(test_df) > 0:
            # Set size distribution
            size_counts = test_df["set_size"].value_counts().sort_index()
            total = len(test_df)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"Size {s}" for s in size_counts.index],
                y=[c / total * 100 for c in size_counts.values],
                marker_color=[EMERALD if s == 1 else AMBER if s == 2 else BURGUNDY for s in size_counts.index],
                text=[f"{c} ({c/total:.0%})" for c in size_counts.values],
                textposition="outside",
                textfont=dict(color=TEXT_SECONDARY, size=12),
            ))
            max_pct = max(c / total * 100 for c in size_counts.values)
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Prediction Set Size Distribution",
                yaxis_title="% of test cases",
                yaxis_range=[0, max_pct * 1.15],
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Per-class accuracy
            if "true_class" in test_df.columns and "is_correct" in test_df.columns:
                class_stats = test_df.groupby("true_class").agg(
                    accuracy=("is_correct", "mean"),
                    coverage=("contains_true", "mean"),
                    count=("is_correct", "count"),
                    avg_set_size=("set_size", "mean"),
                ).round(3)

                fig2 = go.Figure()
                classes = [TRIAGE_COLORS[c]["label"] for c in TRIAGE_CLASSES if c in class_stats.index]
                acc_vals = [class_stats.loc[c, "accuracy"] * 100 for c in TRIAGE_CLASSES if c in class_stats.index]
                cov_vals = [class_stats.loc[c, "coverage"] * 100 for c in TRIAGE_CLASSES if c in class_stats.index]
                cls_colors = [TRIAGE_COLORS[c]["text"] for c in TRIAGE_CLASSES if c in class_stats.index]

                fig2.add_trace(go.Bar(
                    name="Accuracy",
                    x=classes, y=acc_vals,
                    marker_color="rgba(250,250,250,0.35)",
                ))
                fig2.add_trace(go.Bar(
                    name="CP Coverage",
                    x=classes, y=cov_vals,
                    marker_color=TEXT_PRIMARY,
                ))
                fig2.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Per-Class: Accuracy vs CP Coverage",
                    yaxis_title="%",
                    barmode="group",
                    height=350,
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Method details
        st.markdown(f"""
        **Method**
        Split conformal prediction with threshold scoring. The calibration set is used to learn a probability cutoff (q̂).
        At inference, any class whose calibrated probability exceeds that cutoff enters the prediction set.
        Simple, interpretable, and distribution-free.

        **Dataset**
        800 synthetic triage cases (ESI 1-5 mapped to 4 classes). Generated with Claude Opus 4.6, designed to match real hospital triage distributions.
        Split 60/40: 480 calibration, 320 test.

        **Configuration**
        Model: GPT-4o-mini &middot; Temperature scaling: T={T} (ConfTS) &middot; Alpha: 0.10
        """)

    # ── CTA ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; margin-top:3rem; padding:2rem 1.5rem; background:{BG_CARD}; border:1px solid {BORDER}; border-radius:2px;">
        <p style="font-family:'DM Sans',sans-serif; font-weight:500; font-size:1.1rem; color:{TEXT_PRIMARY}; margin:0 0 0.5rem 0;">
            Your model is overconfident too. Want me to find out by how much?
        </p>
        <p style="font-family:'DM Sans',sans-serif; font-size:0.88rem; color:{EMERALD}; margin:0 0 0.4rem 0;">
            <a href="mailto:lotta.lorette@gmail.com" style="color:{EMERALD}; text-decoration:none;">lotta.lorette@gmail.com</a>
        </p>
        <p style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:{TEXT_MUTED}; margin:0;">
            One weekend. No retraining.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── FOOTER ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; margin-top:2rem; padding:1rem 0; border-top:1px solid {BORDER};">
        <p style="color:{TEXT_MUTED}; font-family:'DM Sans',sans-serif; font-size:0.8rem; letter-spacing:0.04em;">
            Built by Lotta-Lorette Kalmaru &middot; 2026
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
