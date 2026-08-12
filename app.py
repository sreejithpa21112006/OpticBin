"""
OpticBin — Streamlit Dashboard
================================
Real-time webcam waste classification with dual-column XAI visual
auditing. Identifies waste type and shows biodegradability, disposal
guidance, and environmental impact information.

Launch:
    streamlit run app.py
"""

import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config.settings import (
    CLASS_LABELS,
    NUM_CLASSES,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    LATENCY_TARGET_MS,
    WASTE_METADATA,
    WEIGHTS_DIR,
)
from src.preprocessor import preprocess_frame, preprocess_pil
from src.xai_engine import RecyclingXAIEngine


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="OpticBin — Edge-AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00C9A7, #845EF7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #8892b0;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(132, 94, 247, 0.3);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #00C9A7;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
    }

    /* ── Biodegradability Badge ── */
    .bio-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .bio-yes {
        background: linear-gradient(135deg, #10B98130, #10B98110);
        color: #10B981;
        border: 1px solid #10B98150;
    }
    .bio-no {
        background: linear-gradient(135deg, #EF444430, #EF444410);
        color: #EF4444;
        border: 1px solid #EF444450;
    }
    .bio-mixed {
        background: linear-gradient(135deg, #F59E0B30, #F59E0B10);
        color: #F59E0B;
        border: 1px solid #F59E0B50;
    }

    /* ── Info Panel ── */
    .info-panel {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(132, 94, 247, 0.2);
        margin-top: 0.5rem;
    }
    .info-panel h3 {
        font-family: 'Inter', sans-serif;
        margin-top: 0;
        color: #e2e8f0;
    }
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        font-family: 'Inter', sans-serif;
    }
    .info-row:last-child { border-bottom: none; }
    .info-key {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .info-val {
        color: #e2e8f0;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: right;
        max-width: 60%;
    }

    /* ── Tips ── */
    .tip-item {
        background: rgba(16, 185, 129, 0.08);
        border-left: 3px solid #10B981;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #cbd5e1;
    }

    /* ── Impact Alert ── */
    .impact-box {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #fca5a5;
    }
    .impact-box.low {
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .badge-pass { background: #00C9A720; color: #00C9A7; }
    .badge-warn { background: #FFD93D20; color: #FFD93D; }

    /* ── Webcam metrics bar ── */
    .webcam-metrics {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border: 1px solid rgba(132, 94, 247, 0.2);
        font-family: 'Inter', sans-serif;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper: Render waste info panel
# ──────────────────────────────────────────────
def render_waste_info(result: dict, latency_ms: float):
    """Render the biodegradability and disposal info cards."""
    label = result["class_label"]
    meta = WASTE_METADATA.get(label, {})
    confidence = result["confidence"] * 100

    if not meta:
        return

    # ── Biodegradability determination ──
    bio = meta["biodegradable"]
    if bio is True:
        bio_text = "✅ BIODEGRADABLE"
        bio_class = "bio-yes"
    elif bio is False:
        bio_text = "❌ NON-BIODEGRADABLE"
        bio_class = "bio-no"
    else:
        bio_text = "⚠️ MIXED / VARIES"
        bio_class = "bio-mixed"

    recyclable = meta["recyclable"]
    if recyclable is True:
        recycle_text = "♻️ Yes — Fully Recyclable"
    elif recyclable is False:
        recycle_text = "🚫 No — Landfill / Special Disposal"
    else:
        recycle_text = f"⚠️ {recyclable}"

    # ── Top row: Prediction + Bio badge ──
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.3rem;">{meta['emoji']}</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 900; color: {meta['color']};">
            {label.upper()}
        </div>
        <div style="margin: 0.8rem 0;">
            <span class="bio-badge {bio_class}">{bio_text}</span>
        </div>
        <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.9rem;">
            Confidence: <strong style="color: #00C9A7;">{confidence:.1f}%</strong>
            &nbsp;·&nbsp; Latency: <strong style="color: #845EF7;">{latency_ms:.0f} ms</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Info Panel ──
    st.markdown(f"""
    <div class="info-panel">
        <h3>{meta['disposal_icon']} Disposal & Classification</h3>
        <div class="info-row">
            <span class="info-key">Category</span>
            <span class="info-val">{meta['category']}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Recyclable</span>
            <span class="info-val">{recycle_text}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Disposal Method</span>
            <span class="info-val">{meta['disposal']}</span>
        </div>
        <div class="info-row">
            <span class="info-key">Decomposition Time</span>
            <span class="info-val">⏳ {meta['decomposition']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disposal Tips ──
    st.markdown("")
    st.markdown("##### 💡 Disposal Tips")
    tips_html = ""
    for tip in meta["tips"]:
        tips_html += f'<div class="tip-item">• {tip}</div>'
    st.markdown(tips_html, unsafe_allow_html=True)

    # ── Environmental Impact ──
    impact = meta["environmental_impact"]
    impact_class = "low" if impact.startswith("Low") else ""
    st.markdown(f"""
    <div class="impact-box {impact_class}">
        🌍 <strong>Environmental Impact:</strong> {impact}
    </div>
    """, unsafe_allow_html=True)


def render_webcam_waste_bar(result: dict, latency_ms: float):
    """Compact waste info bar for webcam live mode."""
    label = result["class_label"]
    meta = WASTE_METADATA.get(label, {})
    confidence = result["confidence"] * 100

    bio = meta.get("biodegradable", "Unknown")
    if bio is True:
        bio_text = "✅ Biodegradable"
        bio_color = "#10B981"
    elif bio is False:
        bio_text = "❌ Non-Biodegradable"
        bio_color = "#EF4444"
    else:
        bio_text = "⚠️ Mixed"
        bio_color = "#F59E0B"

    st.markdown(f"""
    <div class="webcam-metrics">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <span style="font-size: 1.5rem; font-weight: 900; color: {meta.get('color', '#fff')};">
                    {meta.get('emoji', '❓')} {label.upper()}
                </span>
            </div>
            <div>
                <span style="color: {bio_color}; font-weight: 700; font-size: 1rem;">{bio_text}</span>
            </div>
            <div>
                <span style="color: #94a3b8;">Confidence:</span>
                <strong style="color: #00C9A7;">{confidence:.1f}%</strong>
            </div>
            <div>
                <span style="color: #94a3b8;">Dispose:</span>
                <strong style="color: #e2e8f0;">{meta.get('disposal', 'N/A')}</strong>
            </div>
            <div>
                <span style="color: #94a3b8;">Latency:</span>
                <strong style="color: #845EF7;">{latency_ms:.0f} ms</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar — Model Controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        "Backbone Architecture",
        options=list(SUPPORTED_MODELS.keys()),
        index=0,
        help="Switch between CNN and ViT architectures",
    )

    st.caption(SUPPORTED_MODELS[model_choice]["description"])

    st.markdown("---")
    st.markdown("## 📊 Performance Targets")
    st.markdown(f"- **Latency:** ≤ {LATENCY_TARGET_MS} ms")
    st.markdown(f"- **Classes:** {NUM_CLASSES} ({', '.join(CLASS_LABELS)})")
    st.markdown(f"- **Input:** 224 × 224 × 3")

    st.markdown("---")
    st.markdown("## 🔧 Input Source")
    input_mode = st.radio(
        "Select input mode",
        ["📷 Webcam (Live)", "🖼️ Image Upload"],
        index=1,
    )

    st.markdown("---")
    st.markdown("## 🌿 Waste Legend")
    for lbl in CLASS_LABELS:
        m = WASTE_METADATA[lbl]
        bio_tag = "🟢" if m["biodegradable"] is True else ("🔴" if m["biodegradable"] is False else "🟡")
        st.markdown(f"{m['emoji']} **{lbl.title()}** {bio_tag}")


# ──────────────────────────────────────────────
# Initialize XAI Engine  (cached per model type)
# ──────────────────────────────────────────────
@st.cache_resource
def load_engine(model_type: str) -> RecyclingXAIEngine:
    """Load and cache the XAI engine for the selected backbone using fine-tuned weights if available."""
    weights_path = os.path.join(WEIGHTS_DIR, f"{model_type}.pt")
    if not os.path.exists(weights_path):
        weights_path = None
    return RecyclingXAIEngine(model_type=model_type, weights_path=weights_path)


# ──────────────────────────────────────────────
# Main Dashboard
# ──────────────────────────────────────────────
st.markdown('<p class="main-header">♻️ OpticBin — Edge-AI Waste Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Real-time waste classification with <strong>Explainable AI</strong> · '
    'Identifies material type · Biodegradable vs Non-biodegradable · Disposal guidance'
    '</p>',
    unsafe_allow_html=True,
)

st.markdown("---")

if "🖼️" in input_mode:
    # ── Image Upload Mode ──
    uploaded_file = st.file_uploader(
        "Upload a waste image for classification",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        input_tensor, rgb_float = preprocess_pil(image)

        # Run inference + XAI
        with st.spinner("🔍 Analyzing waste material..."):
            engine = load_engine(model_choice)
            start_time = time.perf_counter()
            result = engine.predict_and_explain(input_tensor, rgb_float)
            total_latency = (time.perf_counter() - start_time) * 1000

        # ── Two-column layout: Images | Waste Info ──
        col_images, col_info = st.columns([1.2, 1])

        with col_images:
            # Stacked: input image on top, heatmap below
            tab_original, tab_heatmap = st.tabs(["📸 Original Image", "🔍 Grad-CAM Heatmap"])
            with tab_original:
                st.image(image, width="stretch")
            with tab_heatmap:
                st.image(result["heatmap_overlay"], width="stretch")

        with col_info:
            render_waste_info(result, total_latency)

        # ── Probability Distribution ──
        st.markdown("---")
        st.markdown("### 📊 Class Probability Distribution")
        prob_data = {
            f"{WASTE_METADATA[label]['emoji']} {label.title()}": float(prob)
            for label, prob in zip(CLASS_LABELS, result["probabilities"])
        }
        st.bar_chart(prob_data)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #0f172a, #1e293b);
                    border-radius: 16px; border: 1px dashed rgba(132, 94, 247, 0.3); margin: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
            <div style="font-family: 'Inter', sans-serif; font-size: 1.2rem; color: #e2e8f0; font-weight: 600;">
                Upload a waste image or switch to webcam mode
            </div>
            <div style="font-family: 'Inter', sans-serif; color: #94a3b8; margin-top: 0.5rem; font-size: 0.9rem;">
                OpticBin will identify the waste type, tell you if it's <strong style="color: #10B981;">biodegradable</strong>
                or <strong style="color: #EF4444;">non-biodegradable</strong>, and show you exactly how to dispose of it.
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Webcam Mode ──
    st.markdown("### 📷 Live Webcam Feed")
    st.markdown("Point your camera at waste items — OpticBin will identify and classify in real-time.")

    run_webcam = st.checkbox("▶️ Start Webcam Stream", value=False)

    if run_webcam:
        with st.spinner("🔄 Loading model (first time may download weights)..."):
            engine = load_engine(model_choice)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Could not open webcam. Make sure a camera is connected.")
        else:
            col_feed, col_heatmap = st.columns(2)
            feed_placeholder = col_feed.empty()
            heatmap_placeholder = col_heatmap.empty()
            metrics_placeholder = st.empty()
            info_placeholder = st.empty()

            frame_count = 0
            while run_webcam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam.")
                    break

                input_tensor, rgb_float = preprocess_frame(frame)

                start_time = time.perf_counter()
                result = engine.predict_and_explain(input_tensor, rgb_float)
                total_latency = (time.perf_counter() - start_time) * 1000

                # Display frames
                display_frame = cv2.cvtColor(
                    cv2.resize(frame, (448, 448)),
                    cv2.COLOR_BGR2RGB,
                )
                feed_placeholder.image(display_frame, caption="Live Feed", width="stretch")
                heatmap_placeholder.image(
                    result["heatmap_overlay"],
                    caption=f"Grad-CAM — {result['class_label'].upper()} ({result['confidence']*100:.1f}%)",
                    width="stretch",
                )

                # Waste info bar
                with metrics_placeholder.container():
                    render_webcam_waste_bar(result, total_latency)

                frame_count += 1

            cap.release()


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8892b0; font-size: 0.8rem; font-family: Inter, sans-serif;'>"
    "OpticBin v1.0.0 — Edge-AI Waste Classification System — "
    "Fully Offline · PyTorch + ONNX Runtime · Grad-CAM XAI · Biodegradability Analysis"
    "</div>",
    unsafe_allow_html=True,
)
