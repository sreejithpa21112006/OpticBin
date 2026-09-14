"""
Streamlined, user-first UI components for the OpticBin dashboard.
Focuses on clear bin recommendations, clean visual hierarchy, and intuitive action guidance.
"""

from __future__ import annotations

import streamlit as st

from config.settings import (
    CLASS_LABELS,
    DEVICE,
    INPUT_SIZE,
    LATENCY_TARGET_MS,
    NUM_CLASSES,
    SUPPORTED_MODELS,
    WASTE_METADATA,
)
from ui.styles import IMAGE_MODE, WEBCAM_MODE

PYTORCH_FRAMEWORK = "PyTorch + Grad-CAM"
ONNX_FRAMEWORK = "ONNX Runtime (Fast)"


def render_header() -> None:
    """Render a clean, uncluttered application header."""
    st.markdown(
        """
        <div class="ob-header-container">
            <h1 class="ob-title">OpticBin</h1>
            <p class="ob-subtitle">Smart Waste Sorting Assistant - Scan any item for instant recycling and disposal guidance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """
    Streamlined sidebar prioritizing input mode selection with zero model clutter.
    """
    with st.sidebar:
        st.markdown("### Scan Method")
        input_mode = st.radio(
            "Select input mode",
            [WEBCAM_MODE, IMAGE_MODE],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("**Model:** Fine-Tuned Unified YOLOv8")
        st.caption(f"**Hardware:** `{'GPU (CUDA)' if DEVICE == 'cuda' else 'CPU'}`")
        st.caption(f"**Classes:** `6 Material Categories (Roboflow)`")

    return input_mode


def render_engine_status(engine: object) -> None:
    """Subtle status indicator in sidebar."""
    using_finetuned = getattr(engine, "using_finetuned_weights", False)
    if using_finetuned:
        st.sidebar.caption("[Status] Active: Fine-tuned waste detection weights.")
    else:
        st.sidebar.caption("[Status] Active: Base weights.")

    if not using_finetuned:
        st.sidebar.caption("[Status] Using ImageNet base weights (uncalibrated).")


def render_untrained_warning(engine: object) -> None:
    """Warns if the model is operating without fine-tuned weights."""
    if getattr(engine, "using_finetuned_weights", False):
        return
    st.info(
        "Notice: Using standard base weights. For production sorting accuracy, "
        "ensure fine-tuned weights are present in models/weights/."
    )


def render_hero_bin_card(result: dict, latency_ms: float) -> None:
    """
    Renders a prominent, color-coded hero recommendation card that immediately
    informs the user which bin the item belongs to.
    """
    label = result.get("class_label", "plastic").lower()
    meta = WASTE_METADATA.get(label, {})
    confidence = result.get("confidence", 0.0) * 100
    disposal_bin = meta.get("disposal", "General Recycling Bin")
    bio_label = _bio_label(meta)
    theme_class = f"ob-bin-hero-{label}" if label in ["plastic", "paper", "cardboard", "metal", "glass", "biodegradable"] else "ob-bin-hero-plastic"

    corrected_pill = ""
    if result.get("corrected_by_gemini") or result.get("verified_by_gemini"):
        corrected_pill = '<span class="ob-pill" style="border-color:#10B981; color:#10B981;">Source: <b>AI Supervisor Verified</b></span>'

    st.markdown(
        f"""
        <div class="ob-bin-hero {theme_class}">
            <div class="ob-bin-kicker">Recommended Disposal Destination</div>
            <div class="ob-bin-title">{disposal_bin.upper()}</div>
            <div class="ob-bin-meta-row">
                <span class="ob-pill">Material: <b>{label.title()}</b></span>
                <span class="ob-pill">Confidence: <b>{confidence:.1f}%</b></span>
                <span class="ob-pill">Type: <b>{bio_label}</b></span>
                <span class="ob-pill">Scan Time: <b>{latency_ms:.0f} ms</b></span>
                {corrected_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_actionable_checklist(result: dict) -> None:
    """Renders a clean, actionable preparation and handling checklist."""
    label = result.get("class_label", "plastic").lower()
    meta = WASTE_METADATA.get(label, {})
    if not meta:
        return

    tips = meta.get("tips", ["Empty and rinse containers before discarding."])
    decomposition = meta.get("decomposition", "Varies")
    recyclable = meta.get("recyclable", True)

    recycle_status = "Fully Recyclable" if recyclable is True else "Check local recycling facility rules"

    st.markdown(
        f"""
        <div class="ob-checklist">
            <div style="font-weight:700; font-size:1rem; margin-bottom:0.6rem;">Disposal Instructions</div>
            <div class="ob-checklist-item">
                <span class="ob-checklist-bullet">[x]</span>
                <div><b>Recyclability:</b> {recycle_status}</div>
            </div>
            <div class="ob-checklist-item">
                <span class="ob-checklist-bullet">[x]</span>
                <div><b>Preparation:</b> {tips[0] if tips else 'Empty contents thoroughly.'}</div>
            </div>
            <div class="ob-checklist-item">
                <span class="ob-checklist-bullet">[x]</span>
                <div><b>Environmental Impact:</b> Estimated decomposition time: {decomposition}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_summary(result: dict, latency_ms: float, compact: bool = False) -> None:
    """Unified wrapper for hero bin card and checklist."""
    render_hero_bin_card(result, latency_ms)
    render_actionable_checklist(result)


def render_disposal_guidance(result: dict, advisor=None) -> None:
    """Optional AI advisor section."""
    if advisor is None or not getattr(advisor, "is_available", False):
        return

    with st.expander("AI Recycling Assistant", expanded=False):
        btn_key = f"ai_advice_{result.get('class_label', 'unknown')}"
        if st.button("Generate Detailed Recycling Steps", key=btn_key):
            with st.spinner("Analyzing material..."):
                stream = advisor.stream_recycling_advice(result)
            if stream is not None:
                st.write_stream(_iter_gemini_stream(stream))
            else:
                st.info("AI assistant is temporarily busy. Please try again.")


def render_heatmap(heatmap, engine: object, caption: str | None = None) -> None:
    """Render explainable AI heatmap if available."""
    if heatmap is None or not getattr(engine, "supports_gradcam", True):
        return
    st.image(heatmap, caption=caption or "Visual Attention Heatmap", width="stretch")


def render_llm_xai_explanation(result: dict, advisor, model_type: str = "EfficientNetV2") -> None:
    """Explain Grad-CAM visual features."""
    if advisor is None or not getattr(advisor, "is_available", False):
        return

    with st.expander("Explain Prediction", expanded=False):
        btn_key = f"xai_{result.get('class_label', 'unknown')}"
        if st.button("Explain Model Focus", key=btn_key):
            with st.spinner("Analyzing features..."):
                stream = advisor.explain_prediction(result, model_type=model_type)
            if stream is not None:
                st.write_stream(_iter_gemini_stream(stream))


def render_active_learning_badge(al_result: dict | None) -> None:
    """Displays agreement indicator if active learning cross-check ran."""
    if al_result is None:
        return

    gemini_label = al_result.get("gemini_label", "unknown")
    agreement = al_result.get("agreement", True)
    gemini_conf = al_result.get("gemini_confidence", 0.0) * 100
    ml_conf = al_result.get("ml_confidence", 0.0)

    if agreement:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981; padding: 0.5rem 0.8rem; border-radius: 4px; margin-bottom: 0.8rem; font-size: 0.88rem; color: #D1D5DB;">
                <b style="color:#10B981;">[AI Supervisor Verified]</b> Gemini Vision confirmed material as <b>{gemini_label.title()} ({gemini_conf:.0f}% confidence)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border-left: 4px solid #10B981; padding: 0.5rem 0.8rem; border-radius: 4px; margin-bottom: 0.8rem; font-size: 0.88rem; color: #D1D5DB;">
                <b style="color:#10B981;">[AI Supervisor Correction]</b> Local detection was uncertain ({ml_conf:.0f}%). Gemini Vision cross-checked and verified the item as <b>{gemini_label.title()} ({gemini_conf:.0f}% confidence)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_chart(result: dict) -> None:
    """Renders clean, sorted probability distribution bars."""
    with st.expander("Material Probability Breakdown", expanded=False):
        labels = result.get("class_names", CLASS_LABELS)
        probs = result.get("probabilities", [])
        paired = list(zip(labels, probs))
        paired.sort(key=lambda x: x[1], reverse=True)

        for class_name, prob in paired:
            pct = float(prob) * 100
            c_name, c_bar = st.columns([1, 3])
            c_name.write(f"**{class_name.title()}**")
            c_bar.progress(float(prob), text=f"{pct:.1f}%")


def render_empty_state(title: str, body: str) -> None:
    """Clean dashed placeholder."""
    st.markdown(
        f"""
        <div class="ob-empty-state">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recycling_chatbot(advisor) -> None:
    """Sidebar recycling Q&A chatbot (collapsed under an expander to keep sidebar tidy)."""
    if advisor is None or not getattr(advisor, "is_available", False):
        return

    with st.sidebar.expander("Recycling Q&A Chat", expanded=False):
        if "_chat_history" not in st.session_state:
            st.session_state["_chat_history"] = []
            st.session_state["_chat_display"] = []

        for msg in st.session_state["_chat_display"]:
            prefix = "User: " if msg["role"] == "user" else "Advisor: "
            st.markdown(f"**{prefix}** {msg['text']}")

        user_input = st.chat_input("Ask a waste question...")
        if user_input:
            st.session_state["_chat_display"].append({"role": "user", "text": user_input})
            stream = advisor.chat(user_input, st.session_state["_chat_history"])
            if stream is not None:
                response_text = "".join(chunk.text for chunk in stream if hasattr(chunk, "text"))
                st.session_state["_chat_history"].extend([
                    {"role": "user", "parts": [user_input]},
                    {"role": "model", "parts": [response_text]},
                ])
                st.session_state["_chat_display"].append({"role": "model", "text": response_text})
            st.rerun()


def _iter_gemini_stream(stream):
    """Yield text chunks from streaming Gemini responses."""
    try:
        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text
    except Exception:
        return


def _bio_label(meta: dict) -> str:
    bio = meta.get("biodegradable")
    if bio is True:
        return "Biodegradable"
    if bio is False:
        return "Non-biodegradable"
    return "Recyclable Material"
