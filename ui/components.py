"""
OpticBin — Premium 3-Bin UI Components
==========================================
Glassmorphism hero cards, animated checklists, and rich visual guidance
for the biodegradable / non-biodegradable / e-waste home sorting system.
"""

from __future__ import annotations

import streamlit as st

from config.settings import (
    BIN_COLORS,
    BIN_ICONS,
    CLASS_LABELS,
    DEVICE,
    LATENCY_TARGET_MS,
    NUM_CLASSES,
    WASTE_METADATA,
)
from ui.styles import IMAGE_MODE, WEBCAM_MODE

PYTORCH_FRAMEWORK = "PyTorch + Grad-CAM"
ONNX_FRAMEWORK = "ONNX Runtime (Fast)"

# ──────────────────────────────────────────────
# Theme mapping for CSS classes
# ──────────────────────────────────────────────
_HERO_CLASS_MAP = {
    "biodegradable": "ob-hero-bio",
    "non-biodegradable": "ob-hero-nonbio",
    "e-waste": "ob-hero-ewaste",
}

_CHECK_ICON_MAP = {
    "biodegradable": ("✓", "ob-check-icon-bio"),
    "non-biodegradable": ("✓", "ob-check-icon-nonbio"),
    "e-waste": ("⚠", "ob-check-icon-ewaste"),
}


def render_header() -> None:
    """Render the premium gradient header with 3-bin legend."""
    st.markdown(
        """
        <div class="ob-header">
            <div class="ob-logo-row">
                <span class="ob-logo-icon">♻️</span>
                <h1 class="ob-title">OpticBin</h1>
            </div>
            <p class="ob-subtitle">AI-Powered Home Waste Sorting — Scan any item for instant bin guidance</p>
            <div class="ob-bin-legend">
                <div class="ob-legend-item">🟢 <span>Green Bin — Biodegradable</span></div>
                <div class="ob-legend-item">🔵 <span>Blue Bin — Non-Biodegradable</span></div>
                <div class="ob-legend-item">🟠 <span>E-Waste Collection</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Streamlined sidebar with scan method and system info."""
    with st.sidebar:
        st.markdown("### 📷 Scan Method")
        input_mode = st.radio(
            "Select input mode",
            [WEBCAM_MODE, IMAGE_MODE],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("##### System")
        st.caption(f"🧠 **Model:** Fine-Tuned Unified YOLOv8")
        st.caption(f"⚡ **Hardware:** `{'GPU (CUDA)' if DEVICE == 'cuda' else 'CPU'}`")
        st.caption(f"🏷️ **Categories:** `3 Degradability Bins`")

    return input_mode


def render_engine_status(engine: object) -> None:
    """Subtle status indicator in sidebar."""
    using_finetuned = getattr(engine, "using_finetuned_weights", False)
    if using_finetuned:
        st.sidebar.caption("✅ Active: Fine-tuned waste detection weights")
    else:
        st.sidebar.caption("⚠️ Active: Base weights (uncalibrated)")


def render_untrained_warning(engine: object) -> None:
    """Warns if the model is operating without fine-tuned weights."""
    if getattr(engine, "using_finetuned_weights", False):
        return
    st.info(
        "📌 Using standard base weights. For production sorting accuracy, "
        "ensure fine-tuned weights are present in models/weights/."
    )


def render_hero_bin_card(result: dict, latency_ms: float) -> None:
    """
    Renders the premium glassmorphism hero card showing which bin the item
    should go into, with animated entrance and rich metadata pills.
    """
    label = result.get("class_label", "non-biodegradable").lower()
    meta = WASTE_METADATA.get(label, WASTE_METADATA["non-biodegradable"])
    confidence = result.get("confidence", 0.0) * 100
    disposal = meta.get("disposal", "Check local disposal guidelines")
    bin_icon = meta.get("bin_icon", "♻️")
    bin_color = meta.get("bin_color", "#6B7280")
    category = meta.get("category", "Unknown")
    decomposition = meta.get("decomposition", "Varies")

    hero_class = _HERO_CLASS_MAP.get(label, "ob-hero-nonbio")

    # Category display name
    display_name = {
        "biodegradable": "Biodegradable",
        "non-biodegradable": "Non-Biodegradable",
        "e-waste": "E-Waste",
    }.get(label, label.title())

    # Verified badge
    verified_pill = ""
    if result.get("corrected_by_gemini") or result.get("verified_by_gemini"):
        verified_pill = '<span class="ob-pill ob-pill-verified">✨ <b>AI Supervisor Verified</b></span>'

    st.markdown(
        f"""
        <div class="ob-hero {hero_class}">
            <div class="ob-hero-kicker">Recommended Disposal</div>
            <div class="ob-hero-bin-row">
                <span class="ob-hero-bin-icon">{bin_icon}</span>
                <span class="ob-hero-bin-name" style="color:{bin_color};">{display_name}</span>
            </div>
            <div class="ob-hero-disposal">{disposal}</div>
            <div class="ob-pills-row">
                <span class="ob-pill">🎯 Confidence: <b>{confidence:.1f}%</b></span>
                <span class="ob-pill">⏱️ Scan: <b>{latency_ms:.0f} ms</b></span>
                <span class="ob-pill">🕐 Decomposition: <b>{decomposition}</b></span>
                {verified_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_actionable_checklist(result: dict) -> None:
    """Renders a rich, animated disposal checklist with category-themed icons."""
    label = result.get("class_label", "non-biodegradable").lower()
    meta = WASTE_METADATA.get(label, {})
    if not meta:
        return

    tips = meta.get("tips", ["Follow local disposal guidelines."])
    examples = meta.get("examples", "")
    env_impact = meta.get("environmental_impact", "")
    check_icon, check_class = _CHECK_ICON_MAP.get(label, ("✓", "ob-check-icon-nonbio"))

    # Build checklist HTML
    tips_html = ""
    for tip in tips:
        tips_html += f"""
        <div class="ob-checklist-item">
            <span class="ob-check-icon {check_class}">{check_icon}</span>
            <div>{tip}</div>
        </div>
        """

    # Build examples tags
    examples_html = ""
    if examples:
        tags = [e.strip() for e in examples.split(",")]
        tag_items = "".join(f'<span class="ob-example-tag">{t}</span>' for t in tags[:8])
        examples_html = f"""
        <div style="margin-top:0.8rem;">
            <div style="font-size:0.75rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; opacity:0.5; margin-bottom:0.4rem;">Common Examples</div>
            <div class="ob-examples">{tag_items}</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="ob-card">
            <div class="ob-card-title">♻️ Disposal Instructions</div>
            {tips_html}
            {examples_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Environmental impact card
    if env_impact:
        st.markdown(
            f"""
            <div class="ob-card">
                <div class="ob-card-title">🌍 Environmental Impact</div>
                <div style="font-size:0.9rem; line-height:1.6; opacity:0.85;">{env_impact}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_summary(result: dict, latency_ms: float, compact: bool = False) -> None:
    """Unified wrapper for hero bin card and checklist."""
    render_hero_bin_card(result, latency_ms)
    if not compact:
        render_actionable_checklist(result)


def render_disposal_guidance(result: dict, advisor=None) -> None:
    """Optional AI advisor section."""
    if advisor is None or not getattr(advisor, "is_available", False):
        return

    with st.expander("🤖 AI Recycling Assistant", expanded=False):
        btn_key = f"ai_advice_{result.get('class_label', 'unknown')}"
        if st.button("Get Personalized Recycling Advice", key=btn_key, use_container_width=True):
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


def render_llm_xai_explanation(result: dict, advisor, model_type: str = "YOLOv8") -> None:
    """Explain Grad-CAM visual features."""
    if advisor is None or not getattr(advisor, "is_available", False):
        return

    with st.expander("🔬 Explain Prediction", expanded=False):
        btn_key = f"xai_{result.get('class_label', 'unknown')}"
        if st.button("Explain What The Model Sees", key=btn_key, use_container_width=True):
            with st.spinner("Analyzing features..."):
                stream = advisor.explain_prediction(result, model_type=model_type)
            if stream is not None:
                st.write_stream(_iter_gemini_stream(stream))


def render_active_learning_badge(al_result: dict | None) -> None:
    """Displays a styled agreement/correction badge for active learning results."""
    if al_result is None:
        return

    gemini_label = al_result.get("gemini_label", "unknown")
    agreement = al_result.get("agreement", True)
    gemini_conf = al_result.get("gemini_confidence", 0.0) * 100
    ml_conf = al_result.get("ml_confidence", 0.0)

    # Human-friendly label
    display_label = {
        "biodegradable": "Biodegradable 🟢",
        "non-biodegradable": "Non-Biodegradable 🔵",
        "e-waste": "E-Waste 🟠",
    }.get(gemini_label, gemini_label.title())

    if agreement:
        st.markdown(
            f"""
            <div class="ob-al-badge ob-al-verified">
                <b style="color:#22C55E;">✅ AI Supervisor Verified</b> — Gemini Vision confirmed:
                <b>{display_label}</b> ({gemini_conf:.0f}% confidence)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="ob-al-badge ob-al-corrected">
                <b style="color:#F97316;">🔄 AI Supervisor Correction</b> — Local model was uncertain ({ml_conf:.0f}%).
                Gemini Vision re-classified as <b>{display_label}</b> ({gemini_conf:.0f}% confidence)
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_chart(result: dict) -> None:
    """Renders clean, sorted probability distribution with bin-colored bars."""
    with st.expander("📊 Category Probability Breakdown", expanded=False):
        labels = result.get("class_names", CLASS_LABELS)
        probs = result.get("probabilities", [])
        paired = list(zip(labels, probs))
        paired.sort(key=lambda x: x[1], reverse=True)

        for class_name, prob in paired:
            pct = float(prob) * 100
            icon = BIN_ICONS.get(class_name, "♻️")
            display = {
                "biodegradable": "Biodegradable",
                "non-biodegradable": "Non-Biodegradable",
                "e-waste": "E-Waste",
            }.get(class_name, class_name.title())

            c_name, c_bar = st.columns([1, 3])
            c_name.write(f"{icon} **{display}**")
            c_bar.progress(float(prob), text=f"{pct:.1f}%")


def render_empty_state(title: str, body: str) -> None:
    """Premium dashed placeholder with animated pulse."""
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
    """Sidebar recycling Q&A chatbot."""
    if advisor is None or not getattr(advisor, "is_available", False):
        return

    with st.sidebar.expander("💬 Recycling Q&A Chat", expanded=False):
        if "_chat_history" not in st.session_state:
            st.session_state["_chat_history"] = []
            st.session_state["_chat_display"] = []

        for msg in st.session_state["_chat_display"]:
            prefix = "**You:** " if msg["role"] == "user" else "**Advisor:** "
            st.markdown(f"{prefix}{msg['text']}")

        user_input = st.chat_input("Ask a waste sorting question...")
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


def render_scan_stats() -> None:
    """Render session scan statistics in the sidebar."""
    from ui.state_manager import SessionTracker
    stats = SessionTracker.get_stats()
    if stats["total"] == 0:
        return

    st.sidebar.divider()
    st.sidebar.markdown("##### 📈 Session Stats")

    c1, c2 = st.sidebar.columns(2)
    c1.metric("Scans", stats["total"])
    c2.metric("Recyclable", f"{stats['recyclable_pct']:.0f}%")

    if stats["counts"]:
        for label, count in stats["counts"].items():
            icon = BIN_ICONS.get(label, "♻️")
            display = {
                "biodegradable": "Bio",
                "non-biodegradable": "Non-Bio",
                "e-waste": "E-Waste",
            }.get(label, label[:8])
            st.sidebar.caption(f"{icon} {display}: **{count}**")


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
