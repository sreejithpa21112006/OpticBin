"""Reusable, functional Streamlit components for the OpticBin dashboard."""

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


def render_header() -> None:
    st.title("OpticBin")
    st.caption(
        "Offline Edge-AI Waste Classifier — Instant disposal guidance with explainable AI."
    )


PYTORCH_FRAMEWORK = "PyTorch + Grad-CAM"
ONNX_FRAMEWORK = "ONNX Runtime (Fast)"


def render_sidebar() -> tuple[str, str, str]:
    """Render model, framework, and input controls. Returns `(model_choice, framework, input_mode)`."""
    with st.sidebar:
        st.header("Workspace Controls")
        model_choice = st.selectbox(
            "Backbone Architecture",
            options=list(SUPPORTED_MODELS.keys()),
            index=0,
            help="Switch between CNN (EfficientNetV2) and ViT (MobileViT) architectures.",
        )
        st.caption(SUPPORTED_MODELS[model_choice]["description"])

        framework = st.radio(
            "Inference Engine",
            [PYTORCH_FRAMEWORK, ONNX_FRAMEWORK],
            index=0,
            help="PyTorch for Grad-CAM heatmaps or ONNX Runtime for minimum latency.",
        )

        input_mode = st.radio(
            "Input Source",
            [IMAGE_MODE, WEBCAM_MODE],
            index=0,
        )

        st.divider()
        st.subheader("Performance Specs")
        device_display = "GPU (CUDA)" if DEVICE == "cuda" else "CPU"
        st.markdown(
            f"- **Compute Device:** `{device_display}`\n"
            f"- **Latency Target:** ≤ {LATENCY_TARGET_MS} ms\n"
            f"- **Supported Classes:** {NUM_CLASSES}\n"
            f"- **Resolution:** {INPUT_SIZE[0]} × {INPUT_SIZE[1]}"
        )

        st.divider()
        st.subheader("Taxonomy Reference")
        for label in CLASS_LABELS:
            meta = WASTE_METADATA[label]
            st.markdown(
                f"**{label.title()}**: {_bio_short(meta)}"
            )

    return model_choice, framework, input_mode


def render_engine_status(model_type: str, framework: str, engine: object) -> None:
    is_onnx = getattr(engine, "__class__", None).__name__ == "ONNXInferenceEngine"
    using_finetuned = getattr(engine, "using_finetuned_weights", False)
    provider = getattr(engine, "provider", None)

    if "ONNX" in framework:
        if is_onnx:
            provider_name = "GPU (CUDA)" if "CUDA" in str(provider) else "CPU"
            st.sidebar.success(f"ONNX Engine Active ({model_type}) — Running on {provider_name}")
        else:
            st.sidebar.warning(
                f"No `.onnx` checkpoint found for `{model_type}`. "
                "Falling back to PyTorch."
            )
    else:
        dev_name = "GPU (CUDA)" if DEVICE == "cuda" else "CPU"
        if using_finetuned:
            st.sidebar.success(f"PyTorch Engine Active ({model_type}) — Running on {dev_name}")
        else:
            st.sidebar.warning(
                f"No fine-tuned `{model_type}.pt` found. "
                "Using ImageNet-pretrained weights — waste predictions are not reliable. "
                "Run `python train.py` first."
            )
            if not getattr(engine, "supports_gradcam", True):
                st.sidebar.info("Grad-CAM heatmaps are available only with the PyTorch engine.")



def render_untrained_warning(engine: object) -> None:
    if getattr(engine, "using_finetuned_weights", False):
        return
    st.warning(
        "This session is **not using a fine-tuned waste model**. "
        "Predictions come from an ImageNet-pretrained backbone with a new 5-class head, "
        "so labels and disposal guidance can be wrong. Train with `python train.py` "
        "and keep the checkpoint in `models/weights/`."
    )


def render_heatmap(heatmap, engine: object, caption: str | None = None) -> None:
    if heatmap is None or not getattr(engine, "supports_gradcam", True):
        st.info(
            "Grad-CAM is available only with **PyTorch + Grad-CAM**. "
            "ONNX Runtime classifies faster but does not produce a heatmap."
        )
        return
    st.image(heatmap, caption=caption, width="stretch")


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ob-empty">
            <p class="ob-kicker">Status</p>
            <h4>{title}</h4>
            <p style="opacity:0.75; font-size:0.9rem; margin-top:0.3rem;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_summary(result: dict, latency_ms: float, compact: bool = False) -> None:
    label = result["class_label"]
    meta = WASTE_METADATA.get(label, {})
    confidence = result["confidence"] * 100
    bio_label = _bio_label(meta)
    disposal_action = meta.get("disposal", "General Waste")

    if compact:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Material", label.title())
        m2.metric("Confidence", f"{confidence:.1f}%")
        m3.metric("Latency", f"{latency_ms:.0f} ms")
        m4.metric("Category", bio_label)
        _render_latency_status(latency_ms)
        return

    # Action-first functional card
    st.markdown(
        f"""
        <div class="ob-action-card">
            <p class="ob-kicker">DISPOSAL ACTION GUIDANCE</p>
            <h3>📍 {disposal_action}</h3>
            <div>
                <span class="ob-badge ob-badge-blue">Material: {label.title()}</span>
                <span class="ob-badge ob-badge-green">{bio_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Classification Confidence", f"{confidence:.1f}%")
    c2.metric("Inference Latency", f"{latency_ms:.1f} ms")
    c3.metric("Latency Target", f"≤ {LATENCY_TARGET_MS} ms")
    _render_latency_status(latency_ms)


def render_disposal_guidance(result: dict) -> None:
    label = result["class_label"]
    meta = WASTE_METADATA.get(label)
    if not meta:
        st.info("No disposal metadata available for this class.")
        return

    recyclable = meta["recyclable"]
    recycle_text = (
        "Yes — Fully Recyclable"
        if recyclable is True
        else "No — Landfill / Special Handling"
        if recyclable is False
        else str(recyclable)
    )

    with st.container(border=True):
        st.subheader("Disposal Instructions & Impact")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Recommended Bin:** {meta['disposal']}")
            st.markdown(f"**Recyclability:** {recycle_text}")
        with c2:
            st.markdown(f"**Decomposition Time:** {meta['decomposition']}")
            st.markdown(f"**Material Category:** {meta['category']}")

        st.divider()
        st.markdown("**Handling Tips**")
        for tip in meta["tips"]:
            st.markdown(f'<div class="ob-tip">💡 {tip}</div>', unsafe_allow_html=True)

        st.caption(f"**Environmental Note:** {meta['environmental_impact']}")


def render_probability_chart(result: dict) -> None:
    st.subheader("Classification Probabilities")
    
    # Sort probabilities for immediate functional comparison
    paired = list(zip(CLASS_LABELS, result["probabilities"]))
    paired.sort(key=lambda x: x[1], reverse=True)
    
    for class_name, prob in paired:
        pct = float(prob) * 100
        col_name, col_bar = st.columns([1, 3])
        col_name.write(f"**{class_name.title()}**")
        col_bar.progress(float(prob), text=f"{pct:.1f}%")


def _render_latency_status(latency_ms: float) -> None:
    if latency_ms <= LATENCY_TARGET_MS:
        st.caption(f"✓ Latency target met ({latency_ms:.1f} ms ≤ {LATENCY_TARGET_MS} ms target).")
    else:
        over = latency_ms - LATENCY_TARGET_MS
        st.caption(f"⚠ {over:.1f} ms over performance target.")


def _bio_label(meta: dict) -> str:
    bio = meta.get("biodegradable")
    if bio is True:
        return "Biodegradable"
    if bio is False:
        return "Non-biodegradable"
    return "Mixed"


def _bio_short(meta: dict) -> str:
    bio = meta.get("biodegradable")
    if bio is True:
        return "biodegradable"
    if bio is False:
        return "non-biodegradable"
    return "mixed"
