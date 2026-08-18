"""Reusable Streamlit components for the OpticBin dashboard."""

from __future__ import annotations

import streamlit as st

from config.settings import (
    CLASS_LABELS,
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
        "Offline waste classification with explainable AI, disposal guidance, "
        "and live webcam support."
    )


PYTORCH_FRAMEWORK = "PyTorch + Grad-CAM"
ONNX_FRAMEWORK = "ONNX Runtime (Fast)"


def render_sidebar() -> tuple[str, str, str]:
    """Render model, framework, and input controls. Returns `(model_choice, framework, input_mode)`."""
    with st.sidebar:
        st.header("Workspace")
        model_choice = st.selectbox(
            "Backbone",
            options=list(SUPPORTED_MODELS.keys()),
            index=0,
            help="Switch between CNN and ViT architectures without restarting.",
        )
        st.caption(SUPPORTED_MODELS[model_choice]["description"])

        framework = st.radio(
            "Inference Engine",
            [PYTORCH_FRAMEWORK, ONNX_FRAMEWORK],
            index=0,
            help="Choose PyTorch with Grad-CAM heatmaps or ONNX Runtime for low latency.",
        )

        input_mode = st.radio(
            "Input source",
            [IMAGE_MODE, WEBCAM_MODE],
            index=0,
        )

        st.divider()
        st.subheader("Targets")
        st.markdown(
            f"- Latency: ≤ {LATENCY_TARGET_MS} ms\n"
            f"- Classes: {NUM_CLASSES}\n"
            f"- Input: {INPUT_SIZE[0]} × {INPUT_SIZE[1]}"
        )

        st.divider()
        st.subheader("Waste classes")
        for label in CLASS_LABELS:
            meta = WASTE_METADATA[label]
            st.markdown(
                f"**{label.title()}** · {_bio_short(meta)}"
            )

    return model_choice, framework, input_mode


def render_engine_status(model_type: str, framework: str, engine: object) -> None:
    is_onnx = getattr(engine, "__class__", None).__name__ == "ONNXInferenceEngine"
    using_finetuned = getattr(engine, "using_finetuned_weights", False)

    if "ONNX" in framework:
        if is_onnx:
            st.sidebar.success(f"Using ONNX Runtime engine for `{model_type}`.")
        else:
            st.sidebar.warning(
                f"No `.onnx` checkpoint found for `{model_type}`. "
                "Falling back to PyTorch engine. Run `python train.py` to export ONNX weights."
            )
    else:
        if using_finetuned:
            st.sidebar.success(f"Using fine-tuned PyTorch weights for `{model_type}`.")
        else:
            st.sidebar.warning(
                f"No `{model_type}.pt` checkpoint found. "
                "Falling back to ImageNet-pretrained weights."
            )


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ob-empty">
            <p class="ob-kicker">Ready</p>
            <h3>{title}</h3>
            <p class="ob-muted">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_summary(result: dict, latency_ms: float, compact: bool = False) -> None:
    label = result["class_label"]
    meta = WASTE_METADATA.get(label, {})
    confidence = result["confidence"] * 100
    bio_label = _bio_label(meta)

    if compact:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Material", label.title())
        m2.metric("Confidence", f"{confidence:.1f}%")
        m3.metric("Latency", f"{latency_ms:.0f} ms")
        m4.metric("Biodegradable", bio_label)
        _render_latency_status(latency_ms)
        return

    st.markdown(
        f"""
        <div class="ob-hero">
            <p class="ob-kicker">Prediction</p>
            <h2>{label.title()}</h2>
            <span class="ob-badge">{bio_label}</span>
            <p class="ob-muted">{meta.get('category', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{confidence:.1f}%")
    c2.metric("Latency", f"{latency_ms:.0f} ms")
    c3.metric("Target", f"≤ {LATENCY_TARGET_MS} ms")
    _render_latency_status(latency_ms)


def render_disposal_guidance(result: dict) -> None:
    label = result["class_label"]
    meta = WASTE_METADATA.get(label)
    if not meta:
        st.info("No disposal metadata is available for this class.")
        return

    recyclable = meta["recyclable"]
    recycle_text = (
        "Fully recyclable"
        if recyclable is True
        else "Landfill / special disposal"
        if recyclable is False
        else str(recyclable)
    )

    with st.container(border=True):
        st.subheader("Disposal guidance")
        st.markdown(f"**Method:** {meta['disposal']}")
        st.markdown(f"**Recyclable:** {recycle_text}")
        st.markdown(f"**Decomposition:** {meta['decomposition']}")

        st.markdown("**Tips**")
        for tip in meta["tips"]:
            st.markdown(f'<div class="ob-tip">{tip}</div>', unsafe_allow_html=True)

        st.caption(f"Environmental impact: {meta['environmental_impact']}")


def render_probability_chart(result: dict) -> None:
    st.subheader("Class probabilities")
    chart_data = {
        label.title(): float(prob)
        for label, prob in zip(CLASS_LABELS, result["probabilities"])
    }
    st.bar_chart(chart_data, height=260)


def _render_latency_status(latency_ms: float) -> None:
    if latency_ms <= LATENCY_TARGET_MS:
        st.caption(f"Within the {LATENCY_TARGET_MS} ms latency budget.")
    else:
        over = latency_ms - LATENCY_TARGET_MS
        st.caption(f"{over:.0f} ms over the {LATENCY_TARGET_MS} ms latency budget.")


def _bio_label(meta: dict) -> str:
    bio = meta.get("biodegradable")
    if bio is True:
        return "Biodegradable"
    if bio is False:
        return "Non-biodegradable"
    return "Mixed / varies"


def _bio_short(meta: dict) -> str:
    bio = meta.get("biodegradable")
    if bio is True:
        return "biodegradable"
    if bio is False:
        return "non-biodegradable"
    return "mixed"
