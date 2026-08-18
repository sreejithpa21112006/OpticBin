"""Image-upload classification view."""

from __future__ import annotations

import time

import streamlit as st
from PIL import Image

from src.preprocessor import preprocess_pil
from src.xai_engine import RecyclingXAIEngine
from ui.components import (
    render_disposal_guidance,
    render_empty_state,
    render_prediction_summary,
    render_probability_chart,
)


def render_image_view(engine: RecyclingXAIEngine) -> None:
    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Classify a still image and inspect the Grad-CAM explanation.",
    )

    if uploaded_file is None:
        render_empty_state(
            "Upload an image to begin",
            "OpticBin will identify the material, show an explanation heatmap, "
            "and provide disposal guidance.",
        )
        return

    image = Image.open(uploaded_file)
    input_tensor, rgb_float = preprocess_pil(image)

    try:
        with st.spinner("Analyzing waste material..."):
            started = time.perf_counter()
            result = engine.predict_and_explain(input_tensor, rgb_float)
            latency_ms = (time.perf_counter() - started) * 1000
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
        st.error(f"Inference failed: {exc}")
        return

    media_col, info_col = st.columns([1.25, 1])

    with media_col:
        original_tab, heatmap_tab = st.tabs(["Original", "Grad-CAM"])
        with original_tab:
            st.image(image, width="stretch")
        with heatmap_tab:
            st.image(result["heatmap_overlay"], width="stretch")

    with info_col:
        render_prediction_summary(result, latency_ms)
        render_disposal_guidance(result)

    st.divider()
    render_probability_chart(result)
