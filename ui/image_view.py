"""Image-upload classification view with clear functional guidance."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st
from PIL import Image

from src.preprocessor import preprocess_pil
from ui.components import (
    render_disposal_guidance,
    render_empty_state,
    render_heatmap,
    render_prediction_summary,
    render_probability_chart,
    render_untrained_warning,
)
from ui.state_manager import SessionTracker


def render_image_view(engine: Any) -> None:
    st.subheader("Image File Analysis")
    render_untrained_warning(engine)
    uploaded_file = st.file_uploader(
        "Select a waste image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload an image to identify material, view Grad-CAM heatmaps, and inspect disposal rules.",
    )

    if uploaded_file is None:
        render_empty_state(
            "Ready for Image Upload",
            "Select an image file (JPG, PNG, WebP) to analyze waste material.",
        )
        return

    image = Image.open(uploaded_file)
    input_tensor, rgb_float = preprocess_pil(image)

    try:
        with st.spinner("Analyzing waste material..."):
            started = time.perf_counter()
            result = engine.predict_and_explain(input_tensor, rgb_float)
            latency_ms = (time.perf_counter() - started) * 1000
            SessionTracker.add_scan(result["class_label"], result["confidence"], latency_ms)
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
        st.error(f"Inference failed: {exc}")
        return

    media_col, info_col = st.columns([1.2, 1])

    with media_col:
        original_tab, heatmap_tab = st.tabs(["Original Image", "Grad-CAM Heatmap"])
        with original_tab:
            st.image(image, width="stretch")
        with heatmap_tab:
            render_heatmap(result.get("heatmap_overlay"), engine)

    with info_col:
        render_prediction_summary(result, latency_ms)
        render_disposal_guidance(result)

    st.divider()
    render_probability_chart(result)
