"""
OpticBin — Image Upload Classification View
================================================
Premium 2-column layout: Visual Capture on the left, Hero Disposal Guidance on the right.
Supports the 3-bin degradability taxonomy with rich visual overlays.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from config.settings import CLASS_LABELS
from src.preprocessor import preprocess_pil
from src.yolo_cropper import detect_and_annotate_yolo
from ui.components import (
    render_active_learning_badge,
    render_disposal_guidance,
    render_empty_state,
    render_heatmap,
    render_llm_xai_explanation,
    render_prediction_summary,
    render_probability_chart,
    render_untrained_warning,
)
from ui.state_manager import SessionTracker


def render_image_view(engine: Any, advisor=None, active_learner=None) -> None:
    """Render the premium image analysis workflow."""
    render_untrained_warning(engine)

    uploaded_file = st.file_uploader(
        "Upload Waste Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a photo of any waste item to identify which bin it belongs to.",
    )

    if uploaded_file is None:
        render_empty_state(
            "Ready to Scan",
            "Upload or drag-and-drop any waste item photo to instantly identify the correct disposal bin.",
        )
        return

    image = Image.open(uploaded_file)

    try:
        with st.spinner("Analyzing waste item..."):
            started = time.perf_counter()
            rgb_float = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            result = engine.predict_and_explain(image, rgb_float)

            # Direct precision classification
            if active_learner is not None and active_learner.is_available:
                gemini_res = active_learner.classify_directly(image)
                if gemini_res is not None:
                    result["class_label"] = gemini_res["label"]
                    result["confidence"] = gemini_res["confidence"]
                    result["item_name"] = gemini_res["item_name"]
                    result["ai_reasoning"] = gemini_res["reasoning"]
                    result["classified_by_gemini"] = True

                    # Update probability distribution
                    active_classes = result.get("class_names", CLASS_LABELS)
                    if gemini_res["label"] in active_classes:
                        idx = active_classes.index(gemini_res["label"])
                        rem = max(0.0, 1.0 - gemini_res["confidence"])
                        probs = [rem / max(1, len(active_classes) - 1)] * len(active_classes)
                        probs[idx] = gemini_res["confidence"]
                        result["probabilities"] = probs

                    # Update visual bounding box overlay
                    box = gemini_res.get("box") or result.get("primary_box")
                    result["primary_box"] = box
                    if hasattr(engine, "render_overlay"):
                        result["heatmap_overlay"] = engine.render_overlay(
                            image,
                            box=box,
                            label=gemini_res["label"],
                            confidence=gemini_res["confidence"],
                            subtitle=gemini_res["item_name"],
                        )

            latency_ms = (time.perf_counter() - started) * 1000
            SessionTracker.add_scan(result["class_label"], result["confidence"], latency_ms)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    al_result = None

    # 2-Column Premium Layout
    col_visual, col_guidance = st.columns([1.1, 1])

    with col_visual:
        st.markdown('<div class="ob-section-title">Captured Item</div>', unsafe_allow_html=True)
        if result.get("heatmap_overlay") is not None:
            # Display name for caption
            display_label = {
                "biodegradable": "Biodegradable",
                "non-biodegradable": "Non-Biodegradable",
                "e-waste": "E-Waste",
            }.get(result["class_label"], result["class_label"].title())
            item_tag = f" ({result['item_name']})" if result.get("item_name") else ""

            st.image(
                result["heatmap_overlay"],
                caption=f"Detected: {display_label}{item_tag} ({result['confidence']*100:.0f}%)",
                width="stretch",
            )
        else:
            st.image(image, caption="Uploaded Image", width="stretch")

        # Collapsible technical details
        with st.expander("Technical Inspection", expanded=False):
            crop_display = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            t1, t2 = st.tabs(["Model Input", "Detection Overlay"])
            with t1:
                st.image(crop_display, caption="Preprocessed Input", width="stretch")
            with t2:
                render_heatmap(result.get("heatmap_overlay"), engine)
            render_llm_xai_explanation(result, advisor, model_type="OpticBin Unified Model")

    with col_guidance:
        st.markdown('<div class="ob-section-title">Sorting Guidance</div>', unsafe_allow_html=True)
        render_active_learning_badge(al_result)
        render_prediction_summary(result, latency_ms)
        render_probability_chart(result)
        render_disposal_guidance(result, advisor)
