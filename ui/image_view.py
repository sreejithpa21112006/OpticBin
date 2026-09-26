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
            "📸 Ready to Scan",
            "Upload or drag-and-drop any waste item photo to instantly identify the correct disposal bin.",
        )
        return

    image = Image.open(uploaded_file)

    try:
        with st.spinner("🔍 Analyzing waste item..."):
            started = time.perf_counter()
            rgb_float = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            result = engine.predict_and_explain(image, rgb_float)
            latency_ms = (time.perf_counter() - started) * 1000
            SessionTracker.add_scan(result["class_label"], result["confidence"], latency_ms)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    # Active learning: verify with Gemini Vision if confidence is low
    al_result = None
    if active_learner is not None and active_learner.should_cross_check(result):
        with st.spinner("🤖 Cross-checking with AI supervisor..."):
            al_result = active_learner.verify_with_gemini_vision(image, result)
            if al_result is not None:
                active_learner.log_to_review_queue(image, result, al_result)
                # Apply high-confidence second opinion to correct low-confidence edge scans
                if not al_result.get("agreement") and al_result.get("gemini_confidence", 0.0) >= 0.70:
                    result["class_label"] = al_result["gemini_label"]
                    result["confidence"] = al_result["gemini_confidence"]
                    result["corrected_by_gemini"] = True

                    # Update probability distribution
                    active_classes = result.get("class_names", CLASS_LABELS)
                    if al_result["gemini_label"] in active_classes:
                        idx = active_classes.index(al_result["gemini_label"])
                        rem = max(0.0, 1.0 - al_result["gemini_confidence"])
                        probs = [rem / max(1, len(active_classes) - 1)] * len(active_classes)
                        probs[idx] = al_result["gemini_confidence"]
                        result["probabilities"] = probs

                    # Update visual bounding box overlay
                    if hasattr(engine, "render_overlay"):
                        box = result.get("primary_box")
                        result["heatmap_overlay"] = engine.render_overlay(
                            image,
                            box=box,
                            label=al_result["gemini_label"],
                            confidence=al_result["gemini_confidence"],
                            subtitle="AI Verified",
                        )
                elif al_result.get("agreement") and al_result.get("gemini_confidence", 0.0) >= 0.70:
                    result["confidence"] = max(result.get("confidence", 0.0), al_result["gemini_confidence"])
                    result["verified_by_gemini"] = True
                    if hasattr(engine, "render_overlay"):
                        box = result.get("primary_box")
                        result["heatmap_overlay"] = engine.render_overlay(
                            image,
                            box=box,
                            label=result["class_label"],
                            confidence=result["confidence"],
                            subtitle="AI Verified",
                        )

    # 2-Column Premium Layout
    col_visual, col_guidance = st.columns([1.1, 1])

    with col_visual:
        st.markdown('<div class="ob-section-title">🖼️ Captured Item</div>', unsafe_allow_html=True)
        if result.get("heatmap_overlay") is not None:
            verified_tag = " — AI Verified ✨" if (result.get("corrected_by_gemini") or result.get("verified_by_gemini")) else ""
            # Display name for caption
            display_label = {
                "biodegradable": "Biodegradable 🟢",
                "non-biodegradable": "Non-Biodegradable 🔵",
                "e-waste": "E-Waste 🟠",
            }.get(result["class_label"], result["class_label"].title())

            st.image(
                result["heatmap_overlay"],
                caption=f"Detected: {display_label} ({result['confidence']*100:.0f}%){verified_tag}",
                width="stretch",
            )
        else:
            st.image(image, caption="Uploaded Image", width="stretch")

        # Collapsible technical details
        with st.expander("🔬 Technical Inspection", expanded=False):
            crop_display = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            t1, t2 = st.tabs(["Model Input", "Detection Overlay"])
            with t1:
                st.image(crop_display, caption="Preprocessed Input", width="stretch")
            with t2:
                render_heatmap(result.get("heatmap_overlay"), engine)
            render_llm_xai_explanation(result, advisor, model_type=getattr(engine, "model_type", "YOLOv8"))

    with col_guidance:
        st.markdown('<div class="ob-section-title">🗑️ Sorting Guidance</div>', unsafe_allow_html=True)
        render_active_learning_badge(al_result)
        render_prediction_summary(result, latency_ms)
        render_probability_chart(result)
        render_disposal_guidance(result, advisor)
