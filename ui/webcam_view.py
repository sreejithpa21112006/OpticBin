"""
Streamlined live camera classification for OpticBin.
Features a reliable camera input, clean visual target overlay, and instant hero bin recommendations.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config.settings import CLASS_LABELS
from src.camera import CameraError, ThreadedCameraStream
from src.preprocessor import preprocess_frame, preprocess_pil
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
from ui.state_manager import SessionTracker, SnapshotStateManager

_STREAM_KEY = "_opticbin_camera_stream"
_FRAME_KEY = "_opticbin_stream_frame_index"
_HEATMAP_KEY = "_opticbin_last_heatmap"


def render_webcam_view(engine: Any, advisor=None, active_learner=None) -> None:
    """Render the simplified camera-based classification interface."""
    render_untrained_warning(engine)

    mode = st.radio(
        "Capture Method",
        ["Photo Snapshot", "Continuous Stream"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Photo Snapshot":
        _render_snapshot_mode(engine, advisor=advisor, active_learner=active_learner)
    else:
        _render_continuous_stream_mode(engine)


def _render_snapshot_mode(engine: Any, advisor=None, active_learner=None) -> None:
    """Clear, distraction-free photo capture with visual detection overlay."""
    photo = st.camera_input("Aim camera at waste item and take a photo", label_visibility="collapsed")

    if photo is None:
        render_empty_state(
            "Camera Viewfinder Ready",
            "Point your webcam at any waste item and click 'Take Photo' to classify and view disposal guidance.",
        )
        return

    image = Image.open(photo)

    try:
        with st.spinner("Analyzing waste item..."):
            started = time.perf_counter()
            rgb_float = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            result = engine.predict_and_explain(image, rgb_float)
            latency_ms = (time.perf_counter() - started) * 1000
            SessionTracker.add_scan(result["class_label"], result["confidence"], latency_ms)
    except Exception as exc:
        st.error(f"Scan analysis failed: {exc}")
        return

    # Active learning cross-check if low confidence
    al_result = None
    if active_learner is not None and active_learner.should_cross_check(result):
        with st.spinner("Cross-checking low confidence scan with AI supervisor..."):
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

    # 2-Column User-First Layout
    col_visual, col_guidance = st.columns([1.1, 1])

    with col_visual:
        st.markdown("### Captured Target")
        if result.get("heatmap_overlay") is not None:
            caption_tag = " - AI Supervisor Verified" if (result.get("corrected_by_gemini") or result.get("verified_by_gemini")) else ""
            st.image(
                result["heatmap_overlay"],
                caption=f"Target Detected: {result['class_label'].title()} ({result['confidence']*100:.0f}%){caption_tag}",
                width="stretch",
            )
        else:
            st.image(image, caption="Captured Frame", width="stretch")

        with st.expander("Inspection: Model Crop & Heatmap", expanded=False):
            crop_display = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            t1, t2 = st.tabs(["224x224 Model Input", "Attention Heatmap"])
            with t1:
                st.image(crop_display, caption="Preprocessed Normalized Crop", width="stretch")
            with t2:
                render_heatmap(result.get("heatmap_overlay"), engine)
            render_llm_xai_explanation(result, advisor, model_type=getattr(engine, "model_type", "EfficientNetV2"))

    with col_guidance:
        st.markdown("### Sorting Guidance")
        render_active_learning_badge(al_result)
        render_prediction_summary(result, latency_ms)
        render_probability_chart(result)
        render_disposal_guidance(result, advisor)


def _render_continuous_stream_mode(engine: Any) -> None:
    """Continuous video feed mode with graceful start/stop."""
    running = st.toggle("Activate Live Camera Stream", value=False)
    if not running:
        _stop_stream()
        render_empty_state(
            "Live Stream Inactive",
            "Toggle 'Activate Live Camera Stream' to classify items continuously in real time.",
        )
        return

    try:
        _ensure_stream()
    except CameraError as exc:
        st.error(str(exc))
        return

    _render_live_stream_fragment(engine)


@st.fragment(run_every=0.25)
def _render_live_stream_fragment(engine: Any) -> None:
    """Lightweight real-time stream fragment."""
    stream: ThreadedCameraStream | None = st.session_state.get(_STREAM_KEY)
    if stream is None:
        return

    try:
        frame = stream.read_latest()
    except CameraError:
        st.info("Connecting to camera stream...")
        return

    input_tensor, rgb_float = preprocess_frame(frame)
    started = time.perf_counter()
    result = engine.predict(input_tensor)
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    latency_ms = (time.perf_counter() - started) * 1000

    display_frame = cv2.cvtColor(cv2.resize(frame, (448, 448)), cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(display_frame, caption="Real-Time Feed", width="stretch")
    with col2:
        render_prediction_summary(result, latency_ms, compact=True)


def _ensure_stream() -> ThreadedCameraStream:
    """Ensure stream is running."""
    stream = st.session_state.get(_STREAM_KEY)
    if stream is None or not stream.is_running():
        stream = ThreadedCameraStream(device_index=0).start()
        st.session_state[_STREAM_KEY] = stream
        st.session_state[_FRAME_KEY] = 0
        st.session_state[_HEATMAP_KEY] = None
    return stream


def _stop_stream() -> None:
    """Safely terminate camera thread."""
    stream = st.session_state.get(_STREAM_KEY)
    if stream is not None:
        stream.stop()
        st.session_state[_STREAM_KEY] = None
