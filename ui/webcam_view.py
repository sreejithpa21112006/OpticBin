"""Live webcam classification view with interactive camera targeting and instant snapshot capture."""

from __future__ import annotations

import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config.settings import WASTE_METADATA
from src.camera import CameraError, CameraSession
from src.preprocessor import preprocess_frame, preprocess_pil
from src.xai_engine import RecyclingXAIEngine
from ui.components import (
    render_disposal_guidance,
    render_empty_state,
    render_prediction_summary,
    render_probability_chart,
)
from ui.state_manager import SessionTracker, SnapshotStateManager

DEFAULT_XAI_INTERVAL = 5


def render_webcam_view(engine: RecyclingXAIEngine) -> None:
    st.subheader("Live Camera & Snapshot View")
    st.caption("Target your item in the viewfinder and capture for instant AI disposal analysis.")

    mode = st.radio(
        "Operation Mode",
        options=["🎯 Interactive Camera Viewfinder", "🎥 Continuous Stream Mode"],
        horizontal=True,
    )

    if "Interactive Camera Viewfinder" in mode:
        _render_snapshot_mode(engine)
    else:
        _render_continuous_stream_mode(engine)

    _render_session_statistics()


def _render_snapshot_mode(engine: RecyclingXAIEngine) -> None:
    st.info("🎯 **Target & Capture**: Aim your webcam at the item in the viewfinder below, then click **Take Photo**.")

    camera_photo = st.camera_input("Camera Viewfinder", label_visibility="collapsed")

    if camera_photo is not None:
        try:
            image = Image.open(camera_photo)
            started = time.perf_counter()
            input_tensor, rgb_float = preprocess_pil(image)
            result = engine.explain(input_tensor, rgb_float)
            latency_ms = (time.perf_counter() - started) * 1000

            frame_np = np.array(image.convert("RGB"))
            SnapshotStateManager.save(result, frame_np, latency_ms)
            SessionTracker.add_scan(result["class_label"], result["confidence"], latency_ms)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

        st.success("✓ Object captured and analyzed!")

        feed_col, heatmap_col = st.columns(2)
        feed_col.image(image, caption="Captured Object", width="stretch")
        heatmap_col.image(
            result["heatmap_overlay"],
            caption=f"Grad-CAM Heatmap ({result['class_label'].title()})",
            width="stretch",
        )

        render_prediction_summary(result, latency_ms, compact=False)
        
        c1, c2 = st.columns(2)
        with c1:
            render_disposal_guidance(result)
        with c2:
            render_probability_chart(result)
    else:
        render_empty_state(
            "Camera Viewfinder Ready",
            "Point your camera at an item in the box above and click 'Take Photo' to capture.",
        )


def _render_continuous_stream_mode(engine: RecyclingXAIEngine) -> None:
    controls, _ = st.columns([1, 1])
    with controls:
        xai_interval = st.slider(
            "Explain every N frames",
            min_value=1,
            max_value=15,
            value=DEFAULT_XAI_INTERVAL,
            help="Refreshes Grad-CAM heatmaps periodically to maintain smooth stream FPS.",
        )

    running = st.toggle("Enable Live Camera Stream", value=False)
    if not running:
        render_empty_state(
            "Camera Stream Paused",
            "Toggle 'Enable Live Camera Stream' to classify items continuously in real time.",
        )
        return

    feed_col, heatmap_col = st.columns(2)
    feed_slot = feed_col.empty()
    heatmap_slot = heatmap_col.empty()
    metrics_slot = st.empty()

    try:
        with CameraSession() as camera:
            frame_index = 0
            while True:
                frame = camera.read()
                input_tensor, rgb_float = preprocess_frame(frame)
                explain_this_frame = frame_index % xai_interval == 0

                started = time.perf_counter()
                if explain_this_frame:
                    result = engine.explain(input_tensor, rgb_float)
                else:
                    result = engine.predict(input_tensor)
                latency_ms = (time.perf_counter() - started) * 1000

                display_frame = cv2.cvtColor(cv2.resize(frame, (448, 448)), cv2.COLOR_BGR2RGB)
                feed_slot.image(display_frame, caption="Live Stream Feed", width="stretch")

                if explain_this_frame:
                    is_onnx = getattr(engine, "__class__", None).__name__ == "ONNXInferenceEngine"
                    mode_tag = "ONNX Fast" if is_onnx else "Grad-CAM"
                    heatmap_slot.image(
                        result["heatmap_overlay"],
                        caption=f"{mode_tag} — {result['class_label'].title()} ({result['confidence'] * 100:.1f}%)",
                        width="stretch",
                    )
                    SessionTracker.add_scan(result["class_label"], result["confidence"], latency_ms)

                with metrics_slot.container():
                    render_prediction_summary(result, latency_ms, compact=True)

                frame_index += 1
                time.sleep(0.01)
    except CameraError as exc:
        st.error(str(exc))
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
        st.error(f"Webcam inference failed: {exc}")


def _render_session_statistics() -> None:
    stats = SessionTracker.get_stats()
    if stats["total"] == 0:
        return

    st.divider()
    with st.expander(f"📋 Session Log ({stats['total']} items scanned)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Items Scanned", stats["total"])
        c2.metric("Recyclable Items", stats["recyclable_count"])
        c3.metric("Recycling Percentage", f"{stats['recyclable_pct']:.1f}%")

        st.subheader("Breakdown by Material")
        cols = st.columns(len(stats["counts"]))
        for col, (lbl, cnt) in zip(cols, stats["counts"].items()):
            col.metric(lbl.title(), cnt)

        if st.button("Clear Session Log"):
            SessionTracker.clear_history()
            st.rerun()
