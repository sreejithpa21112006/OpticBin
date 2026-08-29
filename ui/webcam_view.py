"""Live webcam classification with snapshot capture and a non-blocking stream."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.camera import CameraError, ThreadedCameraStream
from src.preprocessor import preprocess_frame, preprocess_pil
from ui.components import (
    render_disposal_guidance,
    render_empty_state,
    render_heatmap,
    render_prediction_summary,
    render_probability_chart,
    render_untrained_warning,
)
from ui.state_manager import SessionTracker, SnapshotStateManager

DEFAULT_XAI_INTERVAL = 5
_STREAM_KEY = "_opticbin_camera_stream"
_FRAME_KEY = "_opticbin_stream_frame_index"
_HEATMAP_KEY = "_opticbin_last_heatmap"


def render_webcam_view(engine: Any) -> None:
    st.subheader("Live Camera & Snapshot View")
    st.caption("Target your item in the viewfinder and capture for instant AI disposal analysis.")
    render_untrained_warning(engine)

    mode = st.radio(
        "Operation Mode",
        options=["Interactive Camera Viewfinder", "Continuous Stream Mode"],
        horizontal=True,
    )

    if "Viewfinder" in mode:
        _stop_stream()
        _render_snapshot_mode(engine)
    else:
        _render_continuous_stream_mode(engine)

    _render_session_statistics()


def _render_snapshot_mode(engine: Any) -> None:
    st.info("Aim the webcam at the item in the viewfinder, then click **Take Photo**.")

    camera_photo = st.camera_input("Camera Viewfinder", label_visibility="collapsed")

    if camera_photo is None:
        render_empty_state(
            "Camera Viewfinder Ready",
            "Point your camera at an item in the box above and click 'Take Photo' to capture.",
        )
        return

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

    st.success("Object captured and analyzed.")

    feed_col, heatmap_col = st.columns(2)
    feed_col.image(image, caption="Captured Object", width="stretch")
    with heatmap_col:
        render_heatmap(
            result.get("heatmap_overlay"),
            engine,
            caption=f"Explanation ({result['class_label'].title()})",
        )

    render_prediction_summary(result, latency_ms, compact=False)

    c1, c2 = st.columns(2)
    with c1:
        render_disposal_guidance(result)
    with c2:
        render_probability_chart(result)


def _render_continuous_stream_mode(engine: Any) -> None:
    xai_interval = st.slider(
        "Explain every N frames",
        min_value=1,
        max_value=15,
        value=DEFAULT_XAI_INTERVAL,
        help="Refreshes Grad-CAM periodically so classification can stay near real time.",
    )

    running = st.toggle("Enable Live Camera Stream", value=False)
    if not running:
        _stop_stream()
        render_empty_state(
            "Camera Stream Paused",
            "Toggle 'Enable Live Camera Stream' to classify items continuously in real time.",
        )
        return

    try:
        _ensure_stream()
    except CameraError as exc:
        st.error(str(exc))
        return

    _render_live_stream_fragment(engine, xai_interval)


@st.fragment(run_every=0.2)
def _render_live_stream_fragment(engine: Any, xai_interval: int) -> None:
    stream: ThreadedCameraStream | None = st.session_state.get(_STREAM_KEY)
    if stream is None:
        return

    try:
        frame = stream.read_latest()
    except CameraError:
        st.info("Waiting for the first camera frame...")
        return

    input_tensor, rgb_float = preprocess_frame(frame)
    frame_index = int(st.session_state.get(_FRAME_KEY, 0))
    explain_this_frame = frame_index % max(1, xai_interval) == 0
    st.session_state[_FRAME_KEY] = frame_index + 1

    started = time.perf_counter()
    if explain_this_frame and getattr(engine, "supports_gradcam", False):
        result = engine.explain(input_tensor, rgb_float)
        st.session_state[_HEATMAP_KEY] = result.get("heatmap_overlay")
        log_scan = True
    else:
        result = engine.predict(input_tensor)
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        log_scan = False
    latency_ms = (time.perf_counter() - started) * 1000
    if log_scan:
        SessionTracker.add_scan(result["class_label"], result["confidence"], latency_ms)

    display_frame = cv2.cvtColor(cv2.resize(frame, (448, 448)), cv2.COLOR_BGR2RGB)
    feed_col, heatmap_col = st.columns(2)
    feed_col.image(display_frame, caption="Live Stream Feed", width="stretch")
    with heatmap_col:
        render_heatmap(st.session_state.get(_HEATMAP_KEY), engine, caption="Latest Grad-CAM")

    render_prediction_summary(result, latency_ms, compact=True)


def _ensure_stream() -> ThreadedCameraStream:
    stream = st.session_state.get(_STREAM_KEY)
    if stream is None or not stream.is_running():
        stream = ThreadedCameraStream(device_index=0).start()
        st.session_state[_STREAM_KEY] = stream
        st.session_state[_FRAME_KEY] = 0
        st.session_state[_HEATMAP_KEY] = None
    return stream


def _stop_stream() -> None:
    stream = st.session_state.pop(_STREAM_KEY, None)
    st.session_state.pop(_FRAME_KEY, None)
    st.session_state.pop(_HEATMAP_KEY, None)
    if stream is not None:
        stream.stop()


def _render_session_statistics() -> None:
    stats = SessionTracker.get_stats()
    if stats["total"] == 0:
        return

    st.divider()
    with st.expander(f"Session Log ({stats['total']} items scanned)", expanded=False):
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
