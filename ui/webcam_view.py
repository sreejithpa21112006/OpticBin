"""Live webcam classification view with guaranteed camera cleanup."""

from __future__ import annotations

import time

import cv2
import streamlit as st

from config.settings import WASTE_METADATA
from src.camera import CameraError, CameraSession
from src.preprocessor import preprocess_frame
from src.xai_engine import RecyclingXAIEngine
from ui.components import render_empty_state, render_prediction_summary
from ui.state_manager import SnapshotStateManager

DEFAULT_XAI_INTERVAL = 5


def render_webcam_view(engine: RecyclingXAIEngine) -> None:
    st.subheader("Live webcam")
    st.caption("Scan waste items live or capture an object snapshot for detailed analysis.")

    mode = st.radio(
        "Select Operation Mode",
        options=["📸 Smart Object Snapshot", "🎥 Continuous Live Stream"],
        horizontal=True,
    )

    if mode == "📸 Smart Object Snapshot":
        _render_snapshot_mode(engine)
    else:
        _render_continuous_stream_mode(engine)


def _render_snapshot_mode(engine: RecyclingXAIEngine) -> None:
    st.info("Point your camera at an object and click **Capture & Analyze Object**.")

    capture_col, _ = st.columns([1, 2])
    with capture_col:
        trigger_capture = st.button("📸 Capture & Analyze Object", type="primary", use_container_width=True)

    snapshot_data = SnapshotStateManager.get()

    if trigger_capture or snapshot_data is not None:
        if trigger_capture:
            try:
                with CameraSession() as camera:
                    # Warm up camera for a clean frame
                    for _ in range(5):
                        frame = camera.read()

                    started = time.perf_counter()
                    input_tensor, rgb_float = preprocess_frame(frame)
                    result = engine.explain(input_tensor, rgb_float)
                    latency_ms = (time.perf_counter() - started) * 1000

                    SnapshotStateManager.save(result, frame, latency_ms)
                    snapshot_data = (result, frame, latency_ms)
            except CameraError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Webcam capture failed: {exc}")
                return

        if snapshot_data is not None:
            result, frame, latency_ms = snapshot_data

            st.success("Object captured and analyzed!")

            feed_col, heatmap_col = st.columns(2)
            display_frame = cv2.cvtColor(cv2.resize(frame, (448, 448)), cv2.COLOR_BGR2RGB)
            feed_col.image(display_frame, caption="Captured Object", width="stretch")
            heatmap_col.image(
                result["heatmap_overlay"],
                caption=f"Grad-CAM Explanation ({result['class_label'].title()})",
                width="stretch",
            )

            render_prediction_summary(result, latency_ms, compact=False)

            if st.button("🔄 Clear Snapshot & Capture Another"):
                SnapshotStateManager.clear()
                st.rerun()
    else:
        render_empty_state(
            "Ready to capture",
            "Click 'Capture & Analyze Object' to take a snapshot of the material.",
        )


def _render_continuous_stream_mode(engine: RecyclingXAIEngine) -> None:
    controls, _ = st.columns([1, 1])
    with controls:
        xai_interval = st.slider(
            "Explain every N frames",
            min_value=1,
            max_value=15,
            value=DEFAULT_XAI_INTERVAL,
            help=(
                "Every frame is classified. The Grad-CAM heatmap needs a much "
                "slower backward pass, so it refreshes on this interval. "
                "Higher values give a smoother feed."
            ),
        )

    running = st.toggle("Start webcam stream", value=False)
    if not running:
        render_empty_state(
            "Webcam is idle",
            "Enable the stream to classify frames in real time. "
            "The camera is released as soon as the stream stops.",
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
                feed_slot.image(display_frame, caption="Live feed", width="stretch")

                # Only re-send the heatmap when it actually changed.
                if explain_this_frame:
                    is_onnx = getattr(engine, "__class__", None).__name__ == "ONNXInferenceEngine"
                    mode_tag = "ONNX Fast Mode" if is_onnx else "Grad-CAM"
                    heatmap_slot.image(
                        result["heatmap_overlay"],
                        caption=(
                            f"{mode_tag} -- {result['class_label'].title()} "
                            f"({result['confidence'] * 100:.1f}%)"
                        ),
                        width="stretch",
                    )

                with metrics_slot.container():
                    render_prediction_summary(result, latency_ms, compact=True)
                    stage = "classify + explain" if explain_this_frame else "classify only"
                    disposal = WASTE_METADATA.get(result["class_label"], {}).get("disposal")
                    caption = f"Frame {frame_index + 1} - {stage}"
                    if disposal:
                        caption += f" - Dispose: {disposal}"
                    st.caption(caption)

                frame_index += 1
                time.sleep(0.01)
    except CameraError as exc:
        st.error(str(exc))
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
        st.error(f"Webcam inference failed: {exc}")
