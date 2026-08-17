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

DEFAULT_XAI_INTERVAL = 5


def render_webcam_view(engine: RecyclingXAIEngine) -> None:
    st.subheader("Live webcam")
    st.caption("Point the camera at waste items for real-time classification.")

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
                    heatmap_slot.image(
                        result["heatmap_overlay"],
                        caption=(
                            f"Grad-CAM — {result['class_label'].title()} "
                            f"({result['confidence'] * 100:.1f}%)"
                        ),
                        width="stretch",
                    )

                with metrics_slot.container():
                    render_prediction_summary(result, latency_ms, compact=True)
                    stage = "classify + explain" if explain_this_frame else "classify only"
                    disposal = WASTE_METADATA.get(result["class_label"], {}).get("disposal")
                    caption = f"Frame {frame_index + 1} · {stage}"
                    if disposal:
                        caption += f" · Dispose: {disposal}"
                    st.caption(caption)

                frame_index += 1
    except CameraError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Webcam inference failed: {exc}")
