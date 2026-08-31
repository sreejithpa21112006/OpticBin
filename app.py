"""
OpticBin — Streamlit Dashboard
================================
Launch:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config.settings import DEFAULT_MODEL, SUPPORTED_MODELS
from src.inference_engine import ONNXInferenceEngine, resolve_onnx_weights_path
from src.model_factory import resolve_weights_path
from src.xai_engine import RecyclingXAIEngine
from ui.components import (
    ONNX_FRAMEWORK,
    render_engine_status,
    render_header,
    render_sidebar,
)
from ui.image_view import render_image_view
from ui.styles import IMAGE_MODE, apply_styles
from ui.webcam_view import render_webcam_view

st.set_page_config(
    page_title="OpticBin — Edge-AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=True)
def load_engine(model_type: str = DEFAULT_MODEL, framework: str = "PyTorch + Grad-CAM"):
    """Load or retrieve cached engine for the active backbone architecture and framework."""
    if ONNX_FRAMEWORK in framework:
        onnx_path = resolve_onnx_weights_path(model_type)
        if onnx_path:
            return ONNXInferenceEngine(onnx_model_path=onnx_path, model_type=model_type)

    weights_path = resolve_weights_path(model_type)
    return RecyclingXAIEngine(
        model_type=model_type,
        weights_path=str(weights_path) if weights_path else None,
    )


def main() -> None:
    apply_styles()
    render_header()
    model_choice, framework, input_mode = render_sidebar()
    engine = load_engine(model_choice, framework)
    render_engine_status(model_choice, framework, engine)

    st.divider()

    if input_mode == IMAGE_MODE:
        render_image_view(engine)
    else:
        render_webcam_view(engine)

    st.divider()
    st.caption(
        "OpticBin v1.0.0 — Offline edge waste classification with Grad-CAM explanations."
    )


if __name__ == "__main__":
    main()
