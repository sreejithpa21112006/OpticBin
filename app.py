"""
OpticBin — Streamlit Dashboard
================================
Launch:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

from ui.components import (
    render_engine_status,
    render_header,
    render_recycling_chatbot,
    render_sidebar,
)
from ui.image_view import render_image_view
from ui.styles import IMAGE_MODE, apply_styles
from ui.webcam_view import render_webcam_view

st.set_page_config(
    page_title="OpticBin - Edge-AI Waste Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=True)
def load_engine():
    """Load and return the fine-tuned Unified YOLOv8 waste detection engine."""
    from src.yolo_unified_engine import YOLOUnifiedInferenceEngine
    return YOLOUnifiedInferenceEngine()


@st.cache_resource(show_spinner=False)
def load_advisor(api_key: str | None = None):
    """Initialise the Gemini LLM advisor (cached per session)."""
    try:
        from src.llm_advisor import GeminiAdvisor
        return GeminiAdvisor(api_key=api_key)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_active_learner(api_key: str | None = None):
    """Initialise the Gemini Vision active learner (cached per session)."""
    try:
        from src.active_learner import ActiveLearner
        return ActiveLearner(api_key=api_key)
    except Exception:
        return None


def _render_api_key_input() -> str | None:
    """
    Render an API key input in the sidebar if no key is available from
    environment variables or Streamlit secrets. Returns the resolved key.
    """
    # Check existing sources first
    existing = None
    try:
        existing = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not existing:
        existing = os.environ.get("GEMINI_API_KEY")
    if not existing:
        existing = st.session_state.get("_gemini_api_key")

    if existing:
        return existing

    # Prompt the user
    st.sidebar.divider()
    st.sidebar.subheader("Gemini API Key")
    st.sidebar.caption(
        "Optional: Enables AI Assistant and active learning cross-checks. "
        "Get a free key at aistudio.google.com."
    )
    key_input = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        key="_gemini_key_input",
        label_visibility="collapsed",
    )
    if key_input:
        st.session_state["_gemini_api_key"] = key_input
        return key_input

    return None


def main() -> None:
    apply_styles()
    render_header()
    input_mode = render_sidebar()
    engine = load_engine()
    render_engine_status(engine)

    # Resolve API key and initialise LLM components
    api_key = _render_api_key_input()
    advisor = load_advisor(api_key)
    active_learner = load_active_learner(api_key)

    # Sidebar chatbot
    render_recycling_chatbot(advisor)

    if input_mode == IMAGE_MODE:
        render_image_view(engine, advisor=advisor, active_learner=active_learner)
    else:
        render_webcam_view(engine, advisor=advisor, active_learner=active_learner)

    st.divider()
    st.caption(
        "OpticBin v1.2.0 - Edge-AI Waste Classifier and Instant Sorting Guidance."
    )


if __name__ == "__main__":
    main()
