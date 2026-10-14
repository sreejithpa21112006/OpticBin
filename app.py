"""
OpticBin — Streamlit Dashboard
================================
AI-Powered Home Waste Sorting — 3-Bin Degradability System.

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
    render_scan_stats,
    render_sidebar,
)
from ui.image_view import render_image_view
from ui.styles import IMAGE_MODE, apply_styles
from ui.webcam_view import render_webcam_view

st.set_page_config(
    page_title="OpticBin — AI Home Waste Sorter",
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
    Silently resolve the backend neural acceleration key from secrets,
    environment variables, or local config without exposing any hints in the UI.
    """
    existing = None
    try:
        existing = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not existing:
        existing = os.environ.get("GEMINI_API_KEY")
    if not existing:
        secrets_file = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if "GEMINI_API_KEY" in line and "=" in line:
                            existing = line.split("=")[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
    if not existing:
        existing = st.session_state.get("_gemini_api_key")

    return existing


def main() -> None:
    apply_styles()
    render_header()
    # Resolve API key and initialise LLM components
    api_key = _render_api_key_input()
    advisor = load_advisor(api_key)
    active_learner = load_active_learner(api_key)

    input_mode = render_sidebar()
    engine = load_engine()
    render_engine_status(engine)

    # Sidebar features
    render_recycling_chatbot(advisor)
    render_scan_stats()

    if input_mode == IMAGE_MODE:
        render_image_view(engine, advisor=advisor, active_learner=active_learner)
    else:
        render_webcam_view(engine, advisor=advisor, active_learner=active_learner)

    # Premium footer
    st.markdown(
        """
        <div class="ob-footer">
            OpticBin v2.0.0 — AI-Powered Home Waste Sorting &nbsp;·&nbsp;
            Biodegradable &nbsp;·&nbsp; Non-Biodegradable &nbsp;·&nbsp; E-Waste
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
