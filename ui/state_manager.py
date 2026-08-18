"""
OpticBin — Streamlit State Manager
==================================
Type-safe wrappers for managing Streamlit session state variables.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
import streamlit as st


class SnapshotStateManager:
    """Manages session state persistence for camera snapshot mode."""

    KEY_RESULT = "snapshot_result"
    KEY_FRAME = "snapshot_frame"
    KEY_LATENCY = "snapshot_latency"

    @classmethod
    def save(cls, result: dict, frame: np.ndarray, latency_ms: float) -> None:
        st.session_state[cls.KEY_RESULT] = result
        st.session_state[cls.KEY_FRAME] = frame
        st.session_state[cls.KEY_LATENCY] = latency_ms

    @classmethod
    def get(cls) -> Optional[Tuple[dict, np.ndarray, float]]:
        if (
            cls.KEY_RESULT in st.session_state
            and cls.KEY_FRAME in st.session_state
            and cls.KEY_LATENCY in st.session_state
        ):
            return (
                st.session_state[cls.KEY_RESULT],
                st.session_state[cls.KEY_FRAME],
                st.session_state[cls.KEY_LATENCY],
            )
        return None

    @classmethod
    def clear(cls) -> None:
        for key in (cls.KEY_RESULT, cls.KEY_FRAME, cls.KEY_LATENCY):
            if key in st.session_state:
                del st.session_state[key]
