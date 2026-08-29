"""
OpticBin — Streamlit State Manager
==================================
Type-safe wrappers for managing Streamlit session state variables.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Tuple

import numpy as np
import streamlit as st

from config.settings import is_recyclable


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


class SessionTracker:
    """Tracks scan session statistics and recent detection history."""

    KEY_HISTORY = "scan_session_history"

    @classmethod
    def add_scan(cls, label: str, confidence: float, latency_ms: float) -> None:
        if cls.KEY_HISTORY not in st.session_state:
            st.session_state[cls.KEY_HISTORY] = []

        st.session_state[cls.KEY_HISTORY].insert(
            0,
            {
                "label": label,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            },
        )
        # Keep last 20 scans
        st.session_state[cls.KEY_HISTORY] = st.session_state[cls.KEY_HISTORY][:20]

    @classmethod
    def get_history(cls) -> list[dict]:
        return st.session_state.get(cls.KEY_HISTORY, [])

    @classmethod
    def get_stats(cls) -> dict:
        history = cls.get_history()
        total = len(history)
        if total == 0:
            return {"total": 0, "recyclable_count": 0, "recyclable_pct": 0.0, "counts": {}}

        counts: dict[str, int] = {}
        recyclable_count = 0
        for item in history:
            lbl = item["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
            if is_recyclable(lbl):
                recyclable_count += 1

        return {
            "total": total,
            "recyclable_count": recyclable_count,
            "recyclable_pct": (recyclable_count / total) * 100,
            "counts": counts,
        }

    @classmethod
    def clear_history(cls) -> None:
        if cls.KEY_HISTORY in st.session_state:
            del st.session_state[cls.KEY_HISTORY]


# Compatibility aliases used by some views.
SessionTracker = SessionTracker
SnapshotStateManager = SnapshotStateManager

