"""
OpticBin — Webcam Session
=========================
Context-managed OpenCV capture with guaranteed release.
"""

from __future__ import annotations

from typing import Self

import sys

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when a webcam cannot be opened or read."""


class CameraSession:
    """Acquire, read, and always release an OpenCV VideoCapture device."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> Self:
        if sys.platform == "win32":
            self._capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if self._capture is None or not self._capture.isOpened():
                self._capture = cv2.VideoCapture(self.device_index)
        else:
            self._capture = cv2.VideoCapture(self.device_index)

        if self._capture is None or not self._capture.isOpened():
            self.release()
            raise CameraError(
                "Could not open webcam. Check that a camera is connected "
                "and not already in use."
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Webcam session is not open.")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to capture a frame from the webcam.")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
