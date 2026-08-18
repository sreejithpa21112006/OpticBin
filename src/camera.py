"""
OpticBin — Webcam Session & Threaded Frame Streamer
===================================================
Context-managed OpenCV capture with guaranteed resource cleanup and
background thread frame buffering for zero-latency live inference streams.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when a webcam cannot be opened or read."""


class CameraSession:
    """Acquire, read, and always release an OpenCV VideoCapture device."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> CameraSession:
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


class ThreadedCameraStream:
    """
    Background-threaded camera streamer that continuously drains the camera
    buffer to ensure zero-latency retrieval of the latest video frame.
    """

    def __init__(self, device_index: int = 0, fps_limit: float = 30.0):
        self.device_index = device_index
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        self._capture: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._running: bool = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> ThreadedCameraStream:
        if sys.platform == "win32":
            self._capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if self._capture is None or not self._capture.isOpened():
                self._capture = cv2.VideoCapture(self.device_index)
        else:
            self._capture = cv2.VideoCapture(self.device_index)

        if self._capture is None or not self._capture.isOpened():
            self.stop()
            raise CameraError(f"Could not open webcam device index {self.device_index}.")

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return self

    def _update_loop(self) -> None:
        while self._running and self._capture and self._capture.isOpened():
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest_frame = frame
            time.sleep(self.frame_interval * 0.5)

    def read_latest(self) -> np.ndarray:
        with self._lock:
            if self._latest_frame is None:
                raise CameraError("No frame available from threaded camera stream yet.")
            return self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> ThreadedCameraStream:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
