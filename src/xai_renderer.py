"""
OpticBin — XAI Visualization & Rendering Engine
================================================
Decoupled visualization module for rendering Grad-CAM heatmaps,
blending overlays onto RGB frames, and formatting confidence overlays.
"""

from __future__ import annotations

import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

from config.settings import CAM_OPACITY


class CAMRenderer:
    """
    Handles blending and post-processing of class activation maps onto source images.
    """

    def __init__(self, default_opacity: float = CAM_OPACITY):
        self.default_opacity = default_opacity

    def blend(
        self,
        rgb_float: np.ndarray,
        grayscale_cam: np.ndarray,
        use_rgb: bool = True,
        opacity: float | None = None,
    ) -> np.ndarray:
        """
        Blend a 2D grayscale CAM activation map onto a normalized float32 RGB image.

        Parameters
        ----------
        rgb_float : np.ndarray
            Source RGB image array [H, W, 3] with values in range [0, 1].
        grayscale_cam : np.ndarray
            Grad-CAM activation map [H, W] with values in range [0, 1].
        use_rgb : bool
            Whether output frame should be in RGB (True) or BGR (False).
        opacity : float | None
            Custom blending ratio (uses default_opacity if None).

        Returns
        -------
        np.ndarray
            Blended uint8 RGB image array ready for display.
        """
        image_weight = 1.0 - (opacity if opacity is not None else self.default_opacity)
        return show_cam_on_image(
            rgb_float,
            grayscale_cam,
            use_rgb=use_rgb,
            image_weight=image_weight,
        )

    def draw_prediction_overlay(
        self,
        image_uint8: np.ndarray,
        label: str,
        confidence: float,
    ) -> np.ndarray:
        """
        Draw a subtle text badge onto an OpenCV image frame.
        """
        out = image_uint8.copy()
        text = f"{label.title()} ({confidence * 100:.1f}%)"
        cv2.putText(
            out,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out
