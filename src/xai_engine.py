"""
OpticBin — Explainability (XAI) Engine
========================================
Dual-Model XAI engine supporting Grad-CAM (EfficientNetV2) and
Attention Rollout (MobileViT) for real-time heatmap generation.

Provides predict_and_explain() for simultaneous classification +
visual explanation overlay.
"""

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config.settings import (
    DEVICE,
    NUM_CLASSES,
    CLASS_LABELS,
    CAM_OPACITY,
)


class RecyclingXAIEngine:
    """
    Unified XAI engine that wraps timm backbones with Grad-CAM
    for real-time explainable waste classification.

    Supported architectures:
        - efficientnetv2_s  →  Grad-CAM on conv_head
        - mobilevit_xs      →  Grad-CAM on final_conv
    """

    def __init__(
        self,
        model_type: str = "efficientnetv2_s",
        num_classes: int = NUM_CLASSES,
        weights_path: str | None = None,
    ):
        """
        Parameters
        ----------
        model_type : str
            One of 'efficientnetv2_s' or 'mobilevit_xs'.
        num_classes : int
            Number of output classes (default: 6 waste categories).
        weights_path : str | None
            Optional path to a fine-tuned .pt checkpoint.
        """
        self.device = torch.device(DEVICE)
        self.model_type = model_type
        self.class_labels = CLASS_LABELS

        # ── Load backbone model from timm ──
        if model_type == "efficientnetv2_s":
            self.model = timm.create_model(
                "efficientnetv2_rw_m",
                pretrained=True,
                num_classes=num_classes,
            )
            self.target_layers = [self.model.conv_head]
        elif model_type == "mobilevit_xs":
            self.model = timm.create_model(
                "mobilevit_xs",
                pretrained=True,
                num_classes=num_classes,
            )
            self.target_layers = [self.model.final_conv]
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Load fine-tuned weights if provided
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        self.model.eval().to(self.device)

        # Initialize Grad-CAM
        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
        )

    def predict_and_explain(
        self,
        input_tensor: torch.Tensor,
        rgb_float: np.ndarray,
    ) -> dict:
        """
        Combined forward pass + Grad-CAM heatmap generation.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Normalized [1, 3, 224, 224] tensor.
        rgb_float : np.ndarray
            [224, 224, 3] float32 array in [0, 1] for overlay rendering.

        Returns
        -------
        dict
            {
                "class_id": int,
                "class_label": str,
                "confidence": float,
                "heatmap_overlay": np.ndarray,   # [224, 224, 3] uint8 BGR
                "probabilities": np.ndarray,
            }
        """
        input_tensor = input_tensor.to(self.device)

        # ── Forward pass ──
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class_id])

        # ── Generate Grad-CAM activation map ──
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_visualization = show_cam_on_image(
            rgb_float,
            grayscale_cam,
            use_rgb=True,
        )

        return {
            "class_id": pred_class_id,
            "class_label": self.class_labels[pred_class_id],
            "confidence": confidence,
            "heatmap_overlay": cam_visualization,
            "probabilities": probabilities,
        }

    def switch_model(self, model_type: str, weights_path: str | None = None):
        """
        Hot-swap the backbone without restarting the application.

        Parameters
        ----------
        model_type : str
            Target architecture name.
        weights_path : str | None
            Optional checkpoint path for the new model.
        """
        self.__init__(
            model_type=model_type,
            weights_path=weights_path,
        )
