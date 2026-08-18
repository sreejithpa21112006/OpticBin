"""
OpticBin — Explainability (XAI) Engine
========================================
Unified XAI engine wrapping timm backbones with Grad-CAM for
real-time heatmap generation and classification.
"""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config.settings import CLASS_LABELS, DEVICE, NUM_CLASSES
from src.model_factory import create_backbone, get_cam_target_layers, load_checkpoint


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
        self.device = torch.device(DEVICE)
        self.model_type = model_type
        self.class_labels = CLASS_LABELS
        self.weights_path = weights_path
        self.using_finetuned_weights = weights_path is not None

        self.model = create_backbone(
            model_type,
            num_classes=num_classes,
            pretrained=not self.using_finetuned_weights,
        )

        if weights_path is not None:
            self.model = load_checkpoint(self.model, weights_path, map_location="cpu")

        self.model.eval().to(self.device)
        self.target_layers = get_cam_target_layers(self.model, model_type)
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def predict(self, input_tensor: torch.Tensor) -> dict:
        """
        Classification-only forward pass.

        Skips the Grad-CAM backward pass, which dominates frame cost, so
        live streams can classify every frame and explain periodically.
        """
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)

        return self._build_result(logits)

    def explain(
        self,
        input_tensor: torch.Tensor,
        rgb_float: np.ndarray,
    ) -> dict:
        """Grad-CAM heatmap generation plus the classification result."""
        input_tensor = input_tensor.to(self.device)

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]

        # Grad-CAM's internal forward already produced the logits.
        logits = getattr(self.cam, "outputs", None)
        if logits is None:
            with torch.no_grad():
                logits = self.model(input_tensor)

        result = self._build_result(logits)
        result["heatmap_overlay"] = show_cam_on_image(
            rgb_float,
            grayscale_cam,
            use_rgb=True,
        )
        return result

    def predict_and_explain(
        self,
        input_tensor: torch.Tensor,
        rgb_float: np.ndarray,
    ) -> dict:
        """Combined forward pass + Grad-CAM heatmap generation."""
        return self.explain(input_tensor, rgb_float)

    def _build_result(self, logits: torch.Tensor) -> dict:
        probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()[0]
        pred_class_id = int(np.argmax(probabilities))

        return {
            "class_id": pred_class_id,
            "class_label": self.class_labels[pred_class_id],
            "confidence": float(probabilities[pred_class_id]),
            "probabilities": probabilities,
        }

    def switch_model(self, model_type: str, weights_path: str | None = None):
        """Hot-swap the backbone without restarting the application."""
        if hasattr(self, "cam"):
            del self.cam
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.__init__(model_type=model_type, weights_path=weights_path)

