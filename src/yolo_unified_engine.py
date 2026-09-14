"""
OpticBin - Unified Single-Stage YOLO Inference Engine
=====================================================
Combines object localization and 5-class material categorization into a single
forward pass (~15 ms). Detects bounding boxes, material categories, and confidence
scores simultaneously, supporting multi-item sorting and human exclusion.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO

from config.settings import CLASS_LABELS
from src.inference_engine import BaseInferenceEngine, PredictionResult

UNIFIED_WEIGHTS_PATH = Path("models/weights/yolov8_waste.pt")
FALLBACK_WEIGHTS_PATH = Path("yolov8s-worldv2.pt")


class YOLOUnifiedInferenceEngine(BaseInferenceEngine):
    """
    Unified Single-Stage Inference Engine.
    Runs a single YOLO detection forward pass and outputs both the bounding boxes
    and the 5 material categories simultaneously.
    """

    def __init__(self, weights_path: str | Path | None = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.supports_gradcam = False
        self.model_type = "yolov8_unified"

        # Check for fine-tuned waste weights
        if weights_path and Path(weights_path).exists():
            self.weights_path = Path(weights_path)
            self.using_finetuned_weights = True
        elif UNIFIED_WEIGHTS_PATH.exists():
            self.weights_path = UNIFIED_WEIGHTS_PATH
            self.using_finetuned_weights = True
        elif Path("yolov8s-worldv2.pt").exists():
            self.weights_path = Path("yolov8s-worldv2.pt")
            self.using_finetuned_weights = False
        else:
            self.weights_path = Path("yolov8n.pt")
            self.using_finetuned_weights = False

        try:
            self.model = YOLO(str(self.weights_path))
            print(f"[OK] YOLOUnifiedInferenceEngine loaded: {self.weights_path.name}")
        except Exception as err:
            print(f"[WARN] Failed to load {self.weights_path}: {err}. Loading base yolov8n.pt...")
            self.model = YOLO("yolov8n.pt")

    def predict(self, input_data: Any) -> PredictionResult:
        """
        Run forward pass on PIL Image or numpy array.
        Returns a standardized PredictionResult.
        """
        if isinstance(input_data, PILImage.Image):
            img_arr = np.asarray(input_data.convert("RGB"))
        elif isinstance(input_data, np.ndarray):
            img_arr = input_data
        elif hasattr(input_data, "cpu"):  # PyTorch tensor
            tensor_np = input_data.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_arr = (np.clip(tensor_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_arr = np.asarray(input_data)

        start = time.perf_counter()
        results = self.model.predict(
            img_arr,
            conf=self.conf_threshold,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        boxes = results[0].boxes if results and len(results[0].boxes) > 0 else []

        waste_detections = []
        person_boxes = []

        # Determine target vocabulary / classes
        raw_names = getattr(self.model, "names", {})
        active_classes = [raw_names[i].lower() for i in sorted(raw_names.keys())] if raw_names else CLASS_LABELS

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = raw_names.get(cls_id, f"class_{cls_id}").lower()
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].cpu().numpy()

            if cls_name == "person":
                person_boxes.append(xyxy)
            else:
                mapped_label = self._map_to_material_class(cls_name)
                waste_detections.append({
                    "label": mapped_label,
                    "raw_label": cls_name,
                    "conf": conf,
                    "xyxy": xyxy,
                })

        # Calculate probability distribution
        active_classes_clean = list(dict.fromkeys([self._map_to_material_class(c) for c in active_classes]))
        if not active_classes_clean:
            active_classes_clean = CLASS_LABELS
        probabilities = np.zeros(len(active_classes_clean), dtype=np.float32)

        # Filter out massive background boxes (room walls, doors) when tighter foreground boxes exist
        h_img, w_img = img_arr.shape[:2]
        img_area = max(1.0, float(w_img * h_img))
        has_focused_item = any(
            ((b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1])) < 0.50 * img_area
            for b in waste_detections
        )
        valid_detections = []
        for det in waste_detections:
            x1, y1, x2, y2 = det["xyxy"]
            box_area = (x2 - x1) * (y2 - y1)
            if has_focused_item and (box_area > 0.90 * img_area):
                continue
            valid_detections.append(det)

        if not valid_detections:
            valid_detections = waste_detections

        primary = None
        if valid_detections:
            # Pick the primary detection strictly by highest confidence
            primary = max(valid_detections, key=lambda d: d["conf"])
            class_label = primary["label"]
            confidence = primary["conf"]
            if class_label in active_classes_clean:
                class_id = active_classes_clean.index(class_label)
            else:
                class_id = 0

            probabilities[class_id] = confidence
            remaining = max(0.0, 1.0 - confidence)
            other_indices = [i for i in range(len(active_classes_clean)) if i != class_id]
            for idx in other_indices:
                probabilities[idx] = remaining / max(1, len(other_indices))
        else:
            class_label = "plastic"
            class_id = active_classes_clean.index("plastic") if "plastic" in active_classes_clean else 0
            confidence = 0.50
            probabilities = np.full(len(active_classes_clean), 1.0 / len(active_classes_clean), dtype=np.float32)

        # Create clean visualization overlay highlighting primary detection and suppressing overlapping labels
        overlay_img = img_arr.copy()
        if primary is not None:
            px1, py1, px2, py2 = [int(v) for v in primary["xyxy"]]
            color = (34, 197, 94)
            label_text = f"{primary['label'].upper()} ({primary['conf']*100:.0f}%)"
            cv2.rectangle(overlay_img, (px1, py1), (px2, py2), color, 3)

            # Crisp dark text pill background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            ty = max(26, py1 - 8)
            cv2.rectangle(overlay_img, (px1, ty - th - 6), (px1 + tw + 8, ty + 4), (0, 0, 0), -1)
            cv2.putText(overlay_img, label_text, (px1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Only draw distinct non-overlapping secondary items
            for det in valid_detections:
                if det is primary:
                    continue
                dx1, dy1, dx2, dy2 = [int(v) for v in det["xyxy"]]
                inter_w = max(0, min(px2, dx2) - max(px1, dx1))
                inter_h = max(0, min(py2, dy2) - max(py1, dy1))
                inter_area = inter_w * inter_h
                box2_area = max(1, (dx2 - dx1) * (dy2 - dy1))
                # Skip if it heavily overlaps the primary object (prevents text clashing)
                if (inter_area / box2_area) > 0.25:
                    continue

                sec_color = (16, 185, 129)
                sec_text = f"{det['label'].upper()} ({det['conf']*100:.0f}%)"
                cv2.rectangle(overlay_img, (dx1, dy1), (dx2, dy2), sec_color, 2)
                (stw, sth), _ = cv2.getTextSize(sec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                sty = max(24, dy1 - 6)
                cv2.rectangle(overlay_img, (dx1, sty - sth - 4), (dx1 + stw + 6, sty + 2), (0, 0, 0), -1)
                cv2.putText(overlay_img, sec_text, (dx1 + 3, sty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sec_color, 2)

        self._last_overlay = overlay_img
        self._last_primary_box = [int(v) for v in primary["xyxy"]] if primary is not None else None

        return PredictionResult(
            class_id=class_id,
            class_label=class_label,
            confidence=float(confidence),
            probabilities=probabilities,
            latency_ms=round(latency_ms, 2),
            class_names=active_classes_clean,
        )

    def explain(self, input_tensor_or_array: Any, rgb_float: np.ndarray) -> dict:
        """Unified explanation returns the bounding box detection overlay."""
        pred_res = self.predict(input_tensor_or_array)
        result = pred_res.to_dict()
        result["heatmap_overlay"] = getattr(self, "_last_overlay", None)
        result["primary_box"] = getattr(self, "_last_primary_box", None)
        result["gradcam_available"] = False
        return result

    def render_overlay(
        self,
        image_input: Any,
        box: list[int] | tuple[int, int, int, int] | None,
        label: str,
        confidence: float,
        subtitle: str | None = None,
    ) -> np.ndarray:
        """
        Renders a crisp bounding box with dark pill label on the given image.
        Supports supervisor correction badges without re-running inference.
        """
        if isinstance(image_input, PILImage.Image):
            overlay_img = np.asarray(image_input.convert("RGB")).copy()
        elif isinstance(image_input, np.ndarray):
            overlay_img = image_input.copy()
        else:
            overlay_img = np.asarray(image_input).copy()

        h, w = overlay_img.shape[:2]
        if box is not None and len(box) == 4:
            px1, py1, px2, py2 = [int(v) for v in box]
        else:
            margin_x, margin_y = int(w * 0.15), int(h * 0.15)
            px1, py1, px2, py2 = margin_x, margin_y, w - margin_x, h - margin_y

        color = (34, 197, 94)  # Emerald green
        cv2.rectangle(overlay_img, (px1, py1), (px2, py2), color, 3)

        label_text = f"{label.upper()} ({confidence*100:.0f}%)"
        if subtitle:
            label_text += f" - {subtitle.upper()}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        ty = max(26, py1 - 8)
        cv2.rectangle(overlay_img, (px1, ty - th - 6), (px1 + tw + 8, ty + 4), (0, 0, 0), -1)
        cv2.putText(overlay_img, label_text, (px1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        return overlay_img

    def predict_and_explain(self, input_tensor_or_array: Any, rgb_float: np.ndarray) -> dict:
        return self.explain(input_tensor_or_array, rgb_float)

    def _map_to_material_class(self, detected_name: str) -> str:
        """Helper to map YOLO class labels to standard OpticBin material categories."""
        name = detected_name.lower().strip()
        if name in ["biodegradable", "organic", "food", "compost", "fruit", "vegetable"]:
            return "biodegradable"
        if name in ["cardboard", "box", "carton"]:
            return "cardboard"
        if name in ["glass", "glass bottle", "jar", "wine glass"]:
            return "glass"
        if name in ["metal", "can", "soda can", "tin can", "aluminum foil"]:
            return "metal"
        if name in ["paper", "book", "newspaper", "magazine", "receipt"]:
            return "paper"
        if name in ["plastic", "bottle", "plastic bag", "container"]:
            return "plastic"
        return name
