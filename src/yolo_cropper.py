"""
OpticBin — YOLO Object Detection ROI Cropper
============================================
Uses YOLOv8 (nano) to detect object bounding boxes in image/webcam inputs.
Crops strictly around the detected object (with optional padding) to remove
room, wall, and desk background clutter before feeding the cropped region
to OpticBin's 5-class waste classification engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

_YOLO_MODEL = None


def get_yolo_model():
    """Lazy-load the YOLOv8 nano model instance."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            # Prefer local models/weights/yolov8n.pt if present, else default to yolov8n.pt
            weights_file = Path("models/weights/yolov8n.pt")
            if weights_file.exists():
                _YOLO_MODEL = YOLO(str(weights_file))
            else:
                _YOLO_MODEL = YOLO("yolov8n.pt")
        except Exception as err:
            print(f"[YOLOCropper] Warning: Could not initialize YOLOv8: {err}")
            _YOLO_MODEL = False
    return _YOLO_MODEL if _YOLO_MODEL is not False else None


def crop_object_yolo(
    image: "PILImageType",
    conf_threshold: float = 0.25,
    padding_pct: float = 0.10,
) -> tuple["PILImageType", bool, Optional[Tuple[int, int, int, int]]]:
    """
    Detect the primary object in `image` using YOLOv8 and crop its bounding box.

    Returns:
        `(cropped_image, object_detected, bbox_coords)`
        where `bbox_coords` is `(xmin, ymin, xmax, ymax)` if detected, else None.
    """
    model = get_yolo_model()
    if model is None:
        return image, False, None

    try:
        # Convert PIL to uint8 RGB array
        img_arr = np.asarray(image.convert("RGB"))
        results = model.predict(img_arr, conf=conf_threshold, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return image, False, None

        boxes = results[0].boxes
        # Get highest-confidence box
        best_box_idx = int(boxes.conf.argmax())
        box_xyxy = boxes.xyxy[best_box_idx].cpu().numpy()  # [xmin, ymin, xmax, ymax]

        w, h = image.size
        xmin, ymin, xmax, ymax = box_xyxy

        # Apply padding
        bw = xmax - xmin
        bh = ymax - ymin
        pad_x = bw * padding_pct
        pad_y = bh * padding_pct

        crop_xmin = max(0, int(xmin - pad_x))
        crop_ymin = max(0, int(ymin - pad_y))
        crop_xmax = min(w, int(xmax + pad_x))
        crop_ymax = min(h, int(ymax + pad_y))

        cropped = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        return cropped, True, (crop_xmin, crop_ymin, crop_xmax, crop_ymax)

    except Exception as err:
        print(f"[YOLOCropper] Error during YOLO crop: {err}")
        return image, False, None
