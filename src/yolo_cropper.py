"""
OpticBin - Multi-Class Waste Object Detection and Cropper
=========================================================
Uses YOLO-World (open-vocabulary) with a curated multi-class waste vocabulary
spanning all 5 OpticBin material categories:
  - Cardboard: cardboard box, carton, cereal box, egg carton, corrugated sheet
  - Glass: glass bottle, wine glass, jar, drinking glass, glass bowl
  - Metal: soda can, tin can, beverage can, metal bottle, aluminum foil, lid
  - Paper: sheet of paper, notebook, book, receipt, paper cup, paper bag, tissue
  - Plastic: pen, marker, highlighter, plastic bottle, plastic cup, wrapper, container

When a human is present in the frame:
1. Recognizes and strictly excludes the human ('person').
2. Rejects background furniture (tables, chairs, walls) from being selected as waste.
3. Locates and isolates the specific waste object presented in the person's hand.
4. If the item is partially occluded, uses handheld foreground vision extraction.
5. Crops strictly around the detected item with padding before feeding into
   OpticBin's 5-class classification and Grad-CAM engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

_YOLO_MODEL = None
_IS_YOLO_WORLD = False

# Comprehensive vocabulary across all 5 OpticBin recycling categories + person
WASTE_CATEGORY_MAP = {
    # Plastic items
    "pen": "plastic",
    "marker": "plastic",
    "highlighter": "plastic",
    "plastic bottle": "plastic",
    "bottle": "plastic",
    "plastic cup": "plastic",
    "cup": "plastic",
    "tub": "plastic",
    "plastic tub": "plastic",
    "container": "plastic",
    "plastic container": "plastic",
    "food container": "plastic",
    "plastic box": "plastic",
    "plastic bag": "plastic",
    "plastic wrapper": "plastic",
    "straw": "plastic",
    "plastic cutlery": "plastic",
    "plastic packaging": "plastic",
    "blister pack": "plastic",
    "plastic": "plastic",

    # Paper items
    "paper": "paper",
    "sheet of paper": "paper",
    "crumpled paper": "paper",
    "notebook": "paper",
    "book": "paper",
    "receipt": "paper",
    "newspaper": "paper",
    "magazine": "paper",
    "paper cup": "paper",
    "paper bag": "paper",
    "tissue": "paper",
    "envelope": "paper",

    # Cardboard items
    "cardboard": "cardboard",
    "cardboard box": "cardboard",
    "box": "cardboard",
    "carton": "cardboard",
    "cereal box": "cardboard",
    "egg carton": "cardboard",
    "corrugated cardboard": "cardboard",
    "cardboard scrap": "cardboard",

    # Metal items
    "can": "metal",
    "soda can": "metal",
    "tin can": "metal",
    "beverage can": "metal",
    "metal bottle": "metal",
    "aluminum foil": "metal",
    "aerosol can": "metal",
    "metal lid": "metal",
    "bottle cap": "metal",
    "metal cutlery": "metal",
    "metal": "metal",

    # Glass items
    "glass bottle": "glass",
    "wine glass": "glass",
    "glass jar": "glass",
    "glass cup": "glass",
    "drinking glass": "glass",
    "glass container": "glass",
    "glass bowl": "glass",
    "glass": "glass",
}

# Ordered vocabulary list for YOLO-World open-vocabulary initialization
WASTE_VOCABULARY = ["person"] + list(WASTE_CATEGORY_MAP.keys())

# Background room furniture classes - strictly prohibited from being selected
# as waste targets when a human is present in the frame.
FURNITURE_BACKGROUND_CLASSES = {
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "clock",
    "potted plant",
    "bench",
    "desk",
}


def get_yolo_model():
    """
    Lazy-load the open-vocabulary YOLO-World model with custom waste vocabulary.
    Falls back to local yolov8n.pt if YOLO-World is not yet cached or offline.
    """
    global _YOLO_MODEL, _IS_YOLO_WORLD
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO

            # Try loading YOLO-World open-vocabulary model first
            try:
                model = YOLO("yolov8s-worldv2.pt")
                model.set_classes(WASTE_VOCABULARY)
                _YOLO_MODEL = model
                _IS_YOLO_WORLD = True
                print("[YOLOCropper] YOLO-World initialized with multi-class waste vocabulary.")
                return _YOLO_MODEL
            except Exception as world_err:
                print(f"[YOLOCropper] YOLO-World unavailable ({world_err}), using local YOLOv8...")

            weights_file = Path("models/weights/yolov8n.pt")
            if weights_file.exists():
                _YOLO_MODEL = YOLO(str(weights_file))
            else:
                _YOLO_MODEL = YOLO("yolov8n.pt")
            _IS_YOLO_WORLD = False
        except Exception as err:
            print(f"[YOLOCropper] Warning: Could not initialize YOLO: {err}")
            _YOLO_MODEL = False
            _IS_YOLO_WORLD = False
    return _YOLO_MODEL if _YOLO_MODEL is not False else None


def _compute_overlap_ratio(
    box_a: np.ndarray,
    box_b: np.ndarray,
) -> float:
    """Return intersection area divided by area of box_a."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, float(ix2 - ix1))
    inter_h = max(0.0, float(iy2 - iy1))
    inter_area = inter_w * inter_h
    a_area = max(1.0, float((ax2 - ax1) * (ay2 - ay1)))
    return inter_area / a_area


def _score_candidate(
    box_xyxy: np.ndarray,
    conf: float,
    cls_name: str,
    human_detected: bool,
    person_boxes: list[np.ndarray],
    img_shape: tuple[int, int],
) -> float:
    """
    Score a non-person candidate box based on:
    - Detection confidence
    - Waste category relevance
    - Spatial interaction with detected humans (held/presented items)
    - Reasonable size bounds
    """
    score = conf
    img_h, img_w = img_shape
    box_area = max(1.0, float((box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])))
    area_ratio = box_area / (img_w * img_h)

    # 1. Category relevance
    if cls_name in WASTE_CATEGORY_MAP:
        score += 0.50
    elif cls_name in FURNITURE_BACKGROUND_CLASSES:
        score -= 0.60

    # 2. Human interaction (item held by or in front of human)
    if human_detected and person_boxes:
        max_overlap = max(
            _compute_overlap_ratio(box_xyxy, p_box)
            for p_box in person_boxes
        )
        if max_overlap > 0.10:
            # Overlapping human body/hand -> strongly prioritize
            score += 0.45
        elif max_overlap < 0.05 and cls_name in FURNITURE_BACKGROUND_CLASSES:
            # Distant background furniture
            score -= 0.50

    # 3. Size realism
    if area_ratio > 0.70:
        score -= 0.40
    elif 0.003 <= area_ratio <= 0.60:
        score += 0.10

    return score


def _detect_handheld_foreground_object(
    img_bgr: np.ndarray,
    person_box: np.ndarray,
) -> Optional[dict]:
    """
    When YOLO does not return a direct bounding box for the held item,
    extracts the foreground handheld object from the person's interaction zone.
    """
    img_h, img_w = img_bgr.shape[:2]
    px1, py1, px2, py2 = [int(v) for v in person_box]
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    # Interaction zone: horizontal center 60% of the person,
    # vertically from below eye-level (25% down) down to hands (100%)
    ix1 = max(0, int(px1 + 0.18 * pw))
    ix2 = min(img_w, int(px2 - 0.18 * pw))
    iy1 = max(0, int(py1 + 0.25 * ph))
    iy2 = min(img_h, int(py2))

    if (ix2 - ix1) < 40 or (iy2 - iy1) < 40:
        return None

    roi = img_bgr[iy1:iy2, ix1:ix2]
    roi_h, roi_w = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (roi_w * roi_h) * 0.010
    max_area = (roi_w * roi_h) * 0.85
    valid_cnts = [c for c in cnts if min_area <= cv2.contourArea(c) <= max_area]

    if valid_cnts:
        valid_cnts.sort(key=cv2.contourArea, reverse=True)
        main_cnts = valid_cnts[:4]
        all_pts = np.vstack(main_cnts)
        bx, by, bw, bh = cv2.boundingRect(all_pts)

        ox1 = ix1 + bx
        oy1 = iy1 + by
        ox2 = ix1 + bx + bw
        oy2 = iy1 + by + bh

        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)
        ox1 = max(0, ox1 - pad_x)
        oy1 = max(0, oy1 - pad_y)
        ox2 = min(img_w, ox2 + pad_x)
        oy2 = min(img_h, oy2 + pad_y)

        return {
            "index": -1,
            "cls_id": -1,
            "cls_name": "handheld item",
            "conf": 0.80,
            "xyxy": np.array([ox1, oy1, ox2, oy2], dtype=np.float32),
            "score": 0.90,
            "category": "waste item",
            "is_handheld_extracted": True,
        }

    # Fallback to focused center-lower interaction zone
    cx = (ix1 + ix2) // 2
    cy = int(iy1 + (iy2 - iy1) * 0.55)
    hw = int((ix2 - ix1) * 0.40)
    hh = int((iy2 - iy1) * 0.40)
    ox1 = max(0, cx - hw)
    oy1 = max(0, cy - hh)
    ox2 = min(img_w, cx + hw)
    oy2 = min(img_h, cy + hh)

    return {
        "index": -1,
        "cls_id": -1,
        "cls_name": "handheld item",
        "conf": 0.70,
        "xyxy": np.array([ox1, oy1, ox2, oy2], dtype=np.float32),
        "score": 0.70,
        "category": "waste item",
        "is_handheld_extracted": True,
    }


def _find_target_object(
    model,
    img_arr: np.ndarray,
    conf_threshold: float = 0.20,
) -> tuple[
    Optional[dict],
    bool,
    list[dict],
    list[dict],
]:
    """
    Run YOLO and select the target object while excluding humans and background clutter.

    Returns:
        (best_target_dict, human_detected, person_boxes_list, all_candidate_boxes_list)
    """
    img_h, img_w = img_arr.shape[:2]
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    results = model.predict(img_arr, conf=conf_threshold, verbose=False)

    boxes = results[0].boxes if results and len(results[0].boxes) > 0 else []
    person_list: list[dict] = []
    candidate_list: list[dict] = []

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        cls_name = model.names.get(cls_id, f"obj_{cls_id}").lower()
        conf = float(boxes.conf[i])
        xyxy = boxes.xyxy[i].cpu().numpy()

        category = WASTE_CATEGORY_MAP.get(cls_name, "general")

        info = {
            "index": i,
            "cls_id": cls_id,
            "cls_name": cls_name,
            "conf": conf,
            "xyxy": xyxy,
            "category": category,
        }

        if cls_id == 0 or cls_name == "person":
            person_list.append(info)
        else:
            candidate_list.append(info)

    human_detected = len(person_list) > 0
    person_boxes_xyxy = [p["xyxy"] for p in person_list]

    for cand in candidate_list:
        cand["score"] = _score_candidate(
            cand["xyxy"],
            cand["conf"],
            cand["cls_name"],
            human_detected,
            person_boxes_xyxy,
            (img_h, img_w),
        )

    # When a human is present:
    # 1. Background furniture is strictly disqualified from being the waste target.
    # 2. Candidate objects must have a positive score.
    if human_detected:
        eligible_candidates = [
            c for c in candidate_list
            if c["cls_name"] not in FURNITURE_BACKGROUND_CLASSES
            and c["score"] >= 0.20
        ]

        if eligible_candidates:
            best_target = max(eligible_candidates, key=lambda c: c["score"])
            return best_target, human_detected, person_list, candidate_list

        # If no recognized bounding box, use handheld foreground vision extraction
        primary_person = max(person_list, key=lambda p: float(p["conf"]))
        handheld_target = _detect_handheld_foreground_object(img_bgr, primary_person["xyxy"])

        if handheld_target is not None:
            return handheld_target, human_detected, person_list, candidate_list

        return None, human_detected, person_list, candidate_list

    # No human in scene (e.g. object on table or conveyor)
    if candidate_list:
        best_target = max(candidate_list, key=lambda c: c["score"])
        return best_target, human_detected, person_list, candidate_list

    return None, False, [], []


def crop_object_yolo(
    image: "PILImageType",
    conf_threshold: float = 0.20,
    padding_pct: float = 0.10,
) -> tuple["PILImageType", bool, Optional[Tuple[int, int, int, int]]]:
    """
    Detect the primary waste object in `image` using YOLO and crop its bounding box.
    When a human is present:
    - Excludes the human
    - Rejects background furniture
    - Isolates the handheld waste item in front of the human
    """
    model = get_yolo_model()
    if model is None:
        return image, False, None

    try:
        img_arr = np.asarray(image.convert("RGB"))
        best_target, _, _, _ = _find_target_object(model, img_arr, conf_threshold=conf_threshold)

        if best_target is None:
            return image, False, None

        w, h = image.size
        xmin, ymin, xmax, ymax = best_target["xyxy"]

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


def detect_and_annotate_yolo(
    image: "PILImageType",
    conf_threshold: float = 0.20,
) -> tuple["PILImageType", bool, str, float]:
    """
    Run YOLO object detection on `image` and return an annotated image:
    - Detected humans are visually tagged as 'Person (Excluded)' with an amber box
    - Target waste object is highlighted with a green bounding box + material category
    - Background objects are rendered in subtle gray
    """
    model = get_yolo_model()
    if model is None:
        return image, False, "", 0.0

    try:
        img_arr = np.asarray(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        best_target, human_detected, person_list, candidate_list = _find_target_object(
            model, img_arr, conf_threshold=conf_threshold
        )

        # 1. Draw detected humans (Amber border, marked as Excluded)
        for p in person_list:
            px1, py1, px2, py2 = [int(v) for v in p["xyxy"]]
            cv2.rectangle(img_bgr, (px1, py1), (px2, py2), (0, 140, 255), 2)
            label = f"Person (Excluded {p['conf'] * 100:.0f}%)"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(
                img_bgr,
                (px1, max(0, py1 - 18)),
                (px1 + t_size[0] + 6, max(18, py1)),
                (0, 140, 255),
                -1,
            )
            cv2.putText(
                img_bgr,
                label,
                (px1 + 3, max(14, py1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw background/unselected items (subtle gray, marked as bg)
        for cand in candidate_list:
            if best_target is not None and cand.get("index") == best_target.get("index"):
                continue
            cx1, cy1, cx2, cy2 = [int(v) for v in cand["xyxy"]]
            cv2.rectangle(img_bgr, (cx1, cy1), (cx2, cy2), (100, 100, 100), 1)
            c_label = f"{cand['cls_name']} (bg)"
            cv2.putText(
                img_bgr,
                c_label,
                (cx1 + 2, max(12, cy1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw primary target object (Bright Green)
        if best_target is not None:
            tx1, ty1, tx2, ty2 = [int(v) for v in best_target["xyxy"]]
            cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), (0, 220, 100), 3)

            if best_target.get("is_handheld_extracted"):
                target_label = "Target: Handheld Waste Item"
            else:
                target_label = f"Target: {best_target['cls_name'].title()} ({best_target['conf'] * 100:.1f}%)"

            t_size = cv2.getTextSize(target_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
            cv2.rectangle(
                img_bgr,
                (tx1, max(0, ty1 - 22)),
                (tx1 + t_size[0] + 8, max(22, ty1)),
                (0, 200, 80),
                -1,
            )
            cv2.putText(
                img_bgr,
                target_label,
                (tx1 + 4, max(16, ty1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return (
                PILImage.fromarray(annotated_rgb),
                True,
                best_target["cls_name"],
                float(best_target["conf"]),
            )

        # 4. If human detected but NO target object found: show guidance banner
        if human_detected:
            cv2.rectangle(img_bgr, (0, 0), (w, 32), (20, 20, 20), -1)
            guidance = "Human detected (Excluded) - Hold waste item toward camera"
            cv2.putText(
                img_bgr,
                guidance,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 180, 255),
                1,
                cv2.LINE_AA,
            )
            annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return (
                PILImage.fromarray(annotated_rgb),
                False,
                "Person (Excluded - No Item Found)",
                0.0,
            )

        annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(annotated_rgb), False, "", 0.0

    except Exception as err:
        print(f"[YOLOCropper] Error during YOLO annotation: {err}")
        return image, False, "", 0.0
