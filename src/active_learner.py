"""
OpticBin — Active Learning Engine
===================================
Uses Gemini Vision (multimodal) to cross-check low-confidence ML predictions.
Disagreements are logged to review_queue/ as structured folders containing
the original image and metadata JSON, ready for human review and retraining.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import (
    ACTIVE_LEARNING_CONFIDENCE_THRESHOLD,
    CLASS_LABELS,
    LLM_MODEL,
    REVIEW_QUEUE_DIR,
)

_VISION_PROMPT = """\
You are an expert waste classification assistant for an industrial recycling system.

An Edge-AI model has classified the waste item in this image:
- ML model prediction: {ml_label}
- ML model confidence: {ml_confidence:.1f}%

Please independently classify the waste item you see in the image.
Respond ONLY with a JSON object in this exact format (no other text):
{{
  "label": "<one of: cardboard, glass, metal, paper, plastic>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence explaining the key visual feature>"
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Active Learner
# ──────────────────────────────────────────────────────────────────────────────

class ActiveLearner:
    """
    Wraps Gemini Vision to provide independent second-opinion classification
    for low-confidence ML predictions, logging disagreements for retraining.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or _get_api_key()
        self._client = None
        self._available = False
        self.queue_dir = Path(REVIEW_QUEUE_DIR)
        self._init_client()

    def _init_client(self) -> None:
        if not self._api_key:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._client = genai
            self._available = True
        except ImportError:
            pass
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    def should_cross_check(self, result: dict) -> bool:
        """
        Returns True when the ML confidence is below the active learning
        threshold and Gemini Vision should be invoked.
        """
        return result.get("confidence", 1.0) < ACTIVE_LEARNING_CONFIDENCE_THRESHOLD

    def verify_with_gemini_vision(
        self,
        image_pil: Any,  # PIL.Image
        ml_result: dict,
    ) -> dict | None:
        """
        Send the image + ML result to Gemini Vision for independent classification.

        Returns a dict:
            {
                "gemini_label": str,
                "gemini_confidence": float,
                "gemini_reasoning": str,
                "agreement": bool,
                "ml_label": str,
                "ml_confidence": float,
            }
        or None on API failure.
        """
        if not self.is_available or image_pil is None:
            return None

        ml_label = ml_result.get("class_label", "unknown")
        ml_confidence = ml_result.get("confidence", 0.0) * 100

        prompt = _VISION_PROMPT.format(
            ml_label=ml_label,
            ml_confidence=ml_confidence,
        )

        try:
            import google.generativeai as genai
            from PIL import Image as PILImage
            import io

            # Convert PIL image to bytes for the Gemini multimodal API
            buf = io.BytesIO()
            image_pil.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            image_part = {
                "mime_type": "image/jpeg",
                "data": buf.read(),
            }

            model = self._client.GenerativeModel(LLM_MODEL)
            response = model.generate_content([prompt, image_part])
            raw = response.text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            gemini_label = parsed.get("label", "unknown").lower().strip()

            # Validate label is a known class
            if gemini_label not in CLASS_LABELS:
                gemini_label = "unknown"

            gemini_confidence = float(parsed.get("confidence", 0.0))
            gemini_reasoning = parsed.get("reasoning", "")
            agreement = gemini_label == ml_label

            return {
                "gemini_label": gemini_label,
                "gemini_confidence": gemini_confidence,
                "gemini_reasoning": gemini_reasoning,
                "agreement": agreement,
                "ml_label": ml_label,
                "ml_confidence": ml_result.get("confidence", 0.0),
            }

        except Exception as exc:
            print(f"[ActiveLearner] Vision cross-check failed: {exc}")
            return None

    def log_to_review_queue(
        self,
        image_pil: Any,
        ml_result: dict,
        gemini_result: dict,
    ) -> Path | None:
        """
        Save flagged image + metadata to review_queue/<timestamp_label>/.
        Returns the folder path, or None on failure.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ml_label = ml_result.get("class_label", "unknown")
            folder_name = f"{timestamp}_{ml_label}"
            folder = self.queue_dir / folder_name
            folder.mkdir(parents=True, exist_ok=True)

            # Save image
            image_pil.save(folder / "image.jpg", format="JPEG", quality=90)

            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "ml_label": ml_label,
                "ml_confidence": float(ml_result.get("confidence", 0.0)),
                "ml_probabilities": {
                    cls: float(prob)
                    for cls, prob in zip(
                        CLASS_LABELS,
                        ml_result.get("probabilities", [0.0] * len(CLASS_LABELS)),
                    )
                },
                "gemini_label": gemini_result.get("gemini_label"),
                "gemini_confidence": gemini_result.get("gemini_confidence"),
                "gemini_reasoning": gemini_result.get("gemini_reasoning"),
                "agreement": gemini_result.get("agreement", True),
                "verified_label": None,  # filled in by human reviewer
                "reviewed": False,
            }

            with open(folder / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            return folder

        except Exception as exc:
            print(f"[ActiveLearner] Failed to log to review queue: {exc}")
            return None

    def get_queue_stats(self) -> dict:
        """
        Scan the review_queue/ directory and return summary statistics.
        """
        if not self.queue_dir.exists():
            return {
                "total": 0,
                "agreements": 0,
                "disagreements": 0,
                "reviewed": 0,
                "by_ml_label": {},
                "agreement_rate": 0.0,
            }

        total = agreements = disagreements = reviewed = 0
        by_ml_label: dict[str, int] = {}

        for meta_path in self.queue_dir.glob("*/metadata.json"):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                total += 1
                if meta.get("agreement"):
                    agreements += 1
                else:
                    disagreements += 1
                if meta.get("reviewed"):
                    reviewed += 1
                lbl = meta.get("ml_label", "unknown")
                by_ml_label[lbl] = by_ml_label.get(lbl, 0) + 1
            except Exception:
                continue

        return {
            "total": total,
            "agreements": agreements,
            "disagreements": disagreements,
            "reviewed": reviewed,
            "by_ml_label": by_ml_label,
            "agreement_rate": (agreements / total * 100) if total else 0.0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    """Resolve API key from secrets, env var, or session state."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    return st.session_state.get("_gemini_api_key")
