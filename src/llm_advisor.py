"""
OpticBin — Gemini LLM Advisor
================================
Wraps the Google Gemini 2.0 Flash Lite API for three capabilities:
  1. stream_recycling_advice()  — post-classification disposal guidance
  2. stream_xai_explanation()   — plain-English Grad-CAM interpretation
  3. chat()                     — multi-turn recycling Q&A chatbot
"""

from __future__ import annotations

import os
from typing import Generator, Iterator

import streamlit as st

from config.settings import CLASS_LABELS, LLM_MODEL, WASTE_METADATA

# ──────────────────────────────────────────────────────────────────────────────
# API Key Resolution
# ──────────────────────────────────────────────────────────────────────────────

def get_api_key() -> str | None:
    """
    Resolve Gemini API key with the following priority:
      1. st.secrets["GEMINI_API_KEY"]  (Streamlit Cloud / secrets.toml)
      2. GEMINI_API_KEY environment variable
      3. st.session_state["_gemini_api_key"]  (entered by user in sidebar)
    Returns None if no key is found.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────────────────────────────────────

_RECYCLING_ADVICE_PROMPT = """\
You are OpticBin's AI Recycling Advisor — an expert on waste management, \
environmental science, and recycling best practices.

The OpticBin Edge-AI system has just classified an item with these results:
- Classified as: {label} ({label_upper})
- Model confidence: {confidence:.1f}%
- Full probability distribution:
{prob_table}

{confidence_note}

Provide clear, actionable recycling advice for this item. Include:
1. The correct disposal bin/method (be specific)
2. Any preparation needed before recycling (rinsing, separating, flattening, etc.)
3. One surprising or motivating environmental fact about this material
4. If the model confidence is low, note any similar materials to double-check against

Keep the response concise (under 150 words), friendly, and practical. \
Do not repeat the material name more than twice. No bullet points — use \
natural flowing sentences.
"""

_XAI_EXPLANATION_PROMPT = """\
You are OpticBin's AI Explainability Advisor, helping non-experts understand \
deep learning model decisions.

The OpticBin Grad-CAM heatmap highlighted specific regions of the image when \
classifying this waste item:
- Classification: {label} ({confidence:.1f}% confidence)
- Model architecture: {model_type}
- Grad-CAM available: Yes

Explain in plain English what visual features a {model_type} model typically \
focuses on when identifying {label}:
- What textures, shapes, or colors are characteristic of {label}?
- Why might the model activate on these specific regions?
- What could cause it to confuse {label} with {second_label} ({second_prob:.1f}%)?

Keep it under 100 words. Use language a non-technical user can understand. \
No jargon.
"""

_CHATBOT_SYSTEM_PROMPT = """\
You are OpticBin's Recycling Assistant — a friendly, knowledgeable expert on \
waste management, sustainability, and recycling for the OpticBin Edge-AI system.

OpticBin classifies 5 waste categories: cardboard, glass, metal, paper, plastic.

Answer questions about:
- How to recycle or dispose of specific items
- Environmental impact of different materials
- Recycling rules and common misconceptions
- How the AI model works (if asked)

Keep answers concise (under 120 words), practical, and conversational. \
If unsure, say so — don't make up recycling rules that vary by region.
"""


# ──────────────────────────────────────────────────────────────────────────────
# GeminiAdvisor Class
# ──────────────────────────────────────────────────────────────────────────────

class GeminiAdvisor:
    """
    Thin wrapper around the Gemini 2.0 Flash Lite API for OpticBin.
    All public methods return None or empty iterator on failure so the
    UI can degrade gracefully without crashing.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_api_key()
        self._client = None
        self._available = False
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
        """True when the Gemini client is initialised and an API key is present."""
        return self._available and self._client is not None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def stream_recycling_advice(self, result: dict) -> Iterator[str] | None:
        """
        Stream post-classification recycling guidance for the predicted class.
        Injects the full probability distribution so Gemini can tailor advice
        to the model's confidence level.

        Args:
            result: dict with keys class_label, confidence, probabilities

        Yields:
            str chunks of the streamed response, or None on failure.
        """
        if not self.is_available:
            return None

        label = result.get("class_label", "unknown")
        confidence = result.get("confidence", 0.0) * 100
        probabilities = result.get("probabilities", [])

        # Build probability table for context
        prob_rows = []
        if len(probabilities) == len(CLASS_LABELS):
            pairs = sorted(zip(CLASS_LABELS, probabilities), key=lambda x: -x[1])
            for cls, prob in pairs:
                bar = "█" * int(prob * 20)
                prob_rows.append(f"  {cls:<12} {prob * 100:5.1f}%  {bar}")
        prob_table = "\n".join(prob_rows) if prob_rows else "  (unavailable)"

        # Confidence-based tone note
        if confidence >= 90:
            confidence_note = "The model is highly confident in this classification."
        elif confidence >= 70:
            confidence_note = "The model is moderately confident. The classification is likely correct."
        else:
            confidence_note = (
                f"The model confidence is relatively low ({confidence:.1f}%). "
                "The item may have mixed material properties. Advise the user to "
                "double-check by looking at the second most probable class."
            )

        prompt = _RECYCLING_ADVICE_PROMPT.format(
            label=label,
            label_upper=label.upper(),
            confidence=confidence,
            prob_table=prob_table,
            confidence_note=confidence_note,
        )

        try:
            model = self._client.GenerativeModel(LLM_MODEL)
            return model.generate_content(prompt, stream=True)
        except Exception:
            return None

    def stream_xai_explanation(self, result: dict, model_type: str = "EfficientNetV2") -> Iterator[str] | None:
        """
        Stream a plain-English explanation of what the Grad-CAM heatmap means.

        Args:
            result:     classification result dict
            model_type: backbone name string for context

        Yields:
            str chunks of the streamed response, or None on failure.
        """
        if not self.is_available:
            return None

        label = result.get("class_label", "unknown")
        confidence = result.get("confidence", 0.0) * 100
        probabilities = result.get("probabilities", [])

        # Second most likely class for confusion explanation
        second_label, second_prob = "unknown", 0.0
        if len(probabilities) == len(CLASS_LABELS):
            pairs = sorted(zip(CLASS_LABELS, probabilities), key=lambda x: -x[1])
            if len(pairs) > 1:
                second_label = pairs[1][0]
                second_prob = pairs[1][1] * 100

        prompt = _XAI_EXPLANATION_PROMPT.format(
            label=label,
            confidence=confidence,
            model_type=model_type,
            second_label=second_label,
            second_prob=second_prob,
        )

        try:
            model = self._client.GenerativeModel(LLM_MODEL)
            return model.generate_content(prompt, stream=True)
        except Exception:
            return None

    def chat(self, message: str, history: list[dict]) -> Iterator[str] | None:
        """
        Send a message to a multi-turn recycling Q&A session.

        Args:
            message: the user's latest message
            history: list of {"role": "user"|"model", "parts": [str]} dicts

        Yields:
            str chunks of the streamed response, or None on failure.
        """
        if not self.is_available:
            return None

        try:
            model = self._client.GenerativeModel(
                LLM_MODEL,
                system_instruction=_CHATBOT_SYSTEM_PROMPT,
            )
            chat_session = model.start_chat(history=history)
            return chat_session.send_message(message, stream=True)
        except Exception:
            return None
