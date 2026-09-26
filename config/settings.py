"""
OpticBin — Central Configuration
=================================
Class labels, input resolutions, device configs, and runtime parameters.
YAML overrides are loaded from config/opticbin.yaml when available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from config.schema import AppConfig, ModelSpec, WasteMetadata

# ──────────────────────────────────────────────
# YAML Config Loader
# ──────────────────────────────────────────────
_YAML_CONFIG_PATH = Path(__file__).resolve().parent / "opticbin.yaml"


def load_yaml_config() -> dict[str, Any]:
    """Load opticbin.yaml config. Returns empty dict if unavailable."""
    if not _YAML_CONFIG_PATH.exists():
        return {}
    try:
        import yaml  # PyYAML is optional; falls back to hardcoded defaults
        with open(_YAML_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


_CFG = load_yaml_config()

# ──────────────────────────────────────────────
# Device Configuration
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "yolov8_unified": {
        "timm_name": "yolov8_unified",
        "description": "Unified Single-Stage YOLO - simultaneous bounding box detection and 3-bin degradability classification",
        "cam_target": "none",
    },
}


DEFAULT_MODEL = "yolov8_unified"

# ──────────────────────────────────────────────
# Class Labels  (3-class home waste sorting)
# ──────────────────────────────────────────────
CLASS_LABELS = [
    "biodegradable",
    "non-biodegradable",
    "e-waste",
]

NUM_CLASSES = len(CLASS_LABELS)

# ──────────────────────────────────────────────
# Bin Theme Colors
# ──────────────────────────────────────────────
BIN_COLORS = {
    "biodegradable": "#22C55E",       # Vibrant green
    "non-biodegradable": "#3B82F6",   # Electric blue
    "e-waste": "#F97316",             # Warm orange
}

BIN_ICONS = {
    "biodegradable": "🟢",
    "non-biodegradable": "🔵",
    "e-waste": "🟠",
}

# ──────────────────────────────────────────────
# Waste Metadata  (3-bin Degradability Taxonomy)
# ──────────────────────────────────────────────
WASTE_METADATA: dict[str, dict] = {
    "biodegradable": {
        "biodegradable": True,
        "category": "Biodegradable / Organic",
        "recyclable": "Compostable",
        "decomposition": "2 weeks – 6 months",
        "disposal": "Green Bin — Wet / Organic Waste",
        "bin_color": "#22C55E",
        "bin_icon": "🟢",
        "color": "#22C55E",
        "gradient_from": "#16A34A",
        "gradient_to": "#4ADE80",
        "tips": [
            "Place food scraps, fruit peels, and vegetable waste in the wet bin",
            "Paper, cardboard, and garden waste are also biodegradable",
            "Keep separate from plastics and packaging materials",
            "Composting at home reduces landfill methane emissions by up to 50%",
        ],
        "examples": "Food scraps, fruit peels, paper, cardboard, garden waste, tea bags, cotton, wood",
        "environmental_impact": "Composting diverts waste from landfills, prevents methane emissions, and creates nutrient-rich soil.",
    },
    "non-biodegradable": {
        "biodegradable": False,
        "category": "Non-Biodegradable / Dry Waste",
        "recyclable": "Partially — depends on material type",
        "decomposition": "50 – 1,000,000 years",
        "disposal": "Blue Bin — Dry / Recyclable Waste",
        "bin_color": "#3B82F6",
        "bin_icon": "🔵",
        "color": "#3B82F6",
        "gradient_from": "#2563EB",
        "gradient_to": "#60A5FA",
        "tips": [
            "Rinse containers before placing in the dry waste bin",
            "Plastics: check the resin code (1, 2, 5 are widely recyclable)",
            "Glass and metals can be recycled infinitely without quality loss",
            "Plastic bags and films go to store drop-off, NOT curbside bins",
        ],
        "examples": "Plastic bottles, glass jars, metal cans, rubber, ceramics, styrofoam, synthetic fabrics",
        "environmental_impact": "Only ~9% of plastic is recycled globally. Proper sorting dramatically improves recycling rates.",
    },
    "e-waste": {
        "biodegradable": False,
        "category": "Electronic Waste / Hazardous",
        "recyclable": "Specialized recycling only",
        "decomposition": "Up to 1,000,000+ years (contains toxic metals)",
        "disposal": "E-Waste Collection Center — DO NOT put in regular bins",
        "bin_color": "#F97316",
        "bin_icon": "🟠",
        "color": "#F97316",
        "gradient_from": "#EA580C",
        "gradient_to": "#FB923C",
        "tips": [
            "⚠️ NEVER throw batteries, phones, or cables in regular trash bins",
            "Take to authorized e-waste collection centers or manufacturer take-back programs",
            "Remove personal data from devices before recycling",
            "Old chargers, earphones, and cables are all e-waste — don't ignore small items",
        ],
        "examples": "Batteries, phones, chargers, cables, circuit boards, keyboards, old laptops, light bulbs",
        "environmental_impact": "E-waste contains lead, mercury, and cadmium. Improper disposal contaminates groundwater and soil.",
    },
}

# ──────────────────────────────────────────────
# Input / Preprocessing
# ──────────────────────────────────────────────
INPUT_SIZE = (224, 224)           # H × W expected by both backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# XAI (Explainability) Settings
# ──────────────────────────────────────────────
CAM_OPACITY = 0.5                 # Heatmap overlay blending alpha

# ──────────────────────────────────────────────
# Inference Performance
# ──────────────────────────────────────────────
LATENCY_TARGET_MS = 100           # ≤ 100 ms end-to-end budget
MAX_RAM_GB = 2.5                  # Hard ceiling during webcam streaming

# ──────────────────────────────────────────────
# Paths  (overridable via config/opticbin.yaml)
# ──────────────────────────────────────────────
WEIGHTS_DIR     = _CFG.get("paths", {}).get("weights_dir",  "models/weights")
ONNX_EXPORT_DIR = _CFG.get("paths", {}).get("weights_dir",  "models/weights")
RESULTS_DIR     = _CFG.get("paths", {}).get("results_dir",  "results")

# ──────────────────────────────────────────────
# LLM / Active Learning
# ──────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"

# Confidence below this → Gemini Vision cross-check is triggered
ACTIVE_LEARNING_CONFIDENCE_THRESHOLD: float = 0.35

# Directory where uncertain predictions are logged for review & retraining
REVIEW_QUEUE_DIR = "review_queue"


def is_recyclable(label: str) -> bool:
    """True when the material is fully or partially recyclable."""
    rec = WASTE_METADATA.get(label, {}).get("recyclable", False)
    if rec is True:
        return True
    if rec is False:
        return False
    if isinstance(rec, str):
        return rec.lower() not in {"no", "false", "landfill"}
    return False


def get_bin_color(label: str) -> str:
    """Return the theme color for a waste category bin."""
    return BIN_COLORS.get(label, "#6B7280")


def get_bin_icon(label: str) -> str:
    """Return the icon emoji for a waste category bin."""
    return BIN_ICONS.get(label, "♻️")


def get_model_spec_obj(model_type: str) -> ModelSpec:
    """Return a type-safe ModelSpec dataclass instance."""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type '{model_type}'")
    info = SUPPORTED_MODELS[model_type]
    return ModelSpec(
        name=model_type,
        timm_name=info["timm_name"],
        description=info["description"],
        cam_target=info["cam_target"],
    )


def get_waste_metadata_obj(label: str) -> WasteMetadata:
    """Return a type-safe WasteMetadata dataclass instance."""
    if label not in WASTE_METADATA:
        raise ValueError(f"Unknown waste label '{label}'")
    info = WASTE_METADATA[label]
    return WasteMetadata(
        label=label,
        emoji=info.get("emoji", "♻️"),
        biodegradable=info["biodegradable"],
        category=info["category"],
        recyclable=info["recyclable"],
        decomposition=info["decomposition"],
        disposal=info["disposal"],
        disposal_icon=info.get("disposal_icon", "♻️"),
        color=info["color"],
        tips=list(info.get("tips", [])),
        environmental_impact=info.get("environmental_impact", ""),
    )


def get_app_config() -> AppConfig:
    """Return the global AppConfig dataclass object."""
    return AppConfig(
        device=DEVICE,
        input_size=INPUT_SIZE,
        imagenet_mean=IMAGENET_MEAN,
        imagenet_std=IMAGENET_STD,
        cam_opacity=CAM_OPACITY,
        latency_target_ms=LATENCY_TARGET_MS,
        max_ram_gb=MAX_RAM_GB,
        weights_dir=WEIGHTS_DIR,
        onnx_export_dir=ONNX_EXPORT_DIR,
    )
