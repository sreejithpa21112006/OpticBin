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
    "efficientnetv2_s": {
        "timm_name": "efficientnetv2_rw_m",   # saved .pt is the Medium variant (2152-ch head)
        "description": "Texture-focused CNN (EfficientNetV2-RW-M) — strong on surface material features",
        "cam_target": "conv_head",
    },
    "mobilevit_xs": {
        "timm_name": "mobilevit_xs",
        "description": "Global spatial ViT — captures shape & structural context",
        "cam_target": "final_conv",
    },
}

DEFAULT_MODEL = "efficientnetv2_s"

# ──────────────────────────────────────────────
# Class Labels  (5-class core waste taxonomy)
# ──────────────────────────────────────────────
CLASS_LABELS = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

NUM_CLASSES = len(CLASS_LABELS)

# ──────────────────────────────────────────────
# Waste Metadata  (Biodegradability & Disposal)
# ──────────────────────────────────────────────
WASTE_METADATA: dict[str, dict] = {
    "glass": {
        "biodegradable": False,
        "category": "Non-Biodegradable",
        "recyclable": True,
        "decomposition": "~1 million years",
        "disposal": "Recycling Bin (Glass)",
        "color": "#3B82F6",
        "tips": [
            "Rinse before recycling — remove caps and lids",
            "Do NOT mix with ceramics or mirrors (different melting points)",
            "Glass can be recycled infinitely without quality loss",
        ],
        "environmental_impact": "High — glass in landfills never decomposes. Recycling saves 30% energy vs. new production.",
    },
    "paper": {
        "biodegradable": True,
        "category": "Biodegradable",
        "recyclable": True,
        "decomposition": "2–6 weeks",
        "disposal": "Recycling Bin (Paper)",
        "color": "#10B981",
        "tips": [
            "Keep dry — wet/soiled paper goes to compost, not recycling",
            "Remove staples, tape, and plastic windows from envelopes",
            "Paper can be recycled 5–7 times before fibers degrade",
        ],
        "environmental_impact": "Low — decomposes naturally. Recycling 1 ton of paper saves 17 trees.",
    },
    "cardboard": {
        "biodegradable": True,
        "category": "Biodegradable",
        "recyclable": True,
        "decomposition": "2–3 months",
        "disposal": "Recycling Bin (Cardboard)",
        "color": "#10B981",
        "tips": [
            "Flatten boxes to save space in recycling bins",
            "Remove packing tape, styrofoam, and bubble wrap first",
            "Pizza boxes with heavy grease stains should be composted, not recycled",
        ],
        "environmental_impact": "Low — naturally biodegradable. Recycling reduces deforestation and water pollution.",
    },
    "plastic": {
        "biodegradable": False,
        "category": "Non-Biodegradable",
        "recyclable": "Partially (Types 1, 2, 5)",
        "decomposition": "450–1000 years",
        "disposal": "Recycling Bin (Plastic) — check type number",
        "color": "#EF4444",
        "tips": [
            "Check the resin code (1-7) on the bottom — only 1, 2, and 5 are widely recyclable",
            "Rinse containers and remove labels when possible",
            "Plastic bags and films go to store drop-off, NOT curbside recycling",
            "Avoid single-use plastics — they often end up in oceans",
        ],
        "environmental_impact": "Very High — microplastics contaminate water, soil, and food chains. Only ~9% of plastic is ever recycled.",
    },
    "metal": {
        "biodegradable": False,
        "category": "Non-Biodegradable",
        "recyclable": True,
        "decomposition": "50–500 years (aluminum: 200 yrs, steel: 50 yrs)",
        "disposal": "Recycling Bin (Metal)",
        "color": "#8B5CF6",
        "tips": [
            "Rinse cans — labels can stay on (they burn off during recycling)",
            "Aluminum cans are the most valuable recyclable material",
            "Recycling 1 aluminum can saves enough energy to run a TV for 3 hours",
        ],
        "environmental_impact": "Medium — mining is destructive but metals are infinitely recyclable with no quality loss.",
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
