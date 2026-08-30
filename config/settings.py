"""
OpticBin — Central Configuration
=================================
Class labels, input resolutions, device configs, and runtime parameters.
Supports external YAML config file for hyperparameter tuning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from config.schema import AppConfig, ModelSpec, WasteMetadata

# ──────────────────────────────────────────────
# External Config File Loading
# ──────────────────────────────────────────────
_CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"
_external_config: Dict[str, Any] = {}

if _CONFIG_FILE.exists():
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            _external_config = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        _external_config = {}


def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """Retrieve a value from external config with fallback to default."""
    return _external_config.get(section, {}).get(key, default)


# ──────────────────────────────────────────────
# Device Configuration
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "efficientnetv2_s": {
        "timm_name": "efficientnetv2_rw_s",
        "description": "Texture-focused CNN (EfficientNetV2-RW-S) — strong on surface material features",
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
# Class Labels  (5-class unified waste taxonomy)
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
_input_size = get_config_value("model", "input_size", [224, 224])
INPUT_SIZE = tuple(_input_size) if isinstance(_input_size, list) else (224, 224)
IMAGENET_MEAN = get_config_value("normalization", "mean", [0.485, 0.456, 0.406])
IMAGENET_STD = get_config_value("normalization", "std", [0.229, 0.224, 0.225])

# ──────────────────────────────────────────────
# XAI (Explainability) Settings
# ──────────────────────────────────────────────
CAM_OPACITY = get_config_value("xai", "cam_opacity", 0.5)

# ──────────────────────────────────────────────
# Inference Performance
# ──────────────────────────────────────────────
LATENCY_TARGET_MS = get_config_value("inference", "latency_target_ms", 100)
MAX_RAM_GB = get_config_value("inference", "max_ram_gb", 2.5)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
WEIGHTS_DIR = get_config_value("paths", "weights_dir", "models/weights")
ONNX_EXPORT_DIR = get_config_value("paths", "onnx_export_dir", "models/weights")
DATASET_DIR = get_config_value("paths", "dataset_dir", "dataset")
RESULTS_DIR = get_config_value("paths", "results_dir", "results")

# ──────────────────────────────────────────────
# Training Defaults (from config.yaml)
# ──────────────────────────────────────────────
DEFAULT_EPOCHS = get_config_value("training", "epochs", 15)
DEFAULT_BATCH_SIZE = get_config_value("training", "batch_size", 32)
DEFAULT_LEARNING_RATE = get_config_value("training", "learning_rate", 0.001)
DEFAULT_PATIENCE = get_config_value("training", "patience", 10)
DEFAULT_SEED = get_config_value("training", "seed", 42)
DEFAULT_VAL_RATIO = get_config_value("training", "val_ratio", 0.2)

# ──────────────────────────────────────────────
# Inference Defaults
# ──────────────────────────────────────────────
DEFAULT_FRAMEWORK = get_config_value("inference", "default_framework", "pytorch")


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
