"""
OpticBin — Model Factory
========================
Single source of truth for backbone construction, checkpoint loading,
and Grad-CAM target-layer selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import timm

from config.settings import NUM_CLASSES, SUPPORTED_MODELS, WEIGHTS_DIR


def get_model_spec(model_type: str) -> dict[str, str]:
    """Return the architecture spec for a supported model key."""
    if model_type not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: {supported}")
    return SUPPORTED_MODELS[model_type]


def create_backbone(
    model_type: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Create a timm classification backbone with the OpticBin class head."""
    spec = get_model_spec(model_type)
    return timm.create_model(
        spec["timm_name"],
        pretrained=pretrained,
        num_classes=num_classes,
    )


def get_cam_target_layers(model: nn.Module, model_type: str) -> list[nn.Module]:
    """Resolve Grad-CAM target layers from the architecture spec."""
    spec = get_model_spec(model_type)
    target_name = spec["cam_target"]
    if not hasattr(model, target_name):
        raise AttributeError(
            f"Model '{model_type}' is missing Grad-CAM target layer '{target_name}'."
        )
    return [getattr(model, target_name)]


def infer_model_type_from_path(path: str | Path) -> str | None:
    """Infer a supported model key from a checkpoint filename stem."""
    stem = Path(path).stem.replace("_int8", "")
    if stem in SUPPORTED_MODELS:
        return stem
    for model_type in SUPPORTED_MODELS:
        if stem.startswith(model_type):
            return model_type
    return None


def resolve_weights_path(model_type: str, weights_dir: str = WEIGHTS_DIR) -> Path | None:
    """Return the fine-tuned .pt path if it exists, otherwise None."""
    get_model_spec(model_type)
    candidate = Path(weights_dir) / f"{model_type}.pt"
    return candidate if candidate.exists() else None


def load_checkpoint(
    model: nn.Module,
    weights_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """Load a state_dict or full serialized model into `model`."""
    checkpoint: Any = torch.load(weights_path, map_location=map_location)

    if isinstance(checkpoint, nn.Module):
        return checkpoint

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        return model

    raise TypeError(
        f"Unsupported checkpoint type {type(checkpoint)!r} in '{weights_path}'. "
        "Expected a state_dict or a full nn.Module."
    )


def create_model_from_checkpoint(
    weights_path: str | Path,
    model_type: str | None = None,
    num_classes: int = NUM_CLASSES,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, str]:
    """
    Reconstruct a backbone and load weights.

    Training saves a state_dict, so export and inference must rebuild the
    architecture before loading. Returns `(model, resolved_model_type)`.
    """
    resolved_type = model_type or infer_model_type_from_path(weights_path)
    if resolved_type is None:
        raise ValueError(
            f"Could not infer model type from '{weights_path}'. "
            "Pass model_type explicitly."
        )

    model = create_backbone(resolved_type, num_classes=num_classes, pretrained=False)
    model = load_checkpoint(model, weights_path, map_location=map_location)
    return model, resolved_type
