"""
OpticBin — Configuration Schemas & Data Structures
=================================================
Type-safe dataclasses and validation models for project configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a supported neural backbone architecture."""
    name: str
    timm_name: str
    description: str
    cam_target: str


@dataclass(frozen=True)
class WasteMetadata:
    """Disposal guidance and environmental properties for a waste class."""
    label: str
    emoji: str
    biodegradable: bool
    category: str
    recyclable: str | bool
    decomposition: str
    disposal: str
    disposal_icon: str
    color: str
    tips: List[str] = field(default_factory=list)
    environmental_impact: str = ""


@dataclass(frozen=True)
class AppConfig:
    """Global runtime configuration parameters."""
    device: str
    input_size: tuple[int, int]
    imagenet_mean: list[float]
    imagenet_std: list[float]
    cam_opacity: float
    latency_target_ms: float
    max_ram_gb: float
    weights_dir: str
    onnx_export_dir: str
