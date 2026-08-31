# -*- coding: utf-8 -*-
"""
OpticBin — Parameter Count Inspector
Shows total params, and EXACTLY which layers are trained vs frozen
during a CPU training run (mirrors the logic in src/trainer.py).

Run: python check_params.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import timm
import torch
from config.settings import NUM_CLASSES

# CPU freeze logic copied from src/trainer.py
CPU_TRAINABLE_KEYWORDS = [
    "classifier", "head", "conv_head", "final_conv",
    "blocks.5", "blocks.6", "stages.3"
]

SEP = "=" * 60

models = [
    ("efficientnetv2_s (Medium variant)", "efficientnetv2_rw_m"),
    ("mobilevit_xs",                      "mobilevit_xs"),
]

for label, timm_name in models:
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)

    m = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)

    # Apply CPU freeze logic (same as trainer.py does at runtime)
    for name, param in m.named_parameters():
        if not any(k in name for k in CPU_TRAINABLE_KEYWORDS):
            param.requires_grad = False

    total      = sum(p.numel() for p in m.parameters())
    trainable  = sum(p.numel() for p in m.parameters() if p.requires_grad)
    frozen     = total - trainable

    print(f"  Total parameters     : {total:>12,}")
    print(f"  Frozen  (not updated): {frozen:>12,}  <- early backbone layers")
    print(f"  Trainable (updated)  : {trainable:>12,}  <- these are fine-tuned on your dataset")
    print(f"  Trainable %          : {trainable/total*100:>11.2f}%")

    print(f"\n  Layers being trained:")
    seen = set()
    for name, param in m.named_parameters():
        if param.requires_grad:
            group = name.split(".")[0] if "blocks" not in name else ".".join(name.split(".")[:2])
            if group not in seen:
                seen.add(group)
                p = sum(p2.numel() for n2, p2 in m.named_parameters()
                        if p2.requires_grad and n2.startswith(group.split(".")[0]))
                print(f"    {group:<30} ~{p:,} params")

print(f"\n{SEP}")
print("  NOTE: On GPU all layers are trainable (no freezing applied).")
print(SEP)
