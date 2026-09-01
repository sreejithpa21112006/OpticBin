<p align="center">
  <img src="assets/banner.png" alt="OpticBin Banner" width="100%"/>
</p>

# OpticBin

**Edge-AI Waste Classification System with Real-Time Explainable AI**

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [System Architecture Diagram](#system-architecture-diagram)
- [Core Specifications](#core-specifications)
- [Supported Waste Taxonomy](#supported-waste-taxonomy)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dataset Preparation](#dataset-preparation)
- [Model Training and Seeded Reproducibility](#model-training-and-seeded-reproducibility)
- [Model Evaluation and Metrics Tracking](#model-evaluation-and-metrics-tracking)
- [Model Parameter Comparison](#model-parameter-comparison)
- [ONNX Export and Quantization](#onnx-export-and-quantization)
- [Benchmarking and Testing](#benchmarking-and-testing)
- [Explainable AI (XAI) Engine](#explainable-ai-xai-engine)
- [Dashboard Usage](#dashboard-usage)
- [Acceptance Verification](#acceptance-verification)
- [License](#license)

---

## Executive Summary

Industrial waste sorting requires real-time, offline automated classification across diverse material types including electronic waste, organic matter, glass, paper, cardboard, plastic, metal, and general trash. Cloud-based inference models introduce latency, bandwidth costs, and network reliability bottlenecks in industrial edge environments. Furthermore, regulatory auditing requires Explainable AI (XAI) feature maps to verify model reasoning.

OpticBin is a fully offline Edge-AI pipeline that classifies 8 categories of waste using fine-tuned backbones (EfficientNetV2-S and MobileViT-XS). It offers dual inference execution paths:
1. **PyTorch Engine:** Full inference with visual Grad-CAM heatmap generation.
2. **ONNX Runtime Engine:** INT8-quantized execution provider optimized for sub-100ms CPU latency.

---

## System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                 OpticBin System                                   |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ Data Ingestion ]                                                               |
|   - TrashNet & Kaggle 8-Class Dataset                                             |
|   - Scripts: download_dataset.py, download_new_classes.py                         |
|                               │                                                   |
|                               ▼                                                   |
|  [ Preprocessing & Data Pipeline ]                                                |
|   - 224x224 Bilinear Resize & ImageNet Normalization                              |
|   - Module: src/preprocessor.py                                                   |
|                               │                                                   |
|                               ▼                                                   |
|  [ Model Backbones & Config ]                                                     |
|   - Config: config/opticbin.yaml & config/settings.py (Seed: 42)                  |
|   - Backbones: EfficientNetV2-S (51.1M params) & MobileViT-XS (1.94M params)      |
|   - Factory: src/model_factory.py                                                 |
|                               │                                                   |
|                               ├──────────────────────────┐                        |
|                               ▼                          ▼                        |
|                  [ PyTorch Training Engine ]    [ Evaluation Pipeline ]           |
|                   - train.py & src/trainer.py    - evaluate.py                    |
|                   - AdamW + Cosine LR            - Top-1 Acc, F1, Latency         |
|                   - Exports: models/weights/*.pt - Reports: results/*_eval.json   |
|                               │                                                   |
|                               ▼                                                   |
|                  [ ONNX Export & INT8 Quant ]                                     |
|                   - models/export_onnx.py                                         |
|                   - Converts .pt -> .onnx -> _int8.onnx                           |
|                               │                                                   |
|                               ├──────────────────────────┐                        |
|                               ▼                          ▼                        |
|                  [ XAI Grad-CAM Engine ]        [ Production ONNX Engine ]        |
|                   - src/xai_engine.py            - src/inference_engine.py        |
|                   - Heatmap Overlay Generation   - Sub-100ms Classify Path        |
|                               │                          │                        |
|                               └────────────┬─────────────┘                        |
|                                            ▼                                      |
|                                 [ Streamlit UI Dashboard ]                        |
|                                  - app.py & ui/                                   |
|                                  - Upload & Live Webcam Modes                     |
+-----------------------------------------------------------------------------------+
```

---

## Core Specifications

| Attribute | Specification |
|---|---|
| Latency Target | Sub-100ms per frame (ONNX INT8 execution) |
| System RAM Limit | Less than or equal to 2.5 GB during live stream mode |
| Supported Classes | 5 classes (Cardboard, Glass, Metal, Paper, Plastic) |
| Execution Mode | 100% Offline (Local CPU / CUDA support) |
| Compression | 4x size reduction via INT8 Post-Training Quantization |
| Explainability | Grad-CAM heatmap overlays on PyTorch execution path |

---

## Supported Waste Taxonomy

| Class Name | Target Materials and Items |
|---|---|
| Cardboard | Corrugated shipping boxes, paperboard packaging |
| Glass | Glass bottles, jars, window fragments |
| Metal | Aluminum cans, tin foil, metallic hardware |
| Paper | Office paper, newsprint, magazines |
| Plastic | PET bottles, HDPE containers, plastic packaging |


---

## Tech Stack

| Layer | Technology | Architectural Purpose |
|---|---|---|
| Deep Learning Framework | PyTorch 2.0+ & timm | Backbone feature extraction, fine-tuning, and gradient hooking |
| Runtime Inference | ONNX Runtime (INT8) | Accelerated CPU/GPU runtime with INT8 quantized execution |
| Explainability (XAI) | pytorch-grad-cam | Grad-CAM activation mapping on Conv and Transformer stages |
| Computer Vision | OpenCV & torchvision | Threaded webcam frames, tensor transformations, and normalization |
| Configuration | PyYAML & dataclass schema | Centralized YAML config parsing and schema validation |
| Dashboard UI | Streamlit 1.28+ | Web-based interface for image uploads and live webcam stream |

---

## Repository Structure

```
OpticBin/
├── config/
│   ├── opticbin.yaml          # External YAML configuration (seed, LR, epochs, paths)
│   ├── schema.py              # Dataclass validation schemas
│   └── settings.py            # Central settings parser and default fallback bindings
├── dataset/                   # Local dataset directory (8 waste class subfolders)
├── models/
│   ├── benchmark.py           # Latency and memory benchmarking tool
│   ├── export_onnx.py         # PyTorch to INT8 ONNX converter
│   └── weights/               # Saved model checkpoints (.pt, .onnx, _int8.onnx, metrics.json)
├── results/                   # Evaluation artifacts (JSON metrics and text summaries)
├── src/
│   ├── camera.py              # Threaded OpenCV webcam frame buffer
│   ├── inference_engine.py    # Unified PyTorch and ONNX Runtime inference wrapper
│   ├── model_factory.py       # Architecture factory, layer target resolver, weights loader
│   ├── preprocessor.py        # Image preprocessor for BGR, PIL, and PyTorch inputs
│   ├── trainer.py             # Model fine-tuning and validation engine
│   ├── xai_engine.py          # Grad-CAM heatmap generator
│   └── xai_renderer.py        # Heatmap blending and color map rendering
├── tests/
│   ├── test_config.py         # Unit tests for configuration schema validation
│   ├── test_model_factory.py  # Unit tests for architecture instantiation
│   ├── test_optic.py          # End-to-end integration tests
│   └── test_preprocessor.py   # Unit tests for preprocessing transformations
├── ui/
│   ├── components.py          # Dashboard UI components, sidebar controls, metrics cards
│   ├── image_view.py          # Single image upload and evaluation view
│   ├── state_manager.py       # Streamlit session state management
│   ├── styles.py              # Custom CSS layout styling
│   └── webcam_view.py         # Live webcam classification view
├── app.py                     # Streamlit application entrypoint
├── check_params.py            # Model parameter inspector tool
├── download_dataset.py        # Dataset downloader and aggregator script
├── download_new_classes.py    # E-waste and organic dataset downloader
├── evaluate.py                # Model evaluation and metrics generation CLI
├── fix_dataset.py             # Dataset verification and repair tool
├── requirements.txt           # Production Python dependencies
├── requirements-dev.txt       # Development and testing dependencies
├── train.py                   # Model training and fine-tuning CLI
├── .gitignore
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 to 3.13
- CUDA 11.8+ (optional, CPU execution fully supported)
- Webcam (optional, for live webcam classification mode)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/sreejithpa21112006/OpticBin.git
cd OpticBin

# 2. Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

To download and structure the 8-class dataset automatically:

```bash
python download_dataset.py
python download_new_classes.py
```

This populates the `dataset/` directory with class subfolders: `cardboard`, `e_waste`, `glass`, `metal`, `organic`, `paper`, `plastic`, and `trash`.

---

## Model Training and Seeded Reproducibility

Train models using CLI flags or default settings from `config/opticbin.yaml`:

```bash
# Train EfficientNetV2-S with global seed
python train.py --model efficientnetv2_s --epochs 15 --batch_size 32 --seed 42

# Train MobileViT-XS
python train.py --model mobilevit_xs --epochs 15 --batch_size 32 --seed 42

# Train all backbones sequentially
python train.py --model all --epochs 15 --seed 42
```

Reproducibility is guaranteed by `set_global_seed()`, which explicitly sets random seeds across Python `random`, `numpy`, PyTorch CPU, and PyTorch CUDA backends.

---

## Model Evaluation and Metrics Tracking

Run evaluation on a stratified held-out test split:

```bash
# Evaluate EfficientNetV2-S
python evaluate.py --model efficientnetv2_s --data_dir dataset --seed 42

# Evaluate MobileViT-XS
python evaluate.py --model mobilevit_xs --data_dir dataset --seed 42

# Evaluate all models and generate combined report
python evaluate.py --model all --data_dir dataset --seed 42
```

Evaluation outputs are persisted in the `results/` directory:
- `results/efficientnetv2_s_eval.json`: Detailed JSON containing Top-1 accuracy, per-class F1, macro F1, latency, and confusion matrix.
- `results/efficientnetv2_s_eval_summary.txt`: Plain-text evaluation report.

---

## Model Parameter Comparison

OpticBin supports two backbone options evaluated with `check_params.py`:

| Backbone Model | Total Parameters | Trainable Parameters (GPU) | Trainable Parameters (CPU) | Primary Strengths |
|---|---|---|---|---|
| **EfficientNetV2-S** | 51,100,666 (~51.1M) | 51,100,666 (100%) | 39,216,694 (76.7%) | Maximum surface texture accuracy |
| **MobileViT-XS** | 1,935,928 (~1.94M) | 1,935,928 (100%) | 739,864 (38.2%) | Ultra-lightweight edge deployment |

---

## ONNX Export and Quantization

Export PyTorch weights (`.pt`) to optimized ONNX format with Post-Training Quantization (PTQ):

```bash
python models/export_onnx.py \
    --pt_path models/weights/efficientnetv2_s.pt \
    --onnx_out models/weights/efficientnetv2_s.onnx \
    --quant_out models/weights/efficientnetv2_s_int8.onnx \
    --model efficientnetv2_s
```

Quantization results in ~4x footprint reduction:
- `efficientnetv2_s.pt` (206 MB) -> `efficientnetv2_s_int8.onnx` (52 MB)
- `mobilevit_xs.pt` (7.9 MB) -> `mobilevit_xs_int8.onnx` (2.4 MB)

---

## Benchmarking and Testing

Run unit tests and latency benchmarks:

```bash
# Execute unit test suite
python -m unittest discover tests

# Benchmark inference latency and memory
python models/benchmark.py --iterations 100
```

---

## Explainable AI (XAI) Engine

Grad-CAM target layers are hooked dynamically:
- **EfficientNetV2-S Target Layer:** `conv_head`
- **MobileViT-XS Target Layer:** `final_conv`

The XAI engine overlays class-activation heatmaps onto original input frames, allowing visual audit of object regions driving classification decisions.

---

## Dashboard Usage

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

### Modes of Operation
1. **Single Image Upload Mode:** Upload JPG, PNG, or WebP images to visualize classification predictions, confidence scores, and Grad-CAM heatmaps.
2. **Live Webcam Stream Mode:** Real-time webcam inference with dual-column view (live video feed alongside live heatmap updates).

---

## Acceptance Verification

| ID | Scenario | Expected Outcome | Verification Method |
|---|---|---|---|
| AC-1 | Held-out test evaluation | Accuracy and per-class metrics reported in results directory | `python evaluate.py --model all` |
| AC-2 | Model switching in UI | Hot-swapping backbones in Streamlit sidebar without app restart | App UI sidebar selection |
| AC-3 | Offline execution | Operational without active internet connection | Disconnect network and run `app.py` |
| AC-4 | INT8 Model Quantization | Size reduction and INT8 ONNX export | `python models/export_onnx.py` |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
