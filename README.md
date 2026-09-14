# OpticBin

Real-Time Edge-AI Waste Sorting Assistant

OpticBin is an intelligent, high-performance waste classification and sorting system designed for real-time edge deployment. Powered by a fine-tuned Unified Single-Stage YOLOv8 architecture and accelerated by NVIDIA CUDA hardware, OpticBin simultaneously localizes waste items and classifies them into standardized material categories in under 12 milliseconds.

For ambiguous edge cases, OpticBin integrates an Active Learning feedback loop backed by a multimodal Gemini Vision supervisor that validates low-confidence predictions, auto-corrects sorting recommendations, and logs edge cases for continuous fine-tuning.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Supported Material Taxonomy](#supported-material-taxonomy)
- [Performance Specifications](#performance-specifications)
- [Repository Structure](#repository-structure)
- [Hardware and Software Requirements](#hardware-and-software-requirements)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Model Training and Retraining](#model-training-and-retraining)
- [Active Learning Loop](#active-learning-loop)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Key Capabilities

- **Unified Single-Stage Detection**: Bounding-box localization and material classification occur simultaneously in a single forward pass, eliminating multi-stage model handoffs and latency bottlenecks.
- **Hardware-Accelerated Inference**: Leverages NVIDIA Tensor Cores with CUDA 12.6 and Automatic Mixed Precision (AMP), achieving sub-12 ms end-to-end inference latency on modern GPUs.
- **Active Learning Supervisor**: Automatically queries multimodal Gemini Vision when local detection confidence is low, providing real-time second opinions, dynamic auto-correction, and automatic edge-case logging.
- **Actionable Disposal Guidance**: Every scan immediately presents color-coded Hero Destination Bin cards, material confidence metrics, and step-by-step preparation checklists (such as rinsing containers and resin code verification).
- **Interactive Conversational Advisor**: Includes an integrated streaming recycling chatbot that answers handling questions, local municipality sorting rules, and environmental impact inquiries.
- **Dual Input Modalities**: Supports both high-resolution image uploads and real-time live webcam capture viewfinders.

---

## System Architecture

```
+---------------------------------------------------------------------------------+
|                                 OpticBin System                                 |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  [ Visual Input Source ]                                                        |
|   - Live Webcam Snapshot or High-Resolution File Upload (JPG/PNG)               |
|                               │                                                 |
|                               ▼                                                 |
|  [ Unified YOLOv8s Inference Engine ]                                           |
|   - Architecture: YOLOv8 Small (11.2M parameters)                               |
|   - Compute: NVIDIA GeForce RTX GPU (CUDA 12.6, FP16 AMP)                       |
|   - Execution Latency: Sub-12 ms                                                |
|   - Output: Coordinates, Material Class, Confidence                             |
|                               │                                                 |
|                               ▼                                                 |
|  [ Confidence Arbitration & Active Learning ]                                   |
|   - If Confidence >= 35%: Instant Local Resolution (< 12 ms)                    |
|   - If Confidence < 35%: Query Multimodal Gemini Vision Supervisor              |
|        - High-Confidence Supervisor Response: Dynamic UI Auto-Correction        |
|        - Log Flagged Image & Metadata to review_queue/ for Retraining           |
|                               │                                                 |
|                               ▼                                                 |
|  [ Interactive Streamlit Dashboard ]                                            |
|   - Visual Capture with Labeled Bounding Box Overlay                            |
|   - Color-Coded Hero Disposal Bin Card                                          |
|   - Actionable Preparation Checklist & Environmental Decomposition Time         |
|   - Material Probability Distribution Chart                                     |
|   - Conversational AI Recycling Advisor                                         |
+---------------------------------------------------------------------------------+
```

---

## Supported Material Taxonomy

| Material | Category | Recyclability | Disposal Destination | Key Handling Rule |
|---|---|---|---|---|
| Biodegradable | Organic | Fully Compostable | Compost / Organic Bin | Separate from plastics; remove packaging |
| Cardboard | Recyclable | Widely Recyclable | Recycling Bin (Cardboard) | Flatten boxes to save space; keep dry |
| Glass | Non-Biodegradable | Infinitely Recyclable | Recycling Bin (Glass) | Empty and rinse; separate broken glass if required |
| Metal | Non-Biodegradable | Infinitely Recyclable | Recycling Bin (Metal) | Rinse cans; aluminum and steel are fully recyclable |
| Paper | Recyclable | Widely Recyclable | Recycling Bin (Paper) | Keep clean and dry; remove plastic wrapping |
| Plastic | Non-Biodegradable | Varies by Type (1, 2, 5) | Recycling Bin (Plastic) | Check resin code on base; rinse residue thoroughly |

---

## Performance Specifications

| Metric | Target Specification | Achieved Performance (RTX 4060 GPU) |
|---|---|---|
| Inference Latency | Sub-50 ms | 8 to 12 ms |
| Preprocessing & Overlay | Sub-20 ms | 6 to 10 ms |
| System RAM Consumption | Under 2.5 GB | Approximately 1.2 GB |
| Detection mAP50 | Greater than 50% | 53.1% (50 Epoch Fine-Tuned Model) |
| Local Model Autonomy | Greater than 85% | Confident classifications resolve 100% locally |

---

## Repository Structure

```
OpticBin/
├── app.py                           # Main Streamlit application entry point
├── train_yolo.py                    # CUDA-accelerated YOLOv8 fine-tuning pipeline
├── add_review_sample_to_dataset.py  # Active learning feedback merger
├── retrain_from_queue.py            # Review queue batch processing utility
├── prepare_yolo_dataset.py          # Dataset structure validator and sample generator
├── requirements.txt                 # Project runtime dependencies
├── config/
│   ├── settings.py                  # Core configuration, thresholds, and metadata
│   ├── schema.py                    # Type-safe configuration dataclasses
│   └── waste_yolo.yaml              # YOLO dataset paths and class configuration
├── models/
│   ├── export_onnx.py               # ONNX runtime model export utility
│   └── weights/
│       ├── yolov8_waste.pt          # Fine-tuned YOLOv8s GPU weights
│       └── yolov8_waste.onnx        # Exported high-speed ONNX model
├── src/
│   ├── yolo_unified_engine.py       # Single-stage YOLOv8 inference engine
│   ├── inference_engine.py          # Abstract engine interface and PredictionResult
│   ├── active_learner.py            # Multimodal Gemini Vision supervisor
│   ├── llm_advisor.py               # Streaming recycling advisory assistant
│   ├── camera.py                    # Threaded webcam capture handler
│   ├── preprocessor.py              # Image normalization and PIL transforms
│   └── yolo_cropper.py              # Visual bounding-box overlay utilities
├── ui/
│   ├── components.py                # Hero cards, checklists, and active learning badges
│   ├── image_view.py                # Upload image analysis view
│   ├── webcam_view.py               # Live camera viewfinder view
│   ├── styles.py                    # Dark-mode theme, CSS variables, and cards
│   └── state_manager.py             # Session statistics tracker
├── dataset_roboflow/                # YOLO annotated dataset (7,340 images)
└── review_queue/                    # Logged edge-case scans for continuous learning
```

---

## Hardware and Software Requirements

- **Operating System**: Windows 10/11 or Ubuntu 20.04/22.04 LTS
- **Python**: Version 3.10 to 3.13
- **GPU (Recommended for Real-Time Speed)**: NVIDIA GeForce RTX GPU with CUDA 12.x support (e.g., RTX 3060, RTX 4060 or higher with 8+ GB VRAM)
- **CPU**: Multi-core processor (Intel Core i5/i7/i9 or AMD Ryzen 5/7/9)
- **RAM**: Minimum 8 GB (16 GB recommended)

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sreejithpa21112006/OpticBin.git
cd OpticBin
```

### 2. Create and Activate a Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA Support

For NVIDIA GPU acceleration (CUDA 12.6):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only environments:
```bash
pip install torch torchvision
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Optional API Key

To enable the interactive AI Recycling Advisor and Active Learning cross-checks:
- Obtain a free API key from Google AI Studio (aistudio.google.com).
- Provide the key via Streamlit secrets (`.streamlit/secrets.toml`):

```toml
GEMINI_API_KEY = "AIzaSy..."
```

Alternatively, set the environment variable:
```bash
export GEMINI_API_KEY="AIzaSy..."     # Linux/macOS
$env:GEMINI_API_KEY="AIzaSy..."       # Windows PowerShell
```

The API key can also be entered interactively in the application sidebar at runtime.

---

## Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Once started, open your web browser to:
```
http://localhost:8501
```

### Application Features:
- **Scan Mode Selection**: Switch between **Live Camera Viewfinder** (webcam snapshot or stream) and **Upload Waste Image** (JPG, PNG, WebP).
- **Hero Recommendation Card**: Displays the designated recycling or disposal destination with material-specific color themes.
- **Disposal Instructions**: Step-by-step preparation tips and estimated decomposition timelines.
- **Attention Heatmap & Bounding Box**: Highlights the detected item with class name and confidence score.
- **AI Recycling Assistant**: Expandable sidebar assistant to answer specific recycling questions.

---

## Model Training and Retraining

### Train YOLOv8 on GPU

The training pipeline uses Automatic Mixed Precision (AMP), cosine learning rate decay, and data augmentations (HSV saturation/value variation and mosaic transforms):

```bash
python train_yolo.py --model yolov8s.pt --epochs 50 --batch_size 16 --device 0
```

### Command Arguments:
- `--model`: Base checkpoint (`yolov8s.pt` for 11.2M parameters or `yolov8n.pt` for 3.2M parameters).
- `--epochs`: Number of training iterations (default: 50).
- `--batch_size`: Batch size per step (default: 16).
- `--device`: Target device (`0` for primary NVIDIA GPU, `cpu` for CPU fallback).
- `--imgsz`: Input resolution (default: 640).

The best checkpoint is automatically saved to `models/weights/yolov8_waste.pt` and exported to ONNX format.

---

## Active Learning Loop

OpticBin incorporates a continuous improvement pipeline:

1. **Uncertainty Flagging**: When local detection confidence falls below the calibrated threshold (`0.35`), the active learning pipeline triggers a supervisor review.
2. **Multimodal Supervisor Cross-Check**: Gemini Vision independently inspects the high-resolution image and returns a detailed assessment with class and confidence scores.
3. **Dynamic UI Auto-Correction**: If the supervisor returns a high-confidence determination (>= 70%), the recommendation card, checklist, and bounding box dynamically update to reflect the verified classification.
4. **Queue Logging**: The image, prediction metadata, and supervisor reasoning are logged to `review_queue/`.
5. **Continuous Retraining**: Incorporate all verified review queue samples into the training dataset with a single command:

```bash
python add_review_sample_to_dataset.py
```

Once merged, rerun `train_yolo.py` to retrain the local model with the new real-world edge cases.

---

## Configuration Reference

Key application parameters can be tuned in `config/settings.py`:

| Parameter | Default Value | Description |
|---|---|---|
| `ACTIVE_LEARNING_CONFIDENCE_THRESHOLD` | `0.35` | Predictions below this threshold trigger the supervisor cross-check |
| `DEFAULT_MODEL` | `"yolov8_unified"` | Active inference engine architecture |
| `LLM_MODEL` | `"gemini-2.5-flash"` | Gemini model used for supervisor checks and conversational advice |
| `LATENCY_TARGET_MS` | `100` | Target end-to-end latency budget |
| `MAX_RAM_GB` | `2.5` | Memory ceiling during continuous webcam streaming |

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
