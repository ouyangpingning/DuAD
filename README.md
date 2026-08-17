# DuAD — Synthetic Anomaly Discrimination with DINOv2

**English** | [简体中文](README_zh-CN.md)

Unsupervised industrial visual anomaly detection using frozen DINOv2 (`dinov2_vits14_reg`) features + dual-branch synthetic anomaly discriminator. Supports both **full-shot** and **few-shot** scenarios.

## Method Overview

![DuAD Method](Method.png)

## Quick Start

### Clone

```bash
git clone --recurse-submodules https://github.com/ouyangpingning/DuAD.git

# If already cloned but missing facebookresearch_dinov2_main/
git submodule init && git submodule update
```

### Install

```bash
# 1. Install PyTorch with CUDA support first (or pip may install CPU-only version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining dependencies
pip install -r requirements.txt
```

> **CUDA compatibility**: Visit https://pytorch.org/get-started/locally/ and choose the matching `--index-url` if your CUDA driver differs. This repo targets CUDA 12.4 (compatible with driver ≥ 525).

Core dependencies: `torch>=2.0`, `scikit-learn`, `opencv-python-headless`, `matplotlib`, `tomli`

> **Server users**: We use `opencv-python-headless` (no GUI dependencies). If you need `cv2.imshow`, replace with `opencv-python` and install system libs: `apt install libgl1-mesa-glx`.

### Prepare Data

Download [MVTec AD](https://www.mvtefactory.com/annotated-dataset) or [VisA](https://amazon-visual-anomaly.s3.amazonaws.com/VisA.tar.gz), then configure paths in `config.toml`:

```toml
[paths]
mvtec_base_dir = "/path/to/mvtec_anomaly_detection"
visa_base_dir = "/path/to/VisA"
```

### Train

```bash
# Full-shot
python src/main.py --categories "bottle screw"

# Few-shot (K=4)
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0

# VisA dataset
python src/main.py --categories "candle" --dataset visa

# Ablation experiments (interactive, supports MVTec AD & VisA)
bash train_ablation.sh

# Interactive batch training (tmux-based, auto GPU partitioning)
bash train_all_tmux.sh
```

### Visualize

```bash
# Random test images with heatmap overlay
python src/viz/visualize_feature.py --categories "bottle screw"

# More samples / analysis-only mode (no checkpoint needed)
python src/viz/visualize_feature.py --categories "bottle" --num_samples 8
python src/viz/visualize_feature.py --categories "bottle" --skip_inference

# Interactive batch visualization (tmux)
bash visualize_all_tmux.sh
```

### Export ONNX

```bash
python src/deploy/export_onnx.py --category bottle
python src/deploy/export_onnx.py --category bottle --verify

# Batch export multiple categories
python src/deploy/export_onnx.py --category "bottle screw" --k_shot 4 --shot_seed 0

# Interactive batch export
bash export_onnx_all_tmux.sh
```

### Client Software: DuAD Software

An industrial anomaly detection **client** built for DuAD. It loads the exported ONNX models directly and reads the deployment thresholds embedded in the model metadata (`duad.*`) — **no manual calibration required** — covering image-level judgment, pixel-level anomaly localization, and fixed heatmap display scale.

**[DuAD Software](https://github.com/ouyangpingning/DuAD_software)** — `PySide6 + QML`:

- **Real-time detection**: industrial camera (Daheng USB3 Vision) frames → ONNX inference → anomaly score / heatmap / pixel localization mask
- **Camera control**: camera search/connect, resolution / exposure / gain / frame rate
- **Light control**: CH340 serial light source controller (4 channels)
- **MQTT alarms**: cloud broker, TLS, alarm reporting
- **Scheduled image collection**, ROI, fullscreen display

### Aggregate Results

```bash
bash aggregate_results.sh
python src/analysis/aggregate_results.py
python src/analysis/aggregate_results.py --csv
```

## Project Structure

```
.
├── config.toml                      # Single source of truth for all hyperparameters
├── requirements.txt
├── train_all_tmux.sh                # Interactive batch training (tmux)
├── train_ablation.sh                # Interactive ablation experiments
├── visualize_all_tmux.sh            # Interactive batch visualization (tmux)
├── export_onnx_all_tmux.sh          # Interactive ONNX export
├── aggregate_results.sh             # Interactive result aggregation
├── src/
│   ├── main.py                      # Training entry point
│   ├── DuAD.py                      # Core model (Trainer, Predictor, etc.)
│   ├── config.py                    # TOML → ModelConfig parser
│   ├── utils.py                     # Metrics, logging, DINOv2 loader
│   ├── commen_import.py             # Shared third-party imports
│   ├── dataset/                     # Dataset layer (Facade pattern)
│   │   ├── __init__.py              #   Unified API: get_dataloader()
│   │   ├── mvtec.py                 #   MVTec AD dataset
│   │   └── visa.py                  #   VisA dataset
│   ├── viz/
│   │   └── visualize_feature.py     # Visualization entry point
│   ├── deploy/
│   │   └── export_onnx.py           # ONNX model export
│   └── analysis/
│       └── aggregate_results.py     # Log aggregation & statistics
├── simplenet/                       # SimpleNet baseline (independent project)
├── facebookresearch_dinov2_main/    # Git submodule: DINOv2 source
├── model_ckpt/                      # Model checkpoints
├── model_log/                       # Training logs
├── model_onnx/                      # ONNX models
├── outputs/                         # Visualization outputs
└── results/                         # Aggregated CSVs
```

## License

Apache License 2.0
