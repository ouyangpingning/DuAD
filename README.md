# DuAD — Dual-Branch Unsupervised Anomaly Detection with DINOv2

**English** | [简体中文](README_zh-CN.md)

Unsupervised industrial visual anomaly detection using frozen DINOv2 (`dinov2_vits14_reg`) features + dual-branch GAN discriminator. Supports both **full-shot** and **few-shot** scenarios.

## Quick Start

### Clone

```bash
git clone --recurse-submodules git@github.com:ouyangpingning/DuAD.git

# If already cloned but missing facebookresearch_dinov2_main/
git submodule init && git submodule update
```

### Install

```bash
pip install -r requirements.txt
```

Core dependencies: `torch>=2.0`, `scikit-learn`, `opencv-python`, `matplotlib`, `tomli`

### Prepare Data

Download [MVTec AD](https://www.mvtefactory.com/annotated-dataset) or [VisA](https://amazon-visual-anomaly.s3.amazonaws.com/VisA.tar.gz), then configure `config.toml` `[paths]`:

```toml
[paths]
mvtec_base_dir = "/path/to/mvtec_anomaly_detection"
visa_base_dir = "/path/to/VisA"
```

### Train

```bash
python src/main.py --categories "bottle screw"
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0
python src/main.py --categories "candle" --dataset visa

# Interactive batch training (tmux)
bash train_all_tmux.sh
```

### Visualize

```bash
python src/visualize_feature.py --categories "bottle screw"
python src/visualize_feature.py --categories "bottle" --num_samples 8
python src/visualize_feature.py --categories "bottle" --skip_inference

# Interactive batch visualization (tmux)
bash visualize_all_tmux.sh
```

### Export ONNX

```bash
python src/export_onnx.py --category bottle
python src/export_onnx.py --category bottle --pca_mode student --verify

bash export_onnx_all_tmux.sh
```

### Aggregate Results

```bash
bash aggregate_results.sh
python src/aggregate_results.py
python src/aggregate_results.py --csv
```

## Project Structure

```
.
├── config.toml                      # Single source of truth for all hyperparameters
├── requirements.txt
├── train_all_tmux.sh                # Interactive batch training (tmux)
├── visualize_all_tmux.sh            # Interactive batch visualization (tmux)
├── export_onnx_all_tmux.sh          # Interactive ONNX export
├── aggregate_results.sh             # Interactive result aggregation
├── .gitmodules                      # Git submodule config
├── src/
│   ├── main.py                      # Training entry point
│   ├── visualize_feature.py         # Visualization entry point
│   ├── export_onnx.py               # ONNX model export
│   ├── aggregate_results.py         # Log aggregation & statistics
│   ├── myAD.py                      # Core model (ModelConfig, Trainer, Predictor, etc.)
│   ├── dataset/                     # Dataset abstraction layer (Facade pattern)
│   │   ├── __init__.py              #   Unified API: get_dataloader()
│   │   ├── mvtec.py                 #   MVTec AD dataset
│   │   └── visa.py                  #   VisA dataset
│   ├── utils.py                     # Metrics, logging, feature aggregation, DINOv2 loader
│   ├── perlin.py                    # Perlin noise mask generation
│   ├── config.py                    # TOML → ModelConfig parser
│   └── commen_import.py             # Shared third-party imports
├── facebookresearch_dinov2_main/    # Git submodule: DINOv2 source
├── model_ckpt/                      # Model checkpoints (gitignored)
├── model_log/                       # Training logs (gitignored)
├── model_onnx/                      # ONNX models (gitignored)
├── outputs/                         # Visualization outputs (gitignored)
└── results/                         # Aggregated CSVs (gitignored)
```

## Key Features

### 1. Dual-Branch GAN

| Branch | Noise Location | Loss | Purpose |
|--------|---------------|------|---------|
| **Perlin** | PCA mask ∩ Perlin noise | BCE | Precise noise localization |
| **PCA** | Entire PCA foreground | Hinge | Global discrimination |

Both branches share a single discriminator. Falls back to single-branch Hinge loss when PCA mask is disabled.

### 2. PCA Foreground Mask

- GPU-accelerated SVD on DINOv2 features (first principal component)
- **PCA Student**: a lightweight MLP trained to predict SVD binary masks — ~343× speedup at inference. Trained on-the-fly per category, not persisted to disk.
- Adaptive threshold with center-region protection; texture classes automatically skipped.
- Per-category threshold tuning supported.

### 3. Perlin Noise Mask

- Multi-scale Perlin noise constrained within eroded PCA foreground regions
- Morphological erosion keeps noise concentrated near object centers

### 4. Noise Annealing

Cosine/linear/exponential decay of Gaussian noise strength over training epochs. Early epochs: large noise explores anomaly space; later epochs: small noise for fine convergence. Disabled in few-shot mode.

### 5. Few-Shot Learning

- `--k_shot N`: use only N normal samples per category
- `--shot_seed S`: control sample selection
- Replacement sampling fills batches; auto-enables geometric + color augmentation
- Multi-seed averaging reduces sampling variance
- Independent checkpoint/log naming: `{cat}_k{K}_s{seed}_best_ckpt.pth`

### 6. Dataset Facade

`src/dataset/__init__.py` exports `get_dataloader(root_dir, category, dataset_type, ...)` — a single unified entry point that dispatches to MVTec or VisA loaders. To add a new dataset: create a module in `dataset/`, register in `_LOADER_MAP`.

### 7. Visualization Suite

`CategoryVisualizer` (Strategy + Template Method) orchestrates 5 visualization types:

| Output | Content |
|--------|---------|
| `{category}_heatmap.png` | N×3 grid: Original + GT Mask + plasma heatmap overlay (random sampling, percentile normalization, F1-threshold filtering) |
| `augmented/{category}_augmented.png` | Data augmentation preview (few-shot) |
| `pca_mask/{category}_pca_mask.png` | PCA mask: SVD vs Student comparison |
| `perlin_mask/{category}_perlin_mask.png` | Perlin mask overlay |
| `feature_map/{category}_feature_map.png` | DINOv2 L2-norm feature activation |

### 8. ONNX Deployment

Export end-to-end ONNX models for inference without PyTorch:

| PCA Mode | Output | Input |
|----------|--------|-------|
| SVD | `{category}_full.onnx` | image + mask |
| PCA Student | `{category}_full_student.onnx` | image only |

## Training Pipeline

```
[Pre-training] PCA Student (optional)
  Train images → DINOv2 features → SVD masks (GT) → BCE train MLP

[GAN Training]
  Input [B, 3, 518, 518]
    ↓ Frozen DINOv2 ViT-S/14 (layers [2,5,8,11])
  Multi-layer features × 4
    ↓ _embed_legacy aggregation
  Feature patches [B*H*W, 1536]
    ↓ PCA foreground mask (PCA Student or SVD)
  Foreground features [N, 1536]
    ↓ Projection MLP
  Projected features [N, 1536]
    ↓ + Gaussian noise
  Real/Fake → Discriminator MLP → anomaly scores
```

Inference:

```
Test image → DINOv2 → aggregate → PCA mask → Projection → Discriminator → negative score
                                                                                  ↓
                                                          Upsample + Gaussian blur → heatmap
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Image AUROC** | Image-level anomaly detection |
| **Pixel AUROC** | Pixel-level anomaly segmentation |
| **Pixel PRO** | Per-Region Overlap — anomaly localization accuracy |
| **AP** | Average Precision |
| **F1** | F1 Score at optimal PR threshold |

Fast evaluation (during training): AUROC only. Full evaluation (post-training): all metrics with cross-normalization ensemble.

## Configuration

All hyperparameters in `config.toml`:

```toml
[architecture]     # Input size, layer indices, feature dimensions
[training]         # meta_epochs, gan_epochs, batch_size, learning rates
[noise]            # Gaussian noise std, annealing parameters
[pca_mask]         # PCA mask threshold, skip categories
[pca_student]      # PCA Student MLP config
[perlin_mask]      # Perlin mask, dual-branch weights
[augment]          # Geometric/color augmentation categories
[category_pca]     # Per-category PCA thresholds & borders
[paths]            # Data, model, log, output paths
```

## License

Apache License 2.0
