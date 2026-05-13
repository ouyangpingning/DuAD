# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick commands

```bash
# Train on specified categories (default: "pill screw toothbrush transistor wood")
python src/main.py --categories "bottle screw"

# Few-shot training (K=4, single seed)
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0

# Few-shot with multiple seeds (run separately, then average results)
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 42
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 123

# Interactive batch training (choose full/few-shot, categories, K, seeds)
bash train_all_tmux.sh

# Visualize anomaly masks (loads best checkpoints, outputs to ./outputs/)
python src/visualize_feature.py --categories "bottle screw"

# Visualize few-shot model results
python src/visualize_feature.py --categories "bottle screw" --k_shot 4 --shot_seed 0

# Interactive batch visualization
bash visualize_all_tmux.sh

# Aggregate results from logs
python scripts/aggregate_results.py
python scripts/aggregate_results.py --csv

# Install dependencies
pip install -r requirements.txt
```

## Project summary

Unsupervised anomaly detection on MVTec AD using frozen DINOv2 (`dinov2_vits14_reg`) features + GAN-style discriminator. Extracts multi-layer features at indices `[2,5,8,11]`, aggregates them via `_embed_legacy`, then trains a 2-layer MLP projection + 2-layer MLP discriminator (Hinge loss) to distinguish real normal features from noise-corrupted fake features. At inference, the negative discriminator output is the anomaly score.

## Key files

| File | Role |
|------|------|
| `src/myAD.py` | Core: `ModelConfig`, `FeatureExtractor`, `PCAMaskGenerator`, `Projection`, `Discriminator`, `Trainer`, `Predictor`, `DINOv2AnomalyDetector`, `Visualizer` |
| `src/main.py` | Training entry point. Iterates categories, trains each, saves best checkpoint. Supports few-shot via `--k_shot` / `--shot_seed` |
| `src/dataset.py` | MVTec AD dataset loader (`MvTecDataset`) + transforms. `get_mvtec_dataloader` supports `k_shot` subsampling |
| `src/utils.py` | Metrics (AUROC, AP, F1, PRO), `_embed_legacy`, logger setup, DINOv2 loader |
| `src/perlin.py` | Perlin noise mask generation |
| `src/commen_import.py` | All shared third-party imports; other modules use `from commen_import import *` |
| `src/visualize_feature.py` | Inference/visualization entry point. Loads best checkpoints, generates anomaly heatmaps, PCA masks, and Perlin masks. Supports `--k_shot` / `--shot_seed` for few-shot models |
| `src/config.py` | Config loader — reads `config.toml`, builds `ModelConfig`, extracts PCA thresholds and paths |
| `config.toml` | **Single source of truth** for all hyperparameters, paths, and category-specific PCA thresholds. Both `main.py` and `visualize_feature.py` read from here |
| `train_all_tmux.sh` | Interactive batch training script. Prompts for full/few-shot mode, categories, K, seeds; auto-distributes across tmux sessions by GPU memory |
| `visualize_all_tmux.sh` | Interactive batch visualization script. Same pattern as train_all_tmux.sh |
| `scripts/aggregate_results.py` | Scan model_log/ recursively, parse results, save CSV to results/ |

## Visualization outputs (in `./outputs/`)

| Path | Content |
|------|---------|
| `{category}_test.png` | Anomaly heatmap for all test samples |
| `pca_mask/{category}_pca_mask.png` | PCA foreground mask (原图, first PC, mask overlay) |
| `perlin_mask/{category}_perlin_mask.png` | Perlin mask visualization (原图, PCA mask, Perlin mask, overlay) |

## Architecture data flow

**Training**: `src/main.py` → `DINOv2AnomalyDetector.fit()` → `Trainer.train_epoch()` → extract features → PCA mask (optional) → Perlin mask (optional, for dual-branch) → Projection → + 高斯噪声 → Discriminator → Hinge/BCE loss → backprop

**Inference**: `Predictor.predict()` → extract features → PCA mask → Projection → Discriminator negative score → fill background with min score → upscale + Gaussian blur → image-level max aggregation

## Important conventions

- **All hyperparameters are managed in `config.toml`** — edit this file to change params across both training and visualization
- All third-party imports go through `commen_import.py` (deliberately misspelled, do not rename)
- `cv2`, `pandas`, `skimage` are imported directly in the files that need them
- DINOv2 is loaded via `torch.hub.load` from a local path: `../facebookresearch_dinov2_main`
- MVTec AD dataset is expected at `/root/siton-tmp/mvtec_anomaly_detection`
- Checkpoints saved per-category to `./model_ckpt/{category}/{category}_best_ckpt.pth` and `*_latest_ckpt.pth`
- Few-shot: `./model_ckpt/{category}/{category}_k{K}_s{seed}_best_ckpt.pth`
- Logs saved per-category: `./model_log/{category}/{category}_full.log` (全样本) or `{category}_k{K}_s{seed}_full.log` (少样本)

## Few-shot support

- `src/main.py`: `--k_shot N` limits training to N normal samples, `--shot_seed S` controls which samples
- `src/visualize_feature.py`: same `--k_shot` / `--shot_seed` to load from per-category subdirectory
- `train_all_tmux.sh`: interactive prompts for mode, categories, K, and seeds
- Checkpoints and logs are suffixed with `_k{K}_s{seed}` to avoid overwriting, stored under `model_ckpt/{category}/` and `model_log/{category}/`

## Key config defaults (in `ModelConfig`)

- `target_size=518`, `layer_indices=[2,5,8,11]`, `input_planes=384*4`
- `meta_epochs=80`, `gan_epochs=4`, `batch_size=8`
- Noise annealing: `use_noise_annealing=True`, `noise_std_max=0.8→0.5`, `noise_anneal_type="cosine"`
- Dual-branch loss: `perlin_branch_weight=1.0`, `pca_branch_weight=1.0`
- PCA mask: `use_pca_mask=True`, `pca_skip_categories` varies by experiment
- `proj_lr=1e-3` in `main.py`, but `Trainer` applies `proj_lr * 0.1`

## Testing strategy

No unit test framework. Validate by running 1-2 full categories (e.g., `bottle`, `screw`) and checking that Image AUROC and Pixel AUROC do not regress.
