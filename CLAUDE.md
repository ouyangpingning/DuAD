# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DuAD — unsupervised industrial visual anomaly detection using frozen DINOv2 (`dinov2_vits14_reg`) features + a dual-branch GAN discriminator. Supports full-shot and few-shot (K-shot) training on MVTec AD and VisA datasets.

## Commands

```bash
# Install
pip install -r requirements.txt

# Train (full-shot, MVTec AD) — default aggregation is "fusion"
python src/main.py --categories "bottle screw"
# Few-shot
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0
# VisA dataset
python src/main.py --categories "candle" --dataset visa

# Ablation experiments (all use k_shot=4, 5 seeds, on MVTec AD)
bash train_ablation.sh dino2_only       # No PCA mask → single-branch Hinge
bash train_ablation.sh pca_only         # PCA only, no Perlin branch
bash train_ablation.sh no_augment       # Full DuAD without data augmentation
bash train_ablation.sh channel_concat   # Channel-concat aggregation instead of fusion
bash train_ablation.sh neighborhood     # Neighborhood aggregation instead of fusion
bash train_ablation.sh all              # Run all five variants sequentially

# Visualization
python src/viz/visualize_feature.py --categories "bottle screw"
python src/viz/visualize_feature.py --categories "bottle" --num_samples 8 --skip_inference
python src/viz/visualize_feature.py --categories "bottle" --k_shot 4 --shot_seed 0 --compare_k_shot 1 --query_seed 100

# ONNX export (SVD mode only)
python src/deploy/export_onnx.py --category bottle
python src/deploy/export_onnx.py --category bottle --verify

# Result aggregation
python src/analysis/aggregate_results.py --csv
```

There are no tests in this repository.

## Architecture

### Training pipeline

```
Input [B,3,518,518]
  → frozen DINOv2 ViT-S/14 (layers [2,5,8,11])
  → FeatureAggregator (neighborhood / channel_concat / fusion)
  → PCA foreground mask (SVD with cached direction vector)
  → Projection MLP (1536→1536)
  → + Gaussian noise (optional annealing)
  → Discriminator MLP → anomaly score
```

### Core module (`src/DuAD.py`)

All major classes live here (~1400 lines). The top-level orchestrator is `DINOv2AnomalyDetector`, which owns:

| Component | Line | Role |
|-----------|------|------|
| `FeatureAggregator` | 86 | Three strategies: `"neighborhood"` (3×3 Unfold → AdaptiveAvgPool1d), `"channel_concat"` (纯通道拼接), `"fusion"` (门控融合: per-patch gate MLP 自适应加权两分支, 默认) |
| `FeatureExtractor` | 240 | Wraps DINOv2 encoder + `FeatureAggregator` (frozen, no training). Moves aggregator to device at init |
| `PCAMaskGenerator` | 303 | GPU-native SVD → binary foreground mask. **SVD direction vector cached on first call**, subsequent calls use O(N·D) projection (∼1000× speedup). Morphological ops done in PyTorch. Texture classes skipped per `skip_categories` |
| `PerlinMaskGenerator` | 551 | Multi-scale Perlin noise constrained within eroded PCA foreground. Uses numpy/scipy on CPU (known hotspot) |
| `Projection` | 639 | 2-layer Linear MLP (1536→1536) — projects features before discriminator |
| `Discriminator` | 663 | Configurable MLP (LeakyReLU) outputting scalar anomaly scores |
| `Trainer` | 693 | GAN training loop: dual-branch loss (Perlin BCE + PCA Hinge), noise annealing, optimizer + scheduler per component |
| `Predictor` | 1103 | Inference: patch scores stay on GPU through upsample+GaussianBlur, aggregation (max/topk) on GPU via torch ops, single CPU transfer at end |
| `ModelConfig` | 2 | Dataclass holding all hyperparameters (populated from `config.toml`) |

**Key design decisions:**
- DINOv2 is **frozen** — only `Projection` and `Discriminator` are trained
- Default aggregation is `"fusion"` (gated blend of neighborhood + channel_concat)
- Dual-branch: Perlin branch uses BCE loss on noise-localized regions; PCA branch uses Hinge loss on entire foreground. Both share one discriminator
- When PCA mask is disabled, Perlin is auto-disabled and the system falls back to single-branch Hinge loss
- All PCA mask operations stay on GPU (PyTorch-native morphological ops)
- PCA SVD direction vector is computed once per category and cached; `set_category()` clears the cache
- The `macaroni2` category uses a combined score (0.5×Image AUROC + 0.5×Pixel AUROC) for best-checkpoint selection; all others use Image AUROC primarily
- **No PCA Student** — deleted. Cached SVD makes the MLP student obsolete

### Configuration system

`config.toml` is the **single source of truth** for all hyperparameters. `src/config.py` reads it and builds a `ModelConfig` dataclass. Sections: `[architecture]`, `[training]`, `[noise]`, `[pca_mask]`, `[perlin_mask]`, `[augment]`, `[category_pca]`, `[paths]`, `[misc]`.

Per-category PCA thresholds and border values live in `[category_pca.threshold]` and `[category_pca.border]` — these override the defaults for specific categories.

### Dataset layer (Facade pattern)

`src/dataset/__init__.py` exports `get_dataloader(root_dir, category, dataset_type, ...)` — a single entry point that dispatches to `mvtec.py` or `visa.py` based on `dataset_type`.

### Entry points

| File | Purpose | CLI |
|------|---------|-----|
| `src/main.py` | Training | Click |
| `src/viz/visualize_feature.py` | Visualization (5 output types + comparison mode) | Click |
| `src/deploy/export_onnx.py` | ONNX export (SVD mode only) | argparse |
| `src/analysis/aggregate_results.py` | Log aggregation & CSV stats | argparse |

### Supporting modules

- `src/utils.py` — metrics (AUROC, AP, F1, PRO), `setup_logger` (colorlog), `set_seed`, DINOv2 model downloader
- `src/commen_import.py` — centralized third-party imports used by all modules via `from commen_import import *`

### `simplenet/` — baseline comparison

A parallel project implementing SimpleNet with WideResNet50-2 backbone. Reuses `src/` utilities via `sys.path` manipulation. Has its own `config.toml`, `main.py`, and training scripts. Independent from the DuAD codebase.

### Shell scripts

The `train_*.sh` and `visualize_*.sh` scripts use **tmux** for parallel GPU scheduling — they partition GPU memory (3 GB per process), split categories and seeds across tmux sessions, and launch training commands inside each session.

## Most important: `Trainer._train_step()` three-path logic (`DuAD.py:919`)

There are **three** training paths, controlled by two config flags:

```
pca_generator = PCAMaskGenerator(...) if use_pca_mask else None   # line ~714

_train_step():
  if pca_mask is not None:               # pca_generator exists
      if use_perlin_mask:                 # both PCA + Perlin
          → DUAL BRANCH (line ~955)
            Perlin BCE on masked features + PCA Hinge on all foreground
      else:                               # PCA only, no Perlin
          → PCA-ONLY HINGE (line ~1039)
            Hinge on foreground patches
  else:                                   # pca_generator = None
      → SINGLE HINGE (line ~1073)
        Hinge on all patches, no foreground split
```

`--no_pca_mask` disables both PCA and Perlin (see `main.py:326-330`), because the Perlin path is nested inside `if pca_mask is not None`.

## `skip_categories` returns ALL-ONES, not None

`PCAMaskGenerator.__call__()` at line 357: for skip categories (e.g. transistor), returns `torch.ones(...)` — a full-True mask. This means:

- `pca_mask` is still **not None** → training enters the dual/PCA branch, NOT single Hinge
- skip categories get Perlin noise on the ENTIRE image (no foreground/background split)
- the only thing skipped is the SVD computation
- PCA mask generation itself is fast (direction vector cached after first call)

## PCA SVD caching (`DuAD.py:431`)

`_compute_first_pc_svd()` caches the first principal component direction vector after the initial SVD:
- First call: full SVD via `torch.linalg.svd`, caches `_pca_mean` and `_pca_component` to CPU
- Subsequent calls: `(features - cached_mean) @ cached_component` — pure GPU matrix multiplication
- Cache cleared on `set_category()` when switching categories
- Fallback to sklearn PCA (CPU) only if torch SVD fails (rare)

## Perlin CPU bottleneck

`PerlinMaskGenerator` (line 551) uses numpy/scipy on CPU:
- `_rand_perlin_2d_np()` (line ~602): 2D Perlin noise in numpy
- `_generate_perlin_masks()` (line ~865): `.cpu().numpy()` at line ~927
- `ndimage.rotate` at line ~597

This is a known memory hotspot — each batch generates Perlin patterns on CPU then transfers to GPU.

## Predictor GPU optimizations (`DuAD.py:1103`)

`Predictor.predict()` keeps tensors on GPU end-to-end:
- patch_scores stay on GPU → upsample (F.interpolate) → GaussianBlur all on GPU
- Aggregation (max/topk) done via torch ops inline (no separate numpy methods)
- Single `all_masks.extend([m for m in masks.cpu().numpy()])` at the end
- Eliminates 2 unnecessary GPU↔CPU round-trips per batch vs old code

## Ablation CLI → config mapping (`main.py:326-348`)

| CLI flag | config fields modified |
|----------|----------------------|
| `--no_pca_mask` | `use_pca_mask=False`, `use_perlin_mask=False` |
| `--no_perlin_mask` | `use_perlin_mask=False` |
| `--no_augment` | clears all 4 augment category lists |
| `--aggregation X` | `aggregation_type=X` (choices: `neighborhood`, `channel_concat`, `fusion`) |

## Asymmetric Hinge (`DuAD.py:1004`)

PCA branch uses asymmetric margins: `max(0, 1 - D(z)) + max(0, D(z̃))`. Real features must score ≥1, fake must score ≤0. This differs from the symmetric formulation `max(0, m - D(z)) + max(0, m + D(z̃))`. The asymmetry aligns with BCE branch's sigmoid range [0,1].

## Key line numbers (`DuAD.py`)

| Line | Component |
|------|-----------|
| 86 | `FeatureAggregator` — `neighborhood` / `channel_concat` / `fusion` |
| 109 | `_METHODS` set: `{"neighborhood", "channel_concat", "fusion"}` |
| 199 | `_aggregate_fusion()` — gated fusion via per-patch gate MLP |
| 303 | `PCAMaskGenerator` — SVD + cached direction → binary mask |
| 357 | `skip_categories` → all-ones early return |
| 431 | `_compute_first_pc_svd()` — SVD with direction vector caching |
| 551 | `PerlinMaskGenerator` — multi-scale Perlin on CPU |
| 639 | `Projection` — 2-layer Linear MLP |
| 663 | `Discriminator` — MLP with LeakyReLU |
| 693 | `Trainer` |
| 919 | `_train_step()` — the three-path branch point |
| 1004 | Asymmetric Hinge loss (dual branch) |
| 1103 | `Predictor` — inference, GPU-optimized |
| 1231 | `DINOv2AnomalyDetector` |

**Deleted components (no longer exist):**
- `PCAStudent` class — removed. Cached SVD makes MLP student unnecessary
- `Trainer.train_pca_student()` — removed
- `DINOv2AnomalyDetector.train_pca_student()` — removed
- `Predictor._aggregate_max()` / `_aggregate_topk()` — inlined as GPU torch ops
- `benchmark_pca_mask.py` — deleted

## Cross-file import chain

```
main.py → DuAD.DINOv2AnomalyDetector, DuAD.ModelConfig
        → config.load_config, config.build_model_config
        → dataset.get_dataloader
        → utils.setup_logger, utils.set_seed

config.py → DuAD.ModelConfig
          → reads config.toml

export_onnx.py → DuAD.DINOv2AnomalyDetector, DuAD.ModelConfig
               → reimplements _embed_legacy inline (needs sync when FeatureAggregator changes)

simplenet/ → adds src/ to sys.path → from DuAD import Projection
```

## Visualization script (`src/viz/visualize_feature.py`)

Key features:
- `--query_seed` (default=42): fixed seed for reproducible test image sampling
- `--compare_k_shot` / `--compare_shot_seed`: load a second checkpoint for N×4 comparison heatmaps
- `_infer_heatmap()`: shared inference helper used by both main and compare detectors
- `visualize_pca_mask()`: SVD-only 3-column view (原图 | 第一主成分 | PCA掩模), no MLP comparison
- All `plt.savefig` at 600 DPI with `bbox_inches='tight'`

## Training vs inference difference

- Training: `Trainer._train_step()` — noise injection + discriminator loss
- Inference: `Predictor.predict()` — raw discriminator score (no noise), background fills min score, then GPU `GaussianBlur(sigma=4)`, max/topk aggregation (GPU)
- Visualization: uses random sampling (seeded), percentile normalization, F1 threshold, background NaN — different from both
