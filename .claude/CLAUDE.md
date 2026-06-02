# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 关于我
1. 我是一名在读研究生，研究方向为无监督工业视觉异常检测，目前正在改进Simplenet的方法作为自己的研究出发点。
2. 我现在的代码部署在一台服务器系统为 Ubuntu 20.04.6 LTS，配备 Intel Xeon Platinum 8350C (128核) 处理器和 512GB 内存并且含有使用 NVIDIA 4090 显卡的服务器上。
3. 我现在是在服务器上和你进行的对话。

# 我的目标
1. 实现我现有的方法在少样本异常检测中可以得到不错的分数，同时全样本下达到sota水平(目前已经做到)。
2. 毕业是我的目标，所以我最终要发一篇sci 2区的论文。

# Quick commands

```bash
# Train on specified categories
python src/main.py --categories "bottle screw"

# Few-shot training (K=4, single seed)
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0

# Few-shot with multiple seeds (run separately, then average results)
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 42

# Interactive batch training (choose full/few-shot, categories, K, seeds; auto-distributes tmux sessions by GPU memory)
bash train_all_tmux.sh

# Visualize anomaly heatmaps (loads best checkpoints, outputs to ./outputs/)
python src/visualize_feature.py --categories "bottle screw"
python src/visualize_feature.py --categories "bottle screw" --k_shot 4 --shot_seed 0
python src/visualize_feature.py --categories "bottle screw" --num_samples 8
python src/visualize_feature.py --categories "bottle" --skip_inference  # analysis-only, no .pth needed

# Interactive batch visualization
bash visualize_all_tmux.sh

# Export ONNX model for deployment
python src/export_onnx.py --category bottle
python src/export_onnx.py --category bottle --k_shot 2 --shot_seed 0 --verify

# Aggregate results from logs
python scripts/aggregate_results.py
python scripts/aggregate_results.py --csv

# Install dependencies
pip install -r requirements.txt
```

# Project summary

Unsupervised anomaly detection on MVTec AD using frozen DINOv2 (`dinov2_vits14_reg`) features + dual-branch GAN-style discriminator. Extracts multi-layer features at indices `[2,5,8,11]`, aggregates them via `_embed_legacy`, then trains a 2-layer MLP projection + 2-layer MLP discriminator (Hinge/BCE loss) to distinguish real normal features from noise-corrupted fake features. At inference, the negative discriminator output is the anomaly score.

## Key files

| File | Role |
|------|------|
| `src/myAD.py` | Core: `ModelConfig`, `FeatureExtractor`, `PCAMaskGenerator`, `PCAStudent`, `Projection`, `Discriminator`, `Trainer`, `Predictor`, `DINOv2AnomalyDetector`, `Visualizer` |
| `src/main.py` | Training entry point. Iterates categories, trains each, saves best checkpoint. Supports few-shot via `--k_shot` / `--shot_seed` |
| `src/dataset.py` | MVTec AD dataset loader (`MvTecDataset`) + transforms. `get_mvtec_dataloader` supports `k_shot` subsampling |
| `src/utils.py` | Metrics (AUROC, AP, F1, PRO), `_embed_legacy`, logger setup, DINOv2 loader |
| `src/perlin.py` | Perlin noise mask generation |
| `src/visualize_feature.py` | Inference/visualization entry point. Uses `CategoryVisualizer` class (Strategy + Template Method) to orchestrate 5 visualization types: anomaly heatmaps (random sampling, percentile norm, F1 threshold, plasma colormap, background NaN), PCA masks (SVD vs PCA Student), Perlin masks, DINOv2 feature maps, and data augmentation previews. Supports `--num_samples`, `--skip_inference` |
| `src/export_onnx.py` | ONNX model export for deployment inference (no PyTorch dependency needed) |
| `src/config.py` | Config loader — reads `config.toml`, builds `ModelConfig`, extracts PCA thresholds and paths |
| `src/commen_import.py` | All shared third-party imports; other modules use `from commen_import import *` (deliberately misspelled, do not rename) |
| `config.toml` | **Single source of truth** for all hyperparameters, paths, category-specific PCA thresholds, and PCA Student config |

## Architecture data flow

**Training**: `src/main.py` → `DINOv2AnomalyDetector.fit()` → `Trainer.train_epoch()`:
  1. Extract features (frozen DINOv2 ViT-S/14, layers [2,5,8,11])
  2. Aggregate via `_embed_legacy` (patchify → adaptive pool per layer → concatenate → pool to 1536-d)
  3. PCA mask (optional): PCAStudent (MLP) predicts foreground probability → sigmoid → >0.5 mask; or fallback SVD → threshold → center-check → mask
  4. Perlin mask (optional): generate Perlin noise within eroded PCA foreground
  5. Projection MLP (1536→1536) on foreground patches
  6. **Dual-branch** (with PCA mask): Perlin branch uses BCE loss on Perlin-region patches; PCA branch uses Hinge loss on all foreground patches. **Single-branch** (no PCA mask): Hinge loss on all patches
  7. Noise annealing: cosine decay of Gaussian noise std over meta_epochs

**Inference**: `Predictor.predict()`:
  1. Extract features → PCA mask → Projection → Discriminator negative score
  2. Background patches filled with min score (for evaluation only) → upscale → Gaussian blur (sigma=4)
  3. Image-level: max aggregation over patch scores. Cross-normalization ensemble for evaluation

**Visualization** (`CategoryVisualizer`): differs from training inference — uses random sampling, percentile normalization, F1-threshold filtering, and background NaN masking for clean overlay.

## Key architectural components

- **PCA Student** (`PCAStudent`): MLP trained on-the-fly by `Trainer.train_pca_student()` before GAN training. Phase 1: collect SVD masks via `PCAMaskGenerator`, Phase 2: train `BCEWithLogitsLoss`, Phase 3: plug into Trainer's PCA generator. At inference, sigmoid(logits) > 0.5 replaces the full SVD PCA pipeline (~343× faster). Config: `[pca_student]` in config.toml. Trained per-category, not persisted to disk.
- **PCAMaskGenerator**: GPU-accelerated SVD for PCA foreground/background separation. Auto-reverses mask if center region has too few foreground pixels. Respects `skip_categories` for texture classes.
- **Dual-branch GAN**: Perlin branch (BCE, localized noise) + PCA branch (Hinge, global noise). Weights: `perlin_branch_weight` / `pca_branch_weight`. Falls back to single-branch Hinge when PCA mask disabled.
## Important conventions

- **All hyperparameters are managed in `config.toml`** — edit this file to change params; both `main.py` and `visualize_feature.py` read from here
- `cv2`, `pandas`, `skimage` are imported directly in the files that need them (not via `commen_import`)
- DINOv2 is loaded via `torch.hub.load` with `source='local'` from `facebookresearch_dinov2_main/`
- MVTec AD dataset path set in `config.toml` `[paths] base_dir`
- Checkpoints: `./model_ckpt/{category}/{category}_best_ckpt.pth` (full) or `{category}_k{K}_s{seed}_best_ckpt.pth` (few-shot)
- Logs: `./model_log/{category}/{category}_full.log` (full) or `{category}_k{K}_s{seed}_full.log` (few-shot)
- ONNX models: `./model_onnx/{category}_k{K}_s{seed}_full.onnx`
- Visual outputs: `./outputs/{category}_heatmap.png` (anomaly N×3 grid), `pca_mask/`, `perlin_mask/`, `feature_map/`, `augmented/`
- Docs: `docs/visualize_feature.md` (complete visualization tool reference)
- `Trainer` applies `proj_lr * 0.1` to the actual AdamW optimizer

## Few-shot support

- `--k_shot N`: limit training to N normal samples per category
- `--shot_seed S`: control which samples are selected
- Uses `RandomSampler` with replacement to fill batches
- Auto-enables data augmentation (flip, rotate) for categories in `augment_categories`
- Color augmentation (hue jitter) for categories in `color_augment_categories`
- Noise annealing disabled in few-shot mode (fixed noise_std used)
- Checkpoints/logs suffixed with `_k{K}_s{seed}` to avoid overwriting

## Evaluation metrics

Image-level: AUROC, AP, F1 (with optimal PR threshold). Pixel-level: AUROC, AP, F1, PRO (Per-Region Overlap). Fast eval (during training) computes only AUROC; full eval (after training) computes all metrics with cross-normalization ensemble.

## Testing strategy

No unit test framework. Validate by running 1-2 full categories (e.g., `bottle`, `screw`) and checking that Image AUROC and Pixel AUROC do not regress.

## 代码质量与验证

- **每次修改代码后必须运行快速验证**，在声称任务完成前确认代码没有语法错误和导入错误：
  ```bash
  python -c "from src.main import main; print('Import OK')"
  ```
- 对于涉及多文件修改（>=3 个文件）的任务，修改完成后要检查跨文件一致性：config.toml 中引用的 key 是否在 `src/config.py` 的 `build_model_config` 中有对应项、函数调用签名是否匹配、变量是否在引用前定义。
- 如果修改了 `config.toml`，要验证 `config.toml` → `src/config.py` → `src/myAD.py (ModelConfig)` 这一条链路上的参数名完全一致。

## 行为准则

- **严格控制修改范围**：只做用户明确要求的事情，不要擅自添加额外功能或行为。例如：用户只要求 flip 增强时，不要同时添加 rotation；用户只要求改某个类别时，不要改动其他类别。
- **在批量删除文件或批量注释代码之前**，先列出所有会被修改的文件清单，等待用户确认后再执行。
- **在尝试复杂的实验方案前**（如对抗噪声、新的损失函数等不确定性较高的方向），先分析可行性和实现复杂度，提出 2-3 个备选方案供用户选择，而不是直接投入大量时间实现一个可能走不通的方案。每次只实现一种方案，保留原始代码以便回退。
- 优先使用 `Edit` 工具进行精确修改，避免整文件重写导致意外改动。
