# AGENTS.md

Guidance for OpenCode agents working in DuAD (unsupervised industrial anomaly detection: frozen DINOv2 features + dual-branch GAN discriminator). Detailed architecture notes live in `CLAUDE.md` (tracked, keep accurate). This file adds corrections and traps not obvious from docs.

## Golden rule: trust CLAUDE.md, NOT src/CLAUDE.md

- `src/CLAUDE.md` and `src/dataset/CLAUDE.md` are **gitignored and stale** — they document `PCAStudent` and `--use_pca_student`, which were **deleted**. Do not trust their line numbers or flags.
- Root `CLAUDE.md` is current but line numbers drift (~+1 vs. today's `DuAD.py`). Verify behavior before citing line numbers.

## Commands

```bash
python src/main.py --categories "bottle screw"                      # full-shot, MVTec AD
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0
python src/main.py --categories "candle" --dataset visa
python src/viz/visualize_feature.py --categories "bottle" --skip_inference
python src/deploy/export_onnx.py --category bottle --verify
python src/analysis/aggregate_results.py --csv
```

- **No tests, no lint, no typecheck in this repo.** The only verification is running a training/viz command on a small category (e.g. `bottle`).
- **All `*.sh` scripts (`train_ablation.sh`, `train_all_tmux.sh`, `visualize_all_tmux.sh`, `export_onnx_all_tmux.sh`, `aggregate_results.sh`) are interactive prompt-driven only — they accept NO CLI arguments.** Do not call them with args like `bash train_ablation.sh dino2_only` (CLAUDE.md's claim is wrong).

## Architecture traps (see `CLAUDE.md` for depth)

- **Three training paths** in `Trainer._train_step()` (`DuAD.py:920`): dual-branch (Perlin BCE + PCA Hinge) → PCA-only Hinge → single Hinge, gated by `use_pca_mask` then `use_perlin_mask`.
- `--no_pca_mask` also disables Perlin (it's nested inside the PCA branch).
- **`skip_categories` returns an all-ones mask, NOT None** (`PCAMaskGenerator.__call__`, `DuAD.py:359`). Those categories still enter the dual/PCA branch — only the SVD is skipped. Perlin then applies to the whole image.
- **PCA SVD direction vector is cached** per category (`_compute_first_pc_svd`, `DuAD.py:432`); cleared on `set_category()`. Don't "optimize" the SVD path into a student MLP — it was removed for this reason.
- **Perlin mask generation is a CPU numpy/scipy bottleneck** — every batch does `.cpu().numpy()` round-trips. Changing it to GPU is a known optimization target.
- **`macaroni2` best-checkpoint selection uses 0.5×(Image AUROC) + 0.5×(Pixel AUROC)** (`main.py:159`); all other categories use Image AUROC.
- **`export_onnx.py` reimplements feature aggregation inline** (`_BaseONNXModel._aggregate_fusion`, `export_onnx.py`) and reproduces `adaptive_avg_pool1d` via cumsum+Gather (non-divisible sizes can't be exported directly). It implements the **fusion** aggregator including `gate_mlp` (weights loaded from checkpoint). Every export runs an automated consistency check vs the real `FeatureAggregator` full forward. It outputs **patch-level heatmaps `[B, H, W]` — no upsampling/Gaussian blur; the deployment runtime must do that** (mirroring `Predictor._upsample_masks`). **Two modes**: `PCA inline` (default, when ckpt has `pca_mean`/`pca_component`) — input is only the normalized image, mask computed inside the graph; `external mask` fallback for old ckpts. Batch dim is dynamic; Python-level `.shape` unpacking (`B, C, H, W = x.shape`) would bake batch=1 into the graph — always use indexed `x.shape[i]`.
- **Checkpoints persist `agg_state` + `pca_mean`/`pca_component`** (`DINOv2AnomalyDetector.save/load`): the fusion `gate_mlp` is never trained, but if it isn't saved every `DINOv2AnomalyDetector()` construction gets a fresh random gate → training/inference/export can't reproduce each other. Same for the PCA SVD mean/1st-PC computed from the first training batch — saving it means inference/deployment never recompute SVD (`_restore_pca` injects into freshly built Trainer/Predictor after `set_category`). Old checkpoints without these keys load with a warning and keep random gate_mlp / recompute SVD (re-save to pin them).

## Repo layout gotchas

- `config.toml` is the single source of truth; **dataset paths are hardcoded there** (`mvtec_base_dir=/root/siton-tmp/mvtec_anomaly_detection`, `visa_base_dir=/root/siton-tmp/VisA/VisA_20220922`). New data → edit config, not code.
- Gitignored (do not commit, and don't be surprised they're absent): `model_ckpt/`, `model_log/`, `model_onnx/`, `outputs/`, `results/`, `.claude/`, `src/CLAUDE.md`.
- `facebookresearch_dinov2_main/` is a **git submodule**; DINOv2 weights load via `torch.hub.load(model_pth, 'dinov2_vits14_reg', source='local', ...)` from that directory.
- Dataset layer is a Facade: `src/dataset/__init__.py` exports `get_dataloader(...)` dispatching via `_LOADER_MAP`. VisA masks need explicit binarization `gt = (gt > 0).float()` (`visa.py:92`) or metrics silently degrade.
- K-shot training uses `RandomSampler(replacement=True, num_samples=32)` so batches are full even when `k_shot < batch_size`.
- `simplenet/` is a separate baseline project that imports `src/` via `sys.path`.
- Code comments are in Chinese throughout and explain "why" — keep that style.
