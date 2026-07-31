# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Architecture — Facade Pattern

`__init__.py` exports a single entry point: `get_dataloader(root_dir, category, dataset_type, ...)`. It dispatches to dataset-specific loaders via `_LOADER_MAP`:

```
_LOADER_MAP = {
    "mvtec": (get_mvtec_dataloader, {}),
    "visa":  (get_visa_dataloader,  {"csv_name": "1cls"}),
}
```

Each value is `(factory_function, extra_defaults)`. `extra_defaults` are merged into the kwargs passed to the factory — this is how VisA gets its default `csv_name` without the caller needing to know about it.

**To add a new dataset:** (1) create `dataset/new_dataset.py`, (2) implement `get_new_dataloader()` with the same signature, (3) register in `_LOADER_MAP`, (4) caller unchanged — `get_dataloader(..., dataset_type="new_dataset")`.

## Shared return contract

All loaders return `(train_loader, test_loader)`. Each yields batches of `(img, gt, label, adtype)`:
- `img`: tensor `[B, 3, H, W]`, normalized
- `gt`: tensor `[B, 1, H, W]` — 0 for normal samples, binary mask for anomalies
- `label`: int tensor `[B]` — 0 = normal, 1 = anomaly
- `adtype`: list of str — anomaly type name (e.g., `"good"`, `"broken_large"`, `"Anomaly"`)

## K-shot across datasets

Both loaders implement identical k-shot logic:
- Full-shot (`k_shot=None`): `DataLoader` with `shuffle=True`
- K-shot: samples `k` indices via `random.Random(shot_seed).sample()`, wraps in `Subset`, then uses `RandomSampler(replacement=True, num_samples=32)` — replacement ensures each batch is full even when `k_shot < batch_size`

## MVTec (`mvtec.py`)

`MvTecDataset` reads the standard MVTec AD directory layout:
```
{root}/{category}/
  train/good/*.png          → label=0, gt=zeros
  test/{defect_type}/*.png  → label=1, gt from ground_truth/{defect_type}/*.png
```

### Custom augmentations

`RandomRotationReplicate` and `RandomTranslationReplicate` use **OpenCV BORDER_REPLICATE** (not black fill) to avoid introducing artificial edges that could confuse the anomaly detector. Both convert torch tensor → numpy → OpenCV → back to torch.

### `get_transform()` — augmentation flags

```python
get_transform(244, 244, flip=True, rotate=True, translate=True, color_jitter=True)
```
Returns `(train_transform, test_transform, gt_transform)`. Test/GT transforms are always deterministic (no augmentation). Four independent boolean flags control which augmentations are added to the train pipeline. These are wired from `config.toml` `[augment]` category lists in `main.py`.

## VisA (`visa.py`)

`VisADataset` uses CSV-based split definitions (unlike MVTec's directory-convention approach):
```
{root}/
  split_csv/{csv_name}.csv   → columns: object,split,label,image,mask
  {category}/
    Data/Images/Normal/*.JPG
    Data/Images/Anomaly/*.JPG
    Data/Masks/Anomaly/*.png
```

### Mask binarization (VisA-specific)

VisA mask PNGs have pixel value 1 for anomaly regions. After `v2.ToDtype(torch.float32, scale=True)`, this becomes 1/255 ≈ 0.004. Line 92 explicitly binarizes: `gt = (gt > 0).float()` to restore clean 0/1 masks. If this step is skipped, downstream metrics (AUROC, PRO) will silently degrade.

### Default csv_name

VisA uses `"1cls"` by default (single-class split). Other options in the split_csv directory include `"2cls_fewshot"` and `"2cls_highshot"`. Pass `csv_name="2cls_fewshot"` via `get_dataloader(..., csv_name="2cls_fewshot")`.

## Callers

| Caller | Purpose | Notes |
|--------|---------|-------|
| `src/main.py` | Training | Wires augment flags from `config.toml` |
| `src/deploy/export_onnx.py` | ONNX export | Only needs `train_loader` (discards `test_loader`) |
| `src/viz/visualize_feature.py` | Visualization | Uses both loaders |
| `simplenet/main.py` | Baseline | Same Facade interface |
