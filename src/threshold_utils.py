"""
阈值计算工具（纯 numpy，无 torch / sklearn 依赖）。
训练端（main.py）使用同一套逻辑，保证「训练时算出的部署阈值」口径固定。

两个量：
1. 图像级部署阈值 —— 原始 patch-max 分数尺度（与 ONNX image_scores 同口径）。
   用途：部署端初始阈值设定，免人工标定（youden / best_f1 / p99）。
2. 像素级 F1-max 分割阈值 —— 原始 amap 尺度（上采样+高斯平滑后、未归一化，
   与 ONNX heatmaps 上采样平滑后的 hm_smooth 同口径）。
   用途：客户端显示异常图时用 hm > threshold 定位异常区域
   （颜色映射仍可做百分位归一化，但二值化必须在原始空间进行，
   因为百分位归一化是逐图自适应的，跨图不可比）。
"""
import numpy as np

_GRID_N = 2000          # 图像级阈值网格点数
_PIXEL_GRID_N = 1000    # 像素级阈值网格点数
_PIXEL_MAX_SAMPLES = 500_000  # 像素级降采样上限（保护内存/耗时）


def percentile_thresholds(normal) -> dict:
    """正常样本高分位（P99 / P99.5）。"""
    normal = np.asarray(normal, dtype=np.float64)
    return {
        "p99": float(np.percentile(normal, 99)),
        "p99_5": float(np.percentile(normal, 99.5)),
    }


def grid_thresholds(normal, abnormal, target_fpr: float = 0.01,
                    grid_n: int = _GRID_N) -> dict:
    """图像级：网格搜索 youden / best_f1 / target_fpr。"""
    normal = np.asarray(normal, dtype=np.float64)
    abnormal = np.asarray(abnormal, dtype=np.float64)
    all_vals = np.concatenate([normal, abnormal])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    grid = np.linspace(lo, hi, grid_n)

    best_youden, best_youden_t = -1.0, 0.0
    best_f1, best_f1_t = -1.0, 0.0
    best_tpr_at_fpr, best_tpr_at_fpr_t = -1.0, 0.0
    for t in grid:
        tp = float((abnormal > t).mean())
        fp = float((normal > t).mean())
        fn = 1.0 - tp
        youden = tp - fp
        if youden > best_youden:
            best_youden, best_youden_t = youden, float(t)
        f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_f1_t = f1, float(t)
        if fp <= target_fpr and tp > best_tpr_at_fpr:
            best_tpr_at_fpr, best_tpr_at_fpr_t = tp, float(t)

    return {
        "youden": best_youden_t,
        "best_f1": best_f1_t,
        f"target_fpr_{target_fpr:g}": best_tpr_at_fpr_t,
        "_youden_j": best_youden,
        "_best_f1": best_f1,
        "_tpr_at_target_fpr": best_tpr_at_fpr,
    }


def evaluate(threshold: float, normal, abnormal) -> dict:
    """给定阈值的回测指标。"""
    normal = np.asarray(normal, dtype=np.float64)
    abnormal = np.asarray(abnormal, dtype=np.float64)
    out = {"normal_fpr": float((normal > threshold).mean())}
    if len(abnormal):
        out["abnormal_tpr"] = float((abnormal > threshold).mean())
    return out


def compute_image_deploy_threshold(normal_scores, abnormal_scores,
                                   target_fpr: float = 0.01,
                                   method: str = "youden",
                                   grid_n: int = _GRID_N) -> dict:
    """图像级部署阈值（打包版）。无缺陷样本回退 p99。"""
    normal = np.asarray(normal_scores, dtype=np.float64)
    abnormal = np.asarray(abnormal_scores, dtype=np.float64)

    if len(abnormal) == 0:
        thresholds = percentile_thresholds(normal)
        recommended = thresholds["p99"]
        method = "p99"
    else:
        grid = grid_thresholds(normal, abnormal, target_fpr, grid_n)
        thresholds = {k: v for k, v in grid.items() if not k.startswith("_")}
        thresholds.update(percentile_thresholds(normal))
        recommended = thresholds[method] if method in thresholds else thresholds["youden"]

    return {"recommended": float(recommended), "method": method, "thresholds": thresholds}


def pixel_f1_max_threshold(amaps, masks_gt, grid_n: int = _PIXEL_GRID_N,
                           max_samples: int = _PIXEL_MAX_SAMPLES) -> dict:
    """像素级 F1-max 分割阈值（原始 amap 尺度）。

    amaps: [N,H,W] 或 [N,1,H,W] 异常分数图；masks_gt: 同 shape 二值 gt。
    返回 {"threshold": float, "f1": float}；f1=-1 表示无法定义（无正/负样本）。
    """
    amaps = np.asarray(amaps, dtype=np.float64)
    masks_gt = np.asarray(masks_gt, dtype=np.int64)
    # 按 batch 维展平，兼容带/不带单 channel 维
    flat_scores = amaps.reshape(amaps.shape[0], -1).ravel()
    flat_gt = masks_gt.reshape(masks_gt.shape[0], -1).ravel()

    n = flat_scores.size
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples).astype(np.int64)
        flat_scores = flat_scores[idx]
        flat_gt = flat_gt[idx]

    gt_pos = flat_gt.astype(bool)
    n_pos = int(gt_pos.sum())
    n_neg = int((~gt_pos).sum())
    if n_pos == 0 or n_neg == 0:
        return {"threshold": float(np.percentile(flat_scores, 99)), "f1": -1.0}

    lo, hi = float(flat_scores.min()), float(flat_scores.max())
    grid = np.linspace(lo, hi, grid_n)

    best_f1, best_t = -1.0, float(lo)
    for t in grid:
        pred = flat_scores > t
        tp = int((pred & gt_pos).sum())
        fp = int((pred & ~gt_pos).sum())
        fn = n_pos - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return {"threshold": best_t, "f1": best_f1}


def heatmap_scale_from_good(masks, labels_gt) -> dict:
    """热力图固定显示尺度（客户端同口径）。

    统计对象 = 仅 good (正常) 样本的 amap 像素, 不掺缺陷样本;
    amap = 上采样 (双线性) + 高斯平滑 (k25/σ4) 后、含背景 min_fg 填充的
    完整图 (即 model.predict() 返回的 masks 元素, 与 ONNX 端 hm_smooth 同尺度)。
    用 P2/P99.9 而非 min/max: 正常样本尾部分布长, max 会被标签/噪声 patch
    拉高, 使缺陷黄区变弱。

    Args:
        masks: list of [1,H,W] numpy amap (model.predict 返回)
        labels_gt: 图像级标签 (0=正常)
    Returns: {"vmin": float, "vmax": float}
    """
    good = [np.asarray(m).reshape(-1) for m, l in zip(masks, labels_gt) if int(l) == 0]
    if not good:
        return {"vmin": 0.0, "vmax": 1.0}
    pixels = np.concatenate(good)
    return {
        "vmin": float(np.percentile(pixels, 2)),
        "vmax": float(np.percentile(pixels, 99.9)),
    }
