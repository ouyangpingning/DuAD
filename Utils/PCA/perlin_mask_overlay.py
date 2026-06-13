#!/usr/bin/env python3
"""
Perlin 噪声掩模可视化脚本 —— 论文用。

输入一张工业图像，先计算 PCA 前景掩模，再在 PCA 前景内生成 Perlin 噪声掩模
（与训练管线完全一致），将 Perlin 掩模以绿色叠加到原图上，保存为 JPG 格式。

管线
────
  原图 → DINOv2 特征 → PCA 掩模 → 腐蚀 PCA → Perlin 噪声（约束在腐蚀区域内）
       → 最终 Perlin 掩模 = PCA_mask & Perlin_noise → 绿色叠加原图

用法示例
────────
  # 默认类别 bottle
  python Utils/perlin_mask_overlay.py -i /path/to/image.png

  # 指定类别和输出路径
  python Utils/perlin_mask_overlay.py -i image.png -c screw -o outputs/screw_perlin.jpg

  # 调整透明度、固定随机种子
  python Utils/perlin_mask_overlay.py -i image.png -c hazelnut --alpha 0.5 --seed 42
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

# 将 src/ 加入 Python 路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from myAD import FeatureExtractor, PCAMaskGenerator, PerlinMaskGenerator
from config import load_config, build_model_config, get_category_pca_thresholds, get_paths


# ============================================================
# 辅助
# ============================================================

def _resolve_config_path() -> str:
    candidate = _PROJECT_ROOT / "config.toml"
    if candidate.exists():
        return str(candidate)
    print(f"错误: 找不到 config.toml（搜索路径: {candidate}）")
    sys.exit(1)


def load_image_preprocessed(image_path: str, target_size: int, device: str) -> torch.Tensor:
    """读取图片并按训练管线预处理，返回 (1, C, H, W) 归一化 tensor。"""
    img = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((target_size, target_size)),
        v2.CenterCrop(target_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0).to(device)


def load_image_display(image_path: str, target_size: int) -> np.ndarray:
    """读取图片用于最终可视化（不归一化），返回 (H, W, 3) uint8 RGB。"""
    img = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((target_size, target_size)),
        v2.CenterCrop(target_size),
    ])
    return transform(img).permute(1, 2, 0).cpu().numpy()


def compute_pca_mask(
    images: torch.Tensor,
    extractor: FeatureExtractor,
    pca_generator: PCAMaskGenerator,
) -> tuple:
    """特征提取 + PCA 掩模，返回 (mask_2d, (feat_h, feat_w))。"""
    with torch.no_grad():
        features, (feat_h, feat_w) = extractor(images)
        mask_1d = pca_generator(features, (feat_h, feat_w))
        mask_2d = mask_1d.reshape(feat_h, feat_w).cpu().numpy()
    return mask_2d.astype(bool), (feat_h, feat_w)


def compute_perlin_mask(
    images: torch.Tensor,
    pca_mask_2d: np.ndarray,
    feat_h: int,
    feat_w: int,
    perlin_gen: PerlinMaskGenerator,
    target_size: int,
) -> np.ndarray:
    """
    在 PCA 前景内生成 Perlin 噪声掩模（与 Trainer._generate_perlin_masks 逻辑一致）。

    1. PCA 掩模上采样到图像分辨率
    2. 腐蚀（erode kernel=5, iterations=2）向内收缩
    3. PerlinMaskGenerator 在腐蚀区域内生成噪声
    4. 最终掩模 = PCA_mask & Perlin_noise

    Returns:
        perlin_2d: (feat_h, feat_w) bool numpy array
    """
    # PCA mask → [1, 1, feat_h, feat_w] float
    pca_tensor = torch.from_numpy(pca_mask_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # 上采样到图像分辨率
    pca_mask_img = F.interpolate(
        pca_tensor,
        size=(target_size, target_size),
        mode='nearest',
    ).squeeze().cpu().numpy().astype(np.uint8)

    # 如果无前景，返回全零
    if pca_mask_img.sum() == 0:
        return np.zeros((feat_h, feat_w), dtype=bool)

    # 腐蚀 PCA 掩模，让 Perlin 区域更靠近前景中心
    kernel = np.ones((5, 5), np.uint8)
    pca_eroded = cv2.erode(pca_mask_img, kernel, iterations=2)

    # 腐蚀后前景消失则回退
    if pca_eroded.sum() == 0:
        pca_eroded = pca_mask_img

    # PerlinMaskGenerator（输出 [feat_h, feat_w]）
    C = images.shape[1]
    perlin_out = perlin_gen(
        img_shape=(C, target_size, target_size),
        feat_size=feat_h,
        mask_fg=pca_eroded.astype(np.float32),
    )  # → [feat_h, feat_w]

    perlin_bool = (perlin_out > 0).astype(bool)

    # 最终 Perlin 掩模：PCA 前景 AND Perlin 噪声
    return pca_mask_2d & perlin_bool


def green_overlay(
    image_bgr: np.ndarray,
    mask_2d: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    在 BGR 原图上以绿色半透明叠加掩模区域，并绘制轮廓。

    Args:
        image_bgr: (H, W, 3) uint8 BGR 原图
        mask_2d:   (H, W) bool，True 为 Perlin 掩模区域
        alpha:     叠加透明度

    Returns:
        (H, W, 3) uint8 BGR 叠加结果
    """
    overlay = image_bgr.copy()

    # 绿色 BGR: (0, 255, 0)
    green_bgr = np.array([0, 255, 0], dtype=np.uint8)

    fg = mask_2d.astype(bool)
    if fg.sum() > 0:
        overlay[fg] = (image_bgr[fg] * (1 - alpha) + green_bgr * alpha).astype(np.uint8)

    # 轮廓
    mask_uint8 = mask_2d.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Perlin 噪声掩模可视化 — 绿色掩模叠加原图（论文用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python Utils/perlin_mask_overlay.py -i data/screw/001.png\n"
            "  python Utils/perlin_mask_overlay.py -i img.png -c hazelnut -o outputs/hazel_perlin.jpg\n"
            "  python Utils/perlin_mask_overlay.py -i img.png -c bottle --alpha 0.5 --seed 42\n"
        ),
    )
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "-o", "--output_path", default=None,
        help="输出图片路径（默认: {input 所在目录}/{name}_perlin_overlay.jpg）",
    )
    parser.add_argument(
        "-c", "--category", default="bottle",
        help="类别名（影响 PCA threshold 等类别特定参数，默认: bottle）",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="掩模透明度 0~1（默认: 0.5）",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子（固定 Perlin 噪声形态）",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（默认: cuda 如果可用）",
    )

    args = parser.parse_args()

    # --- 随机种子 ---
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # --- 检查输入 ---
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"错误: 输入图片不存在: {input_path}")
        sys.exit(1)

    # --- 确定输出路径 ---
    if args.output_path:
        output_path = Path(args.output_path)
        if args.output_path.endswith("/") or args.output_path.endswith("\\") or output_path.is_dir():
            output_path = output_path / f"{input_path.stem}_perlin_overlay.jpg"
        if output_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            output_path = output_path.with_suffix(".jpg")
    else:
        output_path = input_path.parent / f"{input_path.stem}_perlin_overlay.jpg"

    # --- 加载配置 ---
    config_path = _resolve_config_path()
    cfg = load_config(str(config_path))
    paths = get_paths(cfg)
    model_cfg = build_model_config(cfg, args.device)

    # 类别特定 PCA 阈值
    category_thresholds = get_category_pca_thresholds(cfg)
    if args.category in category_thresholds:
        model_cfg.pca_threshold = category_thresholds[args.category]
        print(f"类别特定 PCA threshold: {args.category} = {model_cfg.pca_threshold}")

    # --- 加载 DINOv2 ---
    dinov2_dir = paths["dinov2_model_dir"]
    if not Path(dinov2_dir).is_absolute():
        dinov2_dir = str(_PROJECT_ROOT / dinov2_dir)
    print(f"DINOv2 模型路径: {dinov2_dir}")
    print("加载 DINOv2 ViT-S/14 (reg) …")

    extractor = FeatureExtractor(
        model_path=dinov2_dir,
        layer_indices=model_cfg.layer_indices,
        patch_size=model_cfg.patch_size,
        device=args.device,
    )
    extractor.eval()

    # --- PCA 掩模生成器 ---
    pca_gen = PCAMaskGenerator(
        threshold=model_cfg.pca_threshold,
        border_ratio=model_cfg.pca_border,
        kernel_size=model_cfg.pca_kernel_size,
        use_gpu=model_cfg.pca_use_gpu,
    )
    pca_gen.set_category(args.category)

    # --- Perlin 掩模生成器 ---
    perlin_gen = PerlinMaskGenerator(
        min_scale=model_cfg.perlin_min,
        max_scale=model_cfg.perlin_max,
    )

    # --- 预处理图片 ---
    images = load_image_preprocessed(input_path, model_cfg.target_size, args.device)

    # --- 计算 PCA 掩模 ---
    print("计算 PCA 掩模 …")
    pca_mask_2d, (feat_h, feat_w) = compute_pca_mask(images, extractor, pca_gen)
    print(f"PCA 前景像素比例: {pca_mask_2d.mean():.3f}")

    # --- 计算 Perlin 掩模 ---
    print("生成 Perlin 噪声掩模（约束在腐蚀 PCA 区域内）…")
    perlin_mask_2d = compute_perlin_mask(
        images, pca_mask_2d, feat_h, feat_w, perlin_gen, model_cfg.target_size,
    )
    perlin_ratio = perlin_mask_2d.mean()
    print(f"特征图分辨率: {feat_h}×{feat_w}")
    print(f"Perlin 掩模像素比例: {perlin_ratio:.3f}")

    if perlin_ratio == 0.0:
        print("警告: Perlin 掩模为空，输出全图无叠加。请检查 PCA 前景或重试（Perlin 有随机性）。")

    # --- 上采样到图像尺寸 ---
    mask_tensor = torch.from_numpy(perlin_mask_2d.astype(np.float32))
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_upsampled = F.interpolate(
        mask_tensor,
        size=(model_cfg.target_size, model_cfg.target_size),
        mode='nearest',
    ).squeeze().numpy().astype(bool)

    # --- 原图 + 绿色叠加 ---
    display_img = load_image_display(input_path, model_cfg.target_size)
    display_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

    result_bgr = green_overlay(display_bgr, mask_upsampled, alpha=args.alpha)

    # --- 保存 ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_bgr)
    print(f"叠加结果已保存 → {output_path}")


if __name__ == "__main__":
    main()
