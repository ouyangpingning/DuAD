#!/usr/bin/env python3
"""
PCA 前景掩模可视化脚本 —— 论文用。

输入一张工业图像，利用训练时相同的 DINOv2 特征提取 + PCA 掩模生成管线，
生成 PCA 前景掩模（黄色）与原图的叠加图片，保存为 JPG 格式。

用法示例
────────
  # 使用默认类别 "bottle"
  python Utils/pca_mask_overlay.py -i /path/to/image.png

  # 指定类别（影响 PCA threshold 等类别特定参数）
  python Utils/pca_mask_overlay.py -i image.png -c screw

  # 指定输出路径
  python Utils/pca_mask_overlay.py -i image.png -o ./outputs/screw_pca.jpg

  # 指定类别、调整透明度
  python Utils/pca_mask_overlay.py -i image.png -c metal_nut --alpha 0.5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

# 将 src/ 加入 Python 路径，以便导入项目模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from myAD import FeatureExtractor, PCAMaskGenerator
from config import load_config, build_model_config, get_category_pca_thresholds, get_paths


# ============================================================
# 辅助
# ============================================================

def _resolve_config_path() -> str:
    """在项目根目录寻找 config.toml。"""
    candidate = _PROJECT_ROOT / "config.toml"
    if candidate.exists():
        return str(candidate)
    print(f"错误: 找不到 config.toml（搜索路径: {candidate}）")
    sys.exit(1)


def load_image_preprocessed(image_path: str, target_size: int, device: str) -> torch.Tensor:
    """
    读取图片并按训练管线预处理，返回 (1, C, H, W) 归一化 tensor。

    管线: PIL → RGB → Resize(target_size) → CenterCrop(target_size) → float32/255 → Normalize
    """
    img = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((target_size, target_size)),
        v2.CenterCrop(target_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img)                    # (C, H, W)
    return tensor.unsqueeze(0).to(device)      # (1, C, H, W)


def load_image_display(image_path: str, target_size: int) -> np.ndarray:
    """
    读取图片用于最终可视化（不归一化），返回 (H, W, 3) uint8 RGB。

    与预处理保持同样的 Resize + CenterCrop 以保证空间对齐。
    """
    img = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((target_size, target_size)),
        v2.CenterCrop(target_size),
    ])
    tensor = transform(img)                    # (C, H, W) uint8 [0, 255]
    return tensor.permute(1, 2, 0).cpu().numpy()


def compute_pca_mask(
    images: torch.Tensor,
    extractor: FeatureExtractor,
    pca_generator: PCAMaskGenerator,
    logger=None,
) -> tuple[np.ndarray, tuple]:
    """
    对单张图片跑完整的特征提取 + PCA 掩模管线。

    Args:
        images: (1, C, H, W) 归一化张量
        extractor: FeatureExtractor 实例
        pca_generator: PCAMaskGenerator 实例

    Returns:
        mask_2d: (H, W) bool numpy array，True = 前景
        (feat_h, feat_w): 特征图空间尺寸
    """
    with torch.no_grad():
        features, (feat_h, feat_w) = extractor(images)            # [H*W, 1536]
        mask_1d = pca_generator(features, (feat_h, feat_w))       # [H*W] bool
        mask_2d = mask_1d.reshape(feat_h, feat_w).cpu().numpy()
    return mask_2d, (feat_h, feat_w)


def yellow_overlay(
    image_bgr: np.ndarray,
    mask_2d: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    在 BGR 原图上以黄色半透明叠加掩模区域。

    Args:
        image_bgr: (H, W, 3) uint8 BGR 原图
        mask_2d:   (H, W) bool，True 为前景
        alpha:     叠加透明度 (0=完全透明, 1=完全不透明)

    Returns:
        (H, W, 3) uint8 BGR 叠加结果
    """
    overlay = image_bgr.copy()

    # 黄色 BGR: (0, 255, 255)
    yellow_bgr = np.array([0, 255, 255], dtype=np.uint8)

    fg = mask_2d.astype(bool)
    overlay[fg] = (image_bgr[fg] * (1 - alpha) + yellow_bgr * alpha).astype(np.uint8)

    # 在前景边界画轮廓，使掩模更醒目
    mask_uint8 = mask_2d.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    return overlay


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PCA 前景掩模可视化 — 黄色掩模叠加原图（论文用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python Utils/pca_mask_overlay.py -i data/screw/001.png\n"
            "  python Utils/pca_mask_overlay.py -i img.png -c hazelnut -o outputs/hazel_pca.jpg\n"
            "  python Utils/pca_mask_overlay.py -i img.png -c bottle --alpha 0.5\n"
        ),
    )
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "-o", "--output_path", default=None,
        help="输出图片路径（默认: {input 所在目录}/{name}_pca_overlay.jpg）",
    )
    parser.add_argument(
        "-c", "--category", default="bottle",
        help="类别名（影响 PCA threshold 等类别特定参数，默认: bottle）",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4,
        help="掩模透明度 0~1（默认: 0.4）",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（默认: cuda 如果可用，否则 cpu）",
    )

    args = parser.parse_args()

    # --- 检查输入 ---
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"错误: 输入图片不存在: {input_path}")
        sys.exit(1)

    # --- 确定输出路径 ---
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.parent / f"{input_path.stem}_pca_overlay.jpg"

    # --- 加载配置 ---
    config_path = _resolve_config_path()
    cfg = load_config(str(config_path))
    paths = get_paths(cfg)
    model_cfg = build_model_config(cfg, args.device)

    # 应用类别特定的 PCA 阈值
    category_thresholds = get_category_pca_thresholds(cfg)
    if args.category in category_thresholds:
        model_cfg.pca_threshold = category_thresholds[args.category]
        print(f"类别特定 PCA threshold: {args.category} = {model_cfg.pca_threshold}")

    # --- 加载 DINOv2 特征提取器 ---
    dinov2_dir = paths["dinov2_model_dir"]
    if not Path(dinov2_dir).is_absolute():
        dinov2_dir = str(_PROJECT_ROOT / dinov2_dir)
    print(f"DINOv2 模型路径: {dinov2_dir}")
    print(f"加载 DINOv2 ViT-S/14 (reg) …")

    extractor = FeatureExtractor(
        model_path=dinov2_dir,
        layer_indices=model_cfg.layer_indices,
        patch_size=model_cfg.patch_size,
        device=args.device,
    )
    extractor.eval()

    # --- 初始化 PCA 掩模生成器 ---
    pca_gen = PCAMaskGenerator(
        threshold=model_cfg.pca_threshold,
        border_ratio=model_cfg.pca_border,
        kernel_size=model_cfg.pca_kernel_size,
        use_gpu=model_cfg.pca_use_gpu,
    )
    pca_gen.set_category(args.category)

    # --- 读取并预处理图片 ---
    images = load_image_preprocessed(input_path, model_cfg.target_size, args.device)

    # --- 计算 PCA 掩模 ---
    print(f"计算 PCA 掩模 …")
    mask_2d, (feat_h, feat_w) = compute_pca_mask(images, extractor, pca_gen)

    fg_ratio = mask_2d.mean()
    print(f"特征图分辨率: {feat_h}×{feat_w}")
    print(f"前景像素比例: {fg_ratio:.3f}")

    if fg_ratio == 0.0:
        print("警告: PCA 掩模无前景像素，输出全图无叠加。请检查类别/阈值设置。")

    # --- 上采样掩模到图像尺寸 ---
    mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    mask_upsampled = torch.nn.functional.interpolate(
        mask_tensor,
        size=(model_cfg.target_size, model_cfg.target_size),
        mode='nearest',
    ).squeeze().numpy().astype(bool)  # (target_size, target_size)

    # --- 读取原图用于显示 ---
    display_img = load_image_display(input_path, model_cfg.target_size)  # (H, W, 3) RGB
    display_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

    # --- 黄色叠加 ---
    result_bgr = yellow_overlay(display_bgr, mask_upsampled, alpha=args.alpha)

    # --- 保存 ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_bgr)
    print(f"叠加结果已保存 → {output_path}")


if __name__ == "__main__":
    main()
