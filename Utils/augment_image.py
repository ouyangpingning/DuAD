#!/usr/bin/env python3
"""
数据增强可视化脚本 —— 论文用。

读取一张输入图片，对图片分别应用 config.toml 中定义的四种数据增强方法，
每种方法生成单张（或多张）输出图片，保存为 JPG 格式。

四种增强方法
─────────────
  flip          : 随机水平/垂直翻转
  rotate        : 随机旋转 ±180°（BORDER_REPLICATE 填充，无黑边）
  translate     : 随机平移 ±10%（BORDER_REPLICATE 填充，无黑边）
  color_jitter  : 颜色抖动（hue=0.3）

用法示例
────────
  # 全部四种增强，各生成 3 个变体
  python Utils/augment_image.py --input_path /path/to/image.png --methods all

  # 只做 flip 和 rotate
  python Utils/augment_image.py --input_path /path/to/image.png --methods flip,rotate

  # 每种增强生成 5 个变体，保存到指定目录
  python Utils/augment_image.py -i image.png -o ./paper_figures -m flip,rotate -n 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2


# ============================================================
# 增强算子（与 src/dataset/mvtec.py 实现完全一致）
# ============================================================

class RandomRotationReplicate:
    """随机旋转 ±degrees，OpenCV BORDER_REPLICATE 避免黑边。"""

    def __init__(self, degrees: float = 180.0):
        self.degrees = degrees

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (C, H, W) uint8 [0, 255]"""
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())
        if abs(angle) < 0.1:
            return img

        img_np = img.permute(1, 2, 0).cpu().numpy()
        h, w = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            img_np, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return torch.from_numpy(np.ascontiguousarray(rotated)).permute(2, 0, 1)


class RandomTranslationReplicate:
    """随机平移 ±(max_shift_ratio * 图像尺寸)，BORDER_REPLICATE 避免黑边。"""

    def __init__(self, max_shift_ratio: float = 0.1):
        self.max_shift_ratio = max_shift_ratio

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (C, H, W) uint8 [0, 255]"""
        img_np = img.permute(1, 2, 0).cpu().numpy()
        h, w = img_np.shape[:2]
        max_dx = int(w * self.max_shift_ratio)
        max_dy = int(h * self.max_shift_ratio)
        dx = np.random.randint(-max_dx, max_dx + 1)
        dy = np.random.randint(-max_dy, max_dy + 1)
        if dx == 0 and dy == 0:
            return img
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(
            img_np, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return torch.from_numpy(np.ascontiguousarray(shifted)).permute(2, 0, 1)


# ============================================================
# 核心
# ============================================================

# 增强方法注册表
AUGMENT_METHODS = {
    "flip": "随机水平/垂直翻转",
    "rotate": "随机旋转 ±180°",
    "translate": "随机平移 ±10%",
    "color_jitter": "颜色抖动 (hue=0.3)",
}


def load_image_rgb(path: str) -> torch.Tensor:
    """读取图片并返回 (C, H, W) uint8 tensor。"""
    img = Image.open(path).convert("RGB")
    # 转为 (C, H, W) uint8 tensor（v2.ToImage 自动转换为 uint8）
    return v2.ToImage()(img)


def tensor_to_ndarray(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) uint8 tensor → (H, W, C) uint8 numpy array。"""
    return t.permute(1, 2, 0).cpu().numpy()


def apply_flip(img: torch.Tensor) -> torch.Tensor:
    """随机水平 + 垂直翻转。"""
    img = v2.RandomHorizontalFlip(p=0.5)(img)
    img = v2.RandomVerticalFlip(p=0.5)(img)
    return img


def apply_rotate(img: torch.Tensor) -> torch.Tensor:
    """随机旋转 ±180°，BORDER_REPLICATE。"""
    return RandomRotationReplicate(degrees=180)(img)


def apply_translate(img: torch.Tensor) -> torch.Tensor:
    """随机平移 ±10%，BORDER_REPLICATE。"""
    return RandomTranslationReplicate(max_shift_ratio=0.1)(img)


def apply_color_jitter(img: torch.Tensor) -> torch.Tensor:
    """颜色抖动 (hue=0.3)。"""
    return v2.ColorJitter(hue=0.3)(img)


APPLY_FN = {
    "flip": apply_flip,
    "rotate": apply_rotate,
    "translate": apply_translate,
    "color_jitter": apply_color_jitter,
}


def main():
    parser = argparse.ArgumentParser(
        description="数据增强可视化 — 论文用清晰图片生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  # 全部四种增强，各生成 3 个变体\n"
            "  python Utils/augment_image.py -i data/screw/001.png\n"
            "  # 只做 flip 和 rotate\n"
            "  python Utils/augment_image.py -i img.png -m flip,rotate\n"
            "  # 生成单张图片，指定输出目录\n"
            "  python Utils/augment_image.py -i img.png -m flip -n 1 -o ./figures\n"
        ),
    )
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "-o", "--output_dir", default="./augmented_outputs",
        help="输出目录（默认: ./augmented_outputs）",
    )
    parser.add_argument(
        "-m", "--methods", default="all",
        help="增强方法，逗号分隔。可选: flip, rotate, translate, color_jitter, all（默认: all）",
    )
    parser.add_argument(
        "-n", "--num_variants", type=int, default=3,
        help="每种增强生成的变体数量（默认: 3）",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子（可复现结果）",
    )

    args = parser.parse_args()

    # --- 检查输入 ---
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"错误: 输入图片不存在: {input_path}")
        sys.exit(1)

    # --- 解析方法 ---
    if args.methods.strip().lower() == "all":
        methods = list(AUGMENT_METHODS.keys())
    else:
        methods = [m.strip().lower() for m in args.methods.split(",")]
        unknown = [m for m in methods if m not in AUGMENT_METHODS]
        if unknown:
            print(f"错误: 未知增强方法: {unknown}")
            print(f"可选: {sorted(AUGMENT_METHODS.keys())} 或 all")
            sys.exit(1)

    if args.num_variants < 1:
        print("错误: --num_variants 必须 >= 1")
        sys.exit(1)

    # --- 设置随机种子 ---
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # --- 创建输出目录 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 读取图片 ---
    base_name = input_path.stem  # 不含后缀的文件名
    img_tensor = load_image_rgb(input_path)
    sources_dir = output_dir / "_src"
    sources_dir.mkdir(parents=True, exist_ok=True)
    src_save_path = sources_dir / f"{base_name}.jpg"
    Image.fromarray(tensor_to_ndarray(img_tensor)).save(src_save_path)
    print(f"原图已保存 → {src_save_path}")

    # --- 遍历增强方法 ---
    for method in methods:
        print(f"\n{'='*50}")
        print(f"  {method}: {AUGMENT_METHODS[method]}")
        print(f"{'='*50}")

        apply_fn = APPLY_FN[method]
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.num_variants):
            augmented = apply_fn(img_tensor.clone())
            save_name = f"{base_name}_v{i+1:02d}.jpg"
            save_path = method_dir / save_name
            Image.fromarray(tensor_to_ndarray(augmented)).save(str(save_path))
            print(f"  [{i+1}/{args.num_variants}] {save_path}")

    print(f"\n全部完成！结果保存在: {output_dir.resolve()}")
    print("目录结构:")
    print(f"  {output_dir.name}/")
    print(f"    ├── _src/          ← 原图")
    for m in methods:
        print(f"    ├── {m}/")


if __name__ == "__main__":
    main()
