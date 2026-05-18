#!/usr/bin/env python3
"""
ONNX 模型导出脚本 (端到端版本)

将完整推理流程导出为 ONNX 格式:
    DINOv2 → _embed_legacy 特征聚合 → Projection → Discriminator → 后处理

PCA mask 保留在 Python 端作为预处理,以 mask 输入传给 ONNX 模型。

用法:
    # 导出全样本模型
    python src/export_onnx.py --category bottle

    # 导出少样本模型
    python src/export_onnx.py --category bottle --k_shot 2 --shot_seed 0

    # 导出并验证
    python src/export_onnx.py --category bottle --verify

输出:
    ./model_onnx/{category}_best.onnx             # 全样本, ~85MB
    ./model_onnx/{category}_k{K}_s{S}_best.onnx   # 少样本, ~85MB
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from myAD import DINOv2AnomalyDetector, ModelConfig
from config import load_config, build_model_config


# ─── 卷积核生成 ────────────────────────────────────────────────────

def make_gaussian_kernel(sigma: float = 4.0):
    """生成 2D 高斯卷积核 (替代 cv2.GaussianBlur)。"""
    ksize = int(2 * (4 * sigma) + 1)  # 匹配 OpenCV 行为
    x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g1d = torch.exp(-0.5 * (x / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]  # [K, K]
    return g2d.view(1, 1, ksize, ksize), ksize // 2


# ─── 端到端 ONNX 模型 ──────────────────────────────────────────────

class FullAnomalyDetectorONNX(torch.nn.Module):
    """
    端到端 ONNX 可导出模型。

    输入:
        image:  [B, 3, target_size, target_size]  ImageNet 归一化后的图像
        mask:   [B, H*W] bool, PCA 前景掩码 (True=保留)

    输出:
        heatmaps:     [B, target_size, target_size]  像素级异常热图
        image_scores: [B]                            图像级异常分数 (max)
    """

    def __init__(
        self,
        dino_encoder: torch.nn.Module,
        projection: torch.nn.Module,
        discriminator: torch.nn.Module,
        layer_indices: list,
        embed_patch_size: int,
        target_size: int,
    ):
        super().__init__()
        self.encoder = dino_encoder
        self.projection = projection
        self.discriminator = discriminator
        self.layer_indices = layer_indices          # [2, 5, 8, 11]
        self.embed_patch_size = embed_patch_size    # 3 (Unfold kernel size)
        self.target_size = target_size
        self.dino_patch_size = 14                   # vits14
        self.H = target_size // self.dino_patch_size  # feature map H
        self.W = target_size // self.dino_patch_size  # feature map W
        self.input_planes = 384 * len(layer_indices)  # 1536

        gk, pad = make_gaussian_kernel(4.0)
        self.register_buffer('gaussian_kernel', gk)
        self.padding = pad

    def _extract_intermediate_layers(self, image):
        """
        DINOv2 中间层特征提取。

        直接调用 encoder.get_intermediate_layers()，内部使用显式 for 循环
        (非 hook)，可被 torch.export 追踪。
        """
        outputs = self.encoder.get_intermediate_layers(
            image,
            n=self.layer_indices,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        return list(outputs)

    def _embed_legacy(self, layer_features):
        """
        特征聚合 (与 utils._embed_legacy 等价, 内联实现保证 ONNX 可追踪)。

        layer_features: list of [B, 384, H, W], aligned spatial resolution
        Returns: [B * H * W, input_planes]
        """
        ps = self.embed_patch_size        # 3
        pad = (ps - 1) // 2               # 1
        target_dim = self.input_planes    # 1536
        output_size = self.input_planes   # 1536

        align_features = []
        for feat in layer_features:
            # Unfold: [B, C, H, W] → [B, C*ps*ps, n_patches]
            B, C, H, W = feat.shape
            unfolded = F.unfold(feat, kernel_size=ps, stride=1, padding=pad)
            n_patches = unfolded.shape[-1]                   # H * W

            # → [B, C, ps, ps, n_patches] → [B, n_patches, C, ps, ps]
            unfolded = (
                unfolded.reshape(B, C, ps, ps, n_patches)
                .permute(0, 4, 1, 2, 3)
            )
            # → [B * n_patches, C * ps * ps]
            unfolded = unfolded.reshape(-1, C * ps * ps)

            # AdaptiveAvgPool1d: [B*n_patches, C*ps*ps] → [B*n_patches, target_dim]
            aligned = F.adaptive_avg_pool1d(
                unfolded.unsqueeze(1), target_dim
            ).squeeze(1)
            align_features.append(aligned)

        # Stack layers → flatten → pool
        # [B*n_patches, num_layers, target_dim] → [B*n_patches, 1, num_layers*target_dim]
        stacked = torch.stack(align_features, dim=1)
        stacked = stacked.reshape(stacked.shape[0], 1, -1)
        # → [B*n_patches, output_size]
        pooled = F.adaptive_avg_pool1d(stacked, output_size)
        return pooled.reshape(pooled.shape[0], -1)

    def forward(self, image, mask):
        B = image.shape[0]
        H, W = self.H, self.W
        T = self.target_size
        mask = mask.reshape(-1)  # [B, H*W] → [B*H*W]

        # ── 1. DINOv2 特征提取 ──
        layer_features = self._extract_intermediate_layers(image)

        # ── 2. 特征聚合 ──
        features = self._embed_legacy(layer_features)  # [B*H*W, input_planes]

        # ── 3. Projection → Discriminator ──
        projected = self.projection(features)          # [N, hidden_dim]
        scores = -self.discriminator(projected)        # [N, 1]
        scores = scores.squeeze(-1)                    # [N]

        # ── 4. 背景填充 (PCA mask 之外的 patch 用前景最低分填充) ──
        large = torch.full_like(scores, 1e10)
        fg_scores = torch.where(mask, scores, large)
        min_fg = fg_scores.min()
        scores = torch.where(mask, scores, min_fg.expand_as(scores))

        # ── 5. 空间重塑 → 上采样 → 高斯平滑 ──
        scores_2d = scores.reshape(B, 1, H, W)
        upsampled = F.interpolate(
            scores_2d, size=(T, T), mode='bilinear', align_corners=False,
        )
        blurred = F.conv2d(
            F.pad(upsampled, [self.padding] * 4, mode='reflect'),
            self.gaussian_kernel,
        )

        # ── 6. 输出 ──
        heatmaps = blurred.squeeze(1)                        # [B, T, T]
        image_scores = heatmaps.reshape(B, -1).max(dim=1).values  # [B]
        return heatmaps, image_scores


# ─── 导出 ──────────────────────────────────────────────────────────

def export_onnx(
    ckpt_path: str,
    onnx_path: str,
    config: ModelConfig,
    target_size: int = 518,
    opset_version: int = 17,
):
    device = config.device
    model_path = str(Path("facebookresearch_dinov2_main").resolve())

    # 加载训练好的 PyTorch 模型
    detector = DINOv2AnomalyDetector(
        model_path=model_path, config=config, logger=None,
    )
    detector.load(ckpt_path)

    # 构建 ONNX 导出模型
    model = FullAnomalyDetectorONNX(
        dino_encoder=detector.feature_extractor.encoder,
        projection=detector.projection,
        discriminator=detector.discriminator,
        layer_indices=config.layer_indices,
        embed_patch_size=config.patch_size,
        target_size=target_size,
    )
    model.to(device)
    model.eval()

    # 构造 dummy 输入
    B = 1
    H = target_size // 14  # 37
    W = H
    N_per_image = H * W
    dummy_image = torch.randn(B, 3, target_size, target_size, device=device)
    dummy_mask = torch.ones(B, N_per_image, dtype=torch.bool, device=device)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_image, dummy_mask),
        onnx_path,
        input_names=['image', 'mask'],
        output_names=['heatmaps', 'image_scores'],
        opset_version=opset_version,
    )

    print(f"[OK] ONNX model exported to {onnx_path}")


# ─── 验证 ──────────────────────────────────────────────────────────

def verify_onnx(
    ckpt_path: str,
    onnx_path: str,
    config: ModelConfig,
    target_size: int = 518,
    atol: float = 1e-3,
):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not installed, skip verification")
        return

    device = config.device
    model_path = str(Path("facebookresearch_dinov2_main").resolve())
    B = 1  # 与导出时一致 (固定 batch)
    H = target_size // 14
    W = H
    N_per_image = H * W

    # PyTorch 模型
    detector = DINOv2AnomalyDetector(
        model_path=model_path, config=config, logger=None,
    )
    detector.load(ckpt_path)

    pt_model = FullAnomalyDetectorONNX(
        dino_encoder=detector.feature_extractor.encoder,
        projection=detector.projection,
        discriminator=detector.discriminator,
        layer_indices=config.layer_indices,
        embed_patch_size=config.patch_size,
        target_size=target_size,
    )
    pt_model.to(device)
    pt_model.eval()

    images = torch.randn(B, 3, target_size, target_size, device=device)
    mask = torch.ones(B, N_per_image, dtype=torch.bool, device=device)

    with torch.no_grad():
        pt_heatmaps, pt_scores = pt_model(images, mask)

    # ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = session.run(None, {
        'image': images.cpu().numpy().astype(np.float32),
        'mask': mask.cpu().numpy(),
    })
    onnx_heatmaps, onnx_scores = ort_out

    hm_diff = np.abs(pt_heatmaps.cpu().numpy() - onnx_heatmaps).max()
    sc_diff = np.abs(pt_scores.cpu().numpy() - onnx_scores).max()

    print(f"\n  Verification (B={B}, H={H}, W={W}, target={target_size}, N_per={N_per_image}):")
    print(f"    heatmaps max diff:     {hm_diff:.2e}")
    print(f"    image_scores max diff: {sc_diff:.2e}")

    if hm_diff < atol and sc_diff < atol:
        print(f"  [PASS] ONNX matches PyTorch within {atol:.0e}")
    else:
        print(f"  [FAIL] Difference exceeds tolerance {atol:.0e}")


# ─── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export full model to ONNX (DINOv2 + all)")
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--k_shot', type=int, default=None)
    parser.add_argument('--shot_seed', type=int, default=0)
    parser.add_argument('--target_size', type=int, default=None, help='默认从 config.toml 读取')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    category = args.category
    ckpt_dir = Path('model_ckpt') / category
    onnx_dir = Path('model_onnx')
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if args.k_shot is not None:
        suffix = f"_k{args.k_shot}_s{args.shot_seed}"
        ckpt_path = ckpt_dir / f"{category}{suffix}_best_ckpt.pth"
        onnx_path = onnx_dir / f"{category}{suffix}_full.onnx"
    else:
        ckpt_path = ckpt_dir / f"{category}_best_ckpt.pth"
        onnx_path = onnx_dir / f"{category}_full.onnx"

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    cfg = load_config('config.toml')
    config = build_model_config(cfg, 'cuda' if torch.cuda.is_available() else 'cpu')
    target_size = args.target_size or config.target_size

    print(f"Exporting {ckpt_path} -> {onnx_path}")
    export_onnx(str(ckpt_path), str(onnx_path), config,
                target_size=target_size, opset_version=args.opset)

    if args.verify:
        verify_onnx(str(ckpt_path), str(onnx_path), config,
                    target_size=target_size)


if __name__ == '__main__':
    main()
