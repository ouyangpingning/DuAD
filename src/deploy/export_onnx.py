#!/usr/bin/env python3
"""
ONNX 模型导出脚本

将完整推理流程导出为 ONNX 格式 (SVD mask 外部输入):
    DINOv2 → 特征聚合 → Projection → Discriminator → 后处理

用法:
    python src/deploy/export_onnx.py --category bottle
    python src/deploy/export_onnx.py --category bottle --k_shot 4 --shot_seed 0 --verify

输出:
    ./model_onnx/{category}_full.onnx
"""

import argparse
import logging
import sys
from pathlib import Path
# 让 src/deploy/ 下的模块能导入 src/ 同级模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from DuAD import DINOv2AnomalyDetector, ModelConfig
from config import load_config, build_model_config
from dataset import get_dataloader, get_transform


# ─── 卷积核生成 ────────────────────────────────────────────────────

def make_gaussian_kernel(sigma: float = 4.0):
    """生成 2D 高斯卷积核 (替代 cv2.GaussianBlur)。"""
    ksize = int(2 * (4 * sigma) + 1)
    x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g1d = torch.exp(-0.5 * (x / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d.view(1, 1, ksize, ksize), ksize // 2


# ─── ONNX 模型基类 (共享 DINOv2 特征提取 + 聚合) ─────────────────

class _BaseONNXModel(torch.nn.Module):
    """ONNX 模型基类, 提供 DINOv2 特征提取和 _embed_legacy 聚合。"""

    def __init__(self, dino_encoder, layer_indices, embed_patch_size,
                 target_size, input_planes):
        super().__init__()
        self.encoder = dino_encoder
        self.layer_indices = layer_indices
        self.embed_patch_size = embed_patch_size
        self.target_size = target_size
        self.dino_patch_size = 14
        self.H = target_size // self.dino_patch_size
        self.W = target_size // self.dino_patch_size
        self.input_planes = input_planes

        gk, pad = make_gaussian_kernel(4.0)
        self.register_buffer('gaussian_kernel', gk)
        self.padding = pad

    def _extract_intermediate_layers(self, image):
        """DINOv2 中间层特征提取 (ONNX 可追踪)。"""
        outputs = self.encoder.get_intermediate_layers(
            image, n=self.layer_indices, reshape=True,
            return_class_token=False, norm=True,
        )
        return list(outputs)

    def _embed_legacy(self, layer_features):
        """
        特征聚合 (与 DuAD.FeatureAggregator._aggregate_neighborhood() 等价,
        内联保证 ONNX 可追踪。如需修改聚合逻辑 → 同步更新 DuAD.FeatureAggregator)。

        layer_features: list of [B, 384, H, W]
        Returns: [B*H*W, input_planes]
        """
        ps = self.embed_patch_size
        pad = (ps - 1) // 2
        target_dim = self.input_planes
        output_size = self.input_planes

        align_features = []
        for feat in layer_features:
            B, C, H, W = feat.shape
            unfolded = F.unfold(feat, kernel_size=ps, stride=1, padding=pad)
            unfolded = (unfolded
                        .reshape(B, C, ps, ps, -1)
                        .permute(0, 4, 1, 2, 3)
                        .reshape(-1, C * ps * ps))
            aligned = F.interpolate(
                unfolded.unsqueeze(1), size=target_dim,
                mode='linear', align_corners=False
            ).squeeze(1)
            align_features.append(aligned)

        stacked = torch.stack(align_features, dim=1)
        stacked = stacked.reshape(stacked.shape[0], 1, -1)
        pooled = F.interpolate(
            stacked, size=output_size,
            mode='linear', align_corners=False
        )
        return pooled.reshape(pooled.shape[0], -1)

    def _post_process(self, scores_flat, B):
        """上采样 + 高斯平滑 → heatmaps, image_scores。"""
        H, W, T = self.H, self.W, self.target_size
        scores_2d = scores_flat.reshape(B, 1, H, W)
        upsampled = F.interpolate(
            scores_2d, size=(T, T), mode='bilinear', align_corners=False,
        )
        blurred = F.conv2d(
            F.pad(upsampled, [self.padding] * 4, mode='reflect'),
            self.gaussian_kernel,
        )
        heatmaps = blurred.squeeze(1)
        image_scores = heatmaps.reshape(B, -1).max(dim=1).values
        return heatmaps, image_scores


# ─── ONNX 模型: SVD 模式 (mask 外部输入) ─────────────────────────

class FullAnomalyDetectorONNX(_BaseONNXModel):
    """
    SVD 模式 ONNX 模型。

    输入:
        image:  [B, 3, target_size, target_size]
        mask:   [B, H*W] bool, PCA 前景掩码 (外部 SVD 计算, True=保留)

    输出:
        heatmaps:     [B, target_size, target_size]
        image_scores: [B]
    """

    def __init__(self, dino_encoder, projection, discriminator,
                 layer_indices, embed_patch_size, target_size):
        super().__init__(dino_encoder, layer_indices, embed_patch_size,
                         target_size, 384 * len(layer_indices))
        self.projection = projection
        self.discriminator = discriminator

    def forward(self, image, mask):
        B = image.shape[0]
        mask = mask.reshape(-1)  # [B, H*W] → [B*H*W]

        layer_features = self._extract_intermediate_layers(image)
        features = self._embed_legacy(layer_features)

        projected = self.projection(features)
        scores = -self.discriminator(projected).squeeze(-1)

        # 背景填充
        large = torch.full_like(scores, 1e10)
        min_fg = torch.where(mask, scores, large).min()
        scores = torch.where(mask, scores, min_fg.expand_as(scores))

        return self._post_process(scores, B)


# ─── 导出 ──────────────────────────────────────────────────────────

def _build_detector_and_onnx_model(ckpt_path, config, target_size):
    """
    加载 PyTorch checkpoint, 构建 detector 和 ONNX 模型。

    Returns: (detector, onnx_model)
    """
    model_path = str(Path("facebookresearch_dinov2_main").resolve())
    detector = DINOv2AnomalyDetector( # 创建模型主类
        model_path=model_path, config=config, logger=None,
    )
    detector.load(ckpt_path) # 为 detector 加载 checkpoint 权重

    onnx_model = FullAnomalyDetectorONNX( # 主要的前向传播
        dino_encoder=detector.feature_extractor.encoder,
        projection=detector.projection,
        discriminator=detector.discriminator,
        layer_indices=config.layer_indices,
        embed_patch_size=config.patch_size,
        target_size=target_size,
    )
    return detector, onnx_model


def export_onnx(ckpt_path, onnx_path, config, target_size=518, opset_version=17):
    """导出 SVD 模式 ONNX 模型 (mask 外部输入)。"""
    device = config.device
    _, model = _build_detector_and_onnx_model(ckpt_path, config, target_size)
    model.to(device).eval()

    B, H = 1, target_size // 14
    dummy_image = torch.randn(B, 3, target_size, target_size, device=device)
    dummy_mask = torch.ones(B, H * H, dtype=torch.bool, device=device)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy_image, dummy_mask), onnx_path,
        input_names=['image', 'mask'],
        output_names=['heatmaps', 'image_scores'],
        opset_version=opset_version,
    )
    print(f"[OK] ONNX model exported to {onnx_path}")


# ─── 验证 ──────────────────────────────────────────────────────────

def _ort_available():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        print("[WARN] onnxruntime not installed, skip verification")
        return False


def verify_onnx(ckpt_path, onnx_path, config, target_size=518, atol=1e-3):
    """验证 SVD 模式 ONNX 模型。"""
    if not _ort_available():
        return
    import onnxruntime as ort

    device = config.device
    H = target_size // 14
    N = H * H

    detector, pt_model = _build_detector_and_onnx_model(
        ckpt_path, config, target_size)
    pt_model.to(device).eval()

    images = torch.randn(1, 3, target_size, target_size, device=device)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)

    with torch.no_grad():
        pt_heatmaps, pt_scores = pt_model(images, mask)

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_hm, ort_sc = session.run(None, {
        'image': images.cpu().numpy().astype(np.float32),
        'mask': mask.cpu().numpy(),
    })

    hm_diff = np.abs(pt_heatmaps.cpu().numpy() - ort_hm).max()
    sc_diff = np.abs(pt_scores.cpu().numpy() - ort_sc).max()

    print(f"\n  Verification (target={target_size}, N_per={N}):")
    print(f"    heatmaps max diff:     {hm_diff:.2e}")
    print(f"    image_scores max diff: {sc_diff:.2e}")
    print(f"  {'[PASS]' if max(hm_diff, sc_diff) < atol else '[FAIL]'} "
          f"ONNX matches PyTorch within {atol:.0e}")


# ─── CLI ──────────────────────────────────────────────────────────

def main():
    #
    # ipdb.set_trace()

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Export model to ONNX"
    )
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--k_shot', type=int, default=None)
    parser.add_argument('--shot_seed', type=int, default=0)
    parser.add_argument('--target_size', type=int, default=None,
                        help='默认从 config.toml 读取')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()


    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger("export_onnx")


    # 准备路径
    category = args.category
    ckpt_dir = Path('model_ckpt') / category
    onnx_dir = Path('model_onnx')
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if args.k_shot is not None: # 如果指定了 k_shot, 则使用少样本训练的 checkpoint
        base = f"{category}_k{args.k_shot}_s{args.shot_seed}"
        ckpt_path = ckpt_dir / f"{base}_best_ckpt.pth"
    else:
        base = f"{category}"# 默认使用全量训练的 checkpoint
        ckpt_path = ckpt_dir / f"{category}_best_ckpt.pth"

    if not ckpt_path.exists(): # 检查 checkpoint 是否存在
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return


    # 加载配置
    cfg = load_config('config.toml')
    config = build_model_config(cfg, 'cuda' if torch.cuda.is_available() else 'cpu')
    target_size = args.target_size or config.target_size
    device = config.device

    onnx_path = onnx_dir / f"{base}_full.onnx"
    print(f"Exporting: {ckpt_path} -> {onnx_path}")
    export_onnx(str(ckpt_path), str(onnx_path), config,
                target_size=target_size, opset_version=args.opset)

    if args.verify:
        verify_onnx(str(ckpt_path), str(onnx_path), config,
                    target_size=target_size)


if __name__ == '__main__':
    main()
