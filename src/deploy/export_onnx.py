#!/usr/bin/env python3
"""
ONNX 模型导出脚本

将完整推理流程导出为 ONNX 格式 (SVD mask 外部输入):
    DINOv2 → 特征聚合 → Projection → Discriminator → 背景填充

输出为 patch 级异常热力图 (未上采样、未高斯平滑),
上采样与平滑由部署端推理流程处理, 与 Predictor._upsample_masks 保持一致。

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

        # adaptive_avg_pool1d 窗口边界 (PyTorch 官方公式, 静态, 与输入无关)
        # adaptive_avg_pool1d 非整除输出无法直接导出 ONNX, 用 cumsum+Gather 精确复现
        C = input_planes // len(layer_indices)                 # 384
        ps = embed_patch_size
        align_in = C * ps * ps                                 # 3456 = 384*3*3
        stack_in = len(layer_indices) * input_planes           # 6144 = 4*1536
        self.register_buffer('align_start', self._adaptive_indices(align_in, input_planes)[0])
        self.register_buffer('align_end', self._adaptive_indices(align_in, input_planes)[1])
        self.register_buffer('stack_start', self._adaptive_indices(stack_in, input_planes)[0])
        self.register_buffer('stack_end', self._adaptive_indices(stack_in, input_planes)[1])

    @staticmethod
    def _adaptive_indices(in_len: int, out_len: int):
        """adaptive_avg_pool1d 的窗口边界: start(i)=floor(i*N/L), end(i)=ceil((i+1)*N/L)。

        与 PyTorch ATen 的 AdaptiveAveragePooling 公式一致,
        故 cumsum+Gather 的数值与 F.adaptive_avg_pool1d 逐元素相等。
        """
        r = in_len / out_len
        idx = torch.arange(out_len, dtype=torch.float32)
        start = (idx * r).floor().long()
        end = ((idx + 1) * r).ceil().long()
        return start, end

    def _extract_intermediate_layers(self, image):
        """DINOv2 中间层特征提取 (ONNX 可追踪)。"""
        outputs = self.encoder.get_intermediate_layers(
            image, n=self.layer_indices, reshape=True,
            return_class_token=False, norm=True,
        )
        return list(outputs)

    def _adaptive_avg_pool1d(self, x, start, end):
        """adaptive_avg_pool1d 的 ONNX 可追踪复现 (cumsum + Gather, 全程 GPU)。

        x: [N, 1, in_len] → [N, 1, out_len]; start/end: [out_len] 窗口边界。
        逐元素等于 F.adaptive_avg_pool1d(x, out_len)。
        """
        n = x.shape[0]
        s = torch.cumsum(F.pad(x, [1, 0]), dim=-1)
        num = s.gather(-1, end.expand(n, 1, -1)) - s.gather(-1, start.expand(n, 1, -1))
        return num / (end - start).float()

    def _embed_legacy(self, layer_features):
        """
        特征聚合-邻域分支 (等价 DuAD.FeatureAggregator._aggregate_neighborhood,
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
            # 用 shape 索引而非整体解包, 保证 batch 维在导出后保持动态
            B = feat.shape[0]
            C = feat.shape[1]
            ps_ = self.embed_patch_size
            unfolded = F.unfold(feat, kernel_size=ps, stride=1, padding=pad)
            unfolded = (unfolded
                        .reshape(B, C, ps_, ps_, -1)
                        .permute(0, 4, 1, 2, 3))
            # 与 FeatureAggregator._align_dim 一致: adaptive pool, 而非 interpolate
            aligned = unfolded.reshape(-1, *unfolded.shape[2:])
            aligned = aligned.reshape(aligned.shape[0], 1, -1)
            aligned = self._adaptive_avg_pool1d(aligned, self.align_start, self.align_end)
            align_features.append(aligned.squeeze(1))

        # 与 FeatureAggregator._aggregate_neighborhood 一致: 跨层 adaptive pool
        stacked = torch.stack(align_features, dim=1)
        stacked = stacked.reshape(stacked.shape[0], 1, -1)
        pooled = self._adaptive_avg_pool1d(stacked, self.stack_start, self.stack_end)
        return pooled.reshape(pooled.shape[0], -1)

    def _aggregate_fusion(self, layer_features):
        """门控融合聚合 (等价 DuAD.FeatureAggregator._aggregate_fusion)。

        F_out = gate ⊙ F_neighbor + (1-gate) ⊙ F_channel
        gate  = σ(MLP([F_neighbor; F_channel]))    # gate_mlp 权重来自 checkpoint
        """
        F_neighbor = self._embed_legacy(layer_features)            # [B*N, D]
        concat = torch.cat(layer_features, dim=1)                  # [B, 4C, H, W]
        # 注意: shape 必须用索引访问 (x.shape[0]), 整体解包 (B, C, H, W) 在
        # 导出时会被固化为常量 batch, 动态 batch 部署时 reshape 会错
        C = concat.shape[1]
        F_channel = concat.permute(0, 2, 3, 1).reshape(-1, C)      # [B*N, D]

        gate = self.gate_mlp(torch.cat([F_neighbor, F_channel], dim=-1))
        return gate * F_neighbor + (1 - gate) * F_channel

    def _post_process(self, scores_flat, B):
        """reshape 为 patch 级热力图 + 图像级分数 (不做上采样与高斯平滑)。

        上采样/平滑由部署端推理流程处理, 保证与 Predictor._upsample_masks 解耦。
        """
        H, W = self.H, self.W
        heatmaps = scores_flat.reshape(B, H, W)          # [B, H, W]
        image_scores = heatmaps.reshape(B, -1).max(dim=1).values  # [B]
        return heatmaps, image_scores


# ─── ONNX 模型: PCA 内联 / SVD 掩码外部输入 双模式 ──────────────

class FullAnomalyDetectorONNX(_BaseONNXModel):
    """
    完整 ONNX 模型, 两种模式:

    [PCA 内联模式, 推荐] ckpt 含 pca_mean/pca_component 时自动启用:
        输入:  image  [B, 3, target_size, target_size] (已归一化)
        输出:  heatmaps [B, H, W], image_scores [B]
        掩码在模型内部计算 (mean/1st-PC 由训练时第一个 batch 的
        聚合特征求得并固化, 部署端无需任何 SVD 计算)。

    [mask 外部输入模式] ckpt 无 pca 数据 (旧格式/skip 类别) 时回退:
        输入:  image + mask [B, H*W] bool (外部计算 PCA 前景)
        输出:  heatmaps [B, H, W], image_scores [B]
    """

    def __init__(self, dino_encoder, projection, discriminator,
                 layer_indices, embed_patch_size, target_size, gate_mlp=None,
                 pca_params=None):
        super().__init__(dino_encoder, layer_indices, embed_patch_size,
                         target_size, 384 * len(layer_indices))
        self.projection = projection
        self.discriminator = discriminator
        # fusion 聚合的门控 MLP; 权重 (训练时为初始化值) 从 checkpoint 加载
        self.gate_mlp = gate_mlp
        # PCA 内联参数 (来自 checkpoint): mean/1st-PC 固定, 阈值/边界/核按类别固化
        if pca_params is not None:
            self.register_buffer('pca_mean', pca_params['mean'].float())
            self.register_buffer('pca_component', pca_params['component'].float())
            self.pca_threshold = float(pca_params['threshold'])
            self.pca_border = float(pca_params['border'])
            self.pca_kernel_size = int(pca_params['kernel_size'])
            self.pca_inline = True
        else:
            self.pca_inline = False

    def forward(self, image, mask=None):
        B = image.shape[0]

        layer_features = self._extract_intermediate_layers(image)
        features = self._aggregate_fusion(layer_features)  # fusion: neighborhood + channel_concat

        if self.pca_inline:
            mask = self._pca_mask(features, B).reshape(-1)
        else:
            mask = mask.reshape(-1)  # [B, H*W] → [B*H*W]

        projected = self.projection(features)
        scores = -self.discriminator(projected).squeeze(-1)

        # 背景填充
        large = torch.full_like(scores, 1e10)
        min_fg = torch.where(mask, scores, large).min()
        scores = torch.where(mask, scores, min_fg.expand_as(scores))

        return self._post_process(scores, B)

    def _pca_mask(self, features, B):
        """内联 PCAMaskGenerator.compute_background_mask 完整逻辑。

        投影 → 阈值 → 中心反转 → 膨胀 + 闭运算, 全部 batch 向量化,
        与 DuAD.PCAMaskGenerator 逐元素一致 (形态学为 max_pool2d, 可导出)。
        """
        H, W = self.H, self.W
        feat = features.reshape(B, H * W, -1)
        proj = (feat - self.pca_mean) @ self.pca_component   # [B, N]
        mask_2d = (proj > self.pca_threshold).reshape(B, H, W)

        # 中心区域前景占比 ≤35% 时反转掩码 (float 运算, 避免 bool Where 的 ORT 兼容问题)
        h0, h1 = int(H * self.pca_border), int(H * (1 - self.pca_border))
        w0, w1 = int(W * self.pca_border), int(W * (1 - self.pca_border))
        if h0 < h1 and w0 < w1:
            center = mask_2d.float()[:, h0:h1, w0:w1]
            keep = (center.mean(dim=(1, 2), keepdim=True) > 0.35).float()  # [B,1,1]
            mask_f = mask_2d.float()
            mask_2d = (keep * mask_f + (1 - keep) * (1 - mask_f)) > 0.5

        # 形态学: 膨胀 → 闭运算 (膨胀+腐蚀), 与 PCAMaskGenerator._morphological_process 一致
        k, pad = self.pca_kernel_size, self.pca_kernel_size // 2
        x = mask_2d.float().unsqueeze(1)                                   # [B,1,H,W]
        x = F.max_pool2d(x, k, stride=1, padding=pad) > 0.5                # 膨胀
        x = F.max_pool2d(x.float(), k, stride=1, padding=pad) > 0.5        # 闭运算: 膨胀
        x = (-F.max_pool2d(-x.float(), k, stride=1, padding=pad)) > 0.5    # 闭运算: 腐蚀
        return x.reshape(B, H * W)


# ─── 导出 ──────────────────────────────────────────────────────────

def _pca_params_from_ckpt(detector, config):
    """从 checkpoint 恢复 PCA SVD 参数 (训练第一个 batch 的特征求得)。

    Returns: dict or None (旧 ckpt 无 pca 数据 → 回退 mask 外部输入模式)
    """
    mean, comp = detector._pca_mean, detector._pca_component
    if mean is None or comp is None:
        return None
    return {
        'mean': mean,
        'component': comp,
        'threshold': config.pca_threshold,
        'border': config.pca_border,
        'kernel_size': config.pca_kernel_size,
    }


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
        gate_mlp=detector.feature_extractor.aggregator.gate_mlp,
        layer_indices=config.layer_indices,
        embed_patch_size=config.patch_size,
        target_size=target_size,
        pca_params=_pca_params_from_ckpt(detector, config),
    )
    return detector, onnx_model


def _check_aggregator_consistency(detector, onnx_model, target_size, atol=1e-3):
    """对比 ONNX 内联聚合 vs 真实 FeatureAggregator 完整 forward (fusion)。

    纯 PyTorch 对比 (无需 onnxruntime), 验证聚合重实现与训练/推理路径逐元素一致。
    """
    device = next(onnx_model.parameters()).device
    images = torch.randn(1, 3, target_size, target_size, device=device)
    with torch.no_grad():
        layer_features = detector.feature_extractor._extract_layer_features(images)
        real = detector.feature_extractor.aggregator(layer_features)
        onnx_features = onnx_model._aggregate_fusion(list(layer_features))
    diff = (real - onnx_features).abs().max().item()
    print(f"    feature agg max diff: {diff:.2e}")
    ok = diff < atol
    print(f"  {'[PASS]' if ok else '[FAIL]'} aggregator matches "
          f"FeatureAggregator within {atol:.0e}")
    return ok


def export_onnx(ckpt_path, onnx_path, config, target_size=518, opset_version=17):
    """导出完整 ONNX 模型 (PCA 内联; 无 pca 数据时回退 mask 外部输入)。"""
    device = config.device
    detector, model = _build_detector_and_onnx_model(ckpt_path, config, target_size)
    model.to(device).eval()

    # 聚合一致性检查: ONNX 内联 _aggregate_fusion vs 真实 FeatureAggregator
    _check_aggregator_consistency(detector, model, target_size)

    B, H = 1, target_size // 14
    dummy_image = torch.randn(B, 3, target_size, target_size, device=device)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        'image': {0: 'batch'},
        'heatmaps': {0: 'batch'},
        'image_scores': {0: 'batch'},
    }
    if model.pca_inline:
        # PCA 内联模式: 输入仅 image, mask 在模型内部计算
        torch.onnx.export(
            model, dummy_image, onnx_path,
            input_names=['image'],
            output_names=['heatmaps', 'image_scores'],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        print("[INFO] PCA inline mode: mean/1st-PC embedded from checkpoint")
    else:
        # 回退: mask 外部输入 (旧 ckpt 无 pca 数据)
        dynamic_axes['mask'] = {0: 'batch'}
        dummy_mask = torch.ones(B, H * H, dtype=torch.bool, device=device)
        torch.onnx.export(
            model, (dummy_image, dummy_mask), onnx_path,
            input_names=['image', 'mask'],
            output_names=['heatmaps', 'image_scores'],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        print("[WARN] No PCA params in checkpoint; falling back to external mask input")
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
    """验证 ONNX 模型 (支持 PCA 内联 / mask 外部输入两种模式)。"""
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

    session = ort.InferenceSession(
        onnx_path,
        providers=[p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                   if p in ort.get_available_providers()],
    )

    with torch.no_grad():
        if pt_model.pca_inline:
            pt_heatmaps, pt_scores = pt_model(images)
            ort_hm, ort_sc = session.run(None, {
                'image': images.cpu().numpy().astype(np.float32),
            })
        else:
            mask = torch.ones(1, N, dtype=torch.bool, device=device)
            pt_heatmaps, pt_scores = pt_model(images, mask)
            ort_hm, ort_sc = session.run(None, {
                'image': images.cpu().numpy().astype(np.float32),
                'mask': mask.cpu().numpy(),
            })

    hm_diff = np.abs(pt_heatmaps.cpu().numpy() - ort_hm).max()
    sc_diff = np.abs(pt_scores.cpu().numpy() - ort_sc).max()

    mode = "PCA inline" if pt_model.pca_inline else "external mask"
    print(f"\n  Verification ({mode}, target={target_size}, N_per={N}):")
    print(f"    heatmaps shape:  {ort_hm.shape} (patch 级, 未上采样)")
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
    # 类别特定 PCA 阈值/边界覆盖 (与训练 main.py 一致, 固化进内联掩码)
    from config import get_category_pca_thresholds, get_category_pca_border_thresholds
    config.pca_threshold = get_category_pca_thresholds(cfg).get(category, config.pca_threshold)
    config.pca_border = get_category_pca_border_thresholds(cfg).get(category, config.pca_border)
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
