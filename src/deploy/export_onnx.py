#!/usr/bin/env python3
"""
ONNX 模型导出脚本

将完整推理流程导出为 ONNX 格式 (SVD mask 外部输入):
    DINOv2 → 特征聚合 → Projection → Discriminator → 背景填充

后处理 (上采样→target_size + 高斯平滑) 已内化进图, 与推理路径
Predictor._upsample_masks 数值一致; 前处理 (Resize+CenterCrop+归一化)
由部署端完成 (输入即归一化图, 与训练 get_transform 输出一致)。

模型契约:
    输入:  image [1, 3, target_size, target_size] float32 已归一化图像
           (部署端自行 Resize+CenterCrop+ImageNet 归一化)
    输出:  heatmaps    [1, 1, target_size, target_size] float 上采样+高斯平滑后的 amap
           image_scores [1] float patch 级最大值 (与部署阈值标定尺度一致)

用法:
    python src/deploy/export_onnx.py --category bottle
    python src/deploy/export_onnx.py --category bottle --k_shot 4 --shot_seed 0 --verify

输出:
    ./model_onnx/{category}_full.onnx
"""

import argparse
import json
import logging
import sys
from pathlib import Path
# 让 src/deploy/ 下的模块能导入 src/ 同级模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from DuAD import DINOv2AnomalyDetector
from config import load_config, build_model_config


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

        # adaptive_avg_pool1d 等价复现: 常数平均矩阵 W[in_len, out_len],
        # avg = x @ W 一次 MatMul 完成。窗口边界按 PyTorch 官方公式求得并
        # 在导出前固化为常数 initializer; 不用 cumsum+Gather 是因为 TRT
        # 10.3 对 CumSum/GatherElements 的实现有数值 bug (掩模全 0)。
        C = input_planes // len(layer_indices)                 # 384
        ps = embed_patch_size
        align_in = C * ps * ps                                 # 3456 = 384*3*3
        stack_in = len(layer_indices) * input_planes           # 6144 = 4*1536
        self.register_buffer('align_W', self._adaptive_weight(align_in, input_planes))
        self.register_buffer('stack_W', self._adaptive_weight(stack_in, input_planes))

        # 后处理内化: 高斯核 (与 torchvision.transforms.GaussianBlur(kernel=25, sigma=4)
        # 一致, 即 Predictor 中 blur; cv2.GaussianBlur((0,0), sigmaX=4) 自动核长也是 25)
        self.register_buffer('blur_kernel', self._gaussian_kernel(25, 4.0).view(1, 1, 25, 25))

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float):
        """一维高斯核 → 二维外积, 归一化, 与 torchvision 的 get_gaussian_kernel2d 相同。"""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        k1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        return k1d[:, None] * k1d[None, :]

    @staticmethod
    def _adaptive_indices(in_len: int, out_len: int):
        """adaptive_avg_pool1d 的窗口边界: start(i)=floor(i*N/L), end(i)=ceil((i+1)*N/L)。

        与 PyTorch ATen 的 AdaptiveAveragePooling 公式一致,
        故平均矩阵乘的数值与 F.adaptive_avg_pool1d 逐元素相等。
        """
        r = in_len / out_len
        idx = torch.arange(out_len, dtype=torch.float32)
        start = (idx * r).floor().long()
        end = ((idx + 1) * r).ceil().long()
        return start, end

    @classmethod
    def _adaptive_weight(cls, in_len: int, out_len: int):
        """adaptive_avg_pool1d 的常数平均矩阵 W[in_len, out_len]。

        第 i 列在 [start(i), end(i)) 窗口内为 1/窗口宽, 其余为 0;
        x @ W 逐元素等于 F.adaptive_avg_pool1d(x, out_len),
        且导出为单个 MatMul, CPU/CUDA/TRT 数值一致。
        """
        start, end = cls._adaptive_indices(in_len, out_len)
        W = torch.zeros(in_len, out_len, dtype=torch.float32)
        widths = (end - start).float()
        for i in range(out_len):
            W[start[i]:end[i], i] = 1.0 / widths[i]
        return W

    def _extract_intermediate_layers(self, image):
        """DINOv2 中间层特征提取 (ONNX 可追踪)。"""
        outputs = self.encoder.get_intermediate_layers(
            image, n=self.layer_indices, reshape=True,
            return_class_token=False, norm=True,
        )
        return list(outputs)

    def _adaptive_avg_pool1d(self, x, weight):
        """adaptive_avg_pool1d 的 ONNX 可追踪复现 (常数矩阵乘, 无 CumSum/Gather)。

        x: [N, 1, in_len] → [N, 1, out_len]; weight: [in_len, out_len] 常数平均矩阵。
        数值与 F.adaptive_avg_pool1d(x, out_len) 一致, 导出为单个 MatMul, TRT 稳定。
        """
        return x @ weight

    def _embed_legacy(self, layer_features):
        """
        特征聚合-邻域分支 (等价 DuAD.FeatureAggregator._aggregate_neighborhood,
        内联保证 ONNX 可追踪。如需修改聚合逻辑 → 同步更新 DuAD.FeatureAggregator)。

        layer_features: list of [B, 384, H, W]
        Returns: [B*H*W, input_planes]
        """
        ps = self.embed_patch_size
        pad = (ps - 1) // 2

        align_features = []
        for feat in layer_features:
            # 用 shape 索引而非整体解包 (batch 固定 1 导出, 导出期全部折叠为常量)
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
            aligned = self._adaptive_avg_pool1d(aligned, self.align_W)
            align_features.append(aligned.squeeze(1))

        # 与 FeatureAggregator._aggregate_neighborhood 一致: 跨层 adaptive pool
        stacked = torch.stack(align_features, dim=1)
        stacked = stacked.reshape(stacked.shape[0], 1, -1)
        pooled = self._adaptive_avg_pool1d(stacked, self.stack_W)
        return pooled.reshape(pooled.shape[0], -1)

    def _aggregate_fusion(self, layer_features):
        """门控融合聚合 (等价 DuAD.FeatureAggregator._aggregate_fusion)。

        F_out = gate ⊙ F_neighbor + (1-gate) ⊙ F_channel
        gate  = σ(MLP([F_neighbor; F_channel]))    # gate_mlp 权重来自 checkpoint
        """
        F_neighbor = self._embed_legacy(layer_features)            # [B*N, D]
        concat = torch.cat(layer_features, dim=1)                  # [B, 4C, H, W]
        # 注意: shape 必须用索引访问 (x.shape[0])。batch 固定 1 导出时
        # 这些索引在导出期折叠为常量; 若以后要动态 batch, 请用 dynamo=True
        # 导出, 而不是手写 shape 索引 (整体解包 (B, C, H, W) 会被固化成常量)
        C = concat.shape[1]
        F_channel = concat.permute(0, 2, 3, 1).reshape(-1, C)      # [B*N, D]

        gate = self.gate_mlp(torch.cat([F_neighbor, F_channel], dim=-1))
        return gate * F_neighbor + (1 - gate) * F_channel

    def _post_process(self, scores_flat, B):
        """reshape 为 patch 级热力图 + 图像级分数 (不做上采样与高斯平滑)。

        上采样/平滑由 _upsample_blur 内化处理, 与 Predictor._upsample_masks 解耦。
        """
        H, W = self.H, self.W
        heatmaps = scores_flat.reshape(B, H, W)          # [B, H, W]
        image_scores = heatmaps.reshape(B, -1).max(dim=1).values  # [B]
        return heatmaps, image_scores

    def _upsample_blur(self, heatmaps):
        """patch 热力图 → 上采样 target_size + 高斯平滑 (等价 Predictor._upsample_masks)。

        reflect padding + 常数核卷积, 导出为 Pad(reflect)+Conv, TRT 均支持。
        """
        x = heatmaps.unsqueeze(1).float()                # [B, 1, H, W]
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode='bilinear', align_corners=False)
        p = self.blur_kernel.shape[-1] // 2              # 12
        x = F.pad(x, (p, p, p, p), mode='reflect')
        return F.conv2d(x, self.blur_kernel)             # [B, 1, target, target]


# ─── ONNX 模型: PCA 内联 / SVD 掩码外部输入 双模式 ──────────────

class FullAnomalyDetectorONNX(_BaseONNXModel):
    """
    完整 ONNX 模型, 两种模式:

    [PCA 内联模式, 推荐] ckpt 含 pca_mean/pca_component 时自动启用:
        输入:  image [1, 3, target, target] float32 已归一化图像
        输出:  heatmaps [1, 1, target, target], image_scores [1]
        掩码在模型内部计算 (mean/1st-PC 由训练时第一个 batch 的
        聚合特征求得并固化, 部署端无需任何 SVD 计算)。

    [mask 外部输入模式] ckpt 无 pca 数据 (旧格式/skip 类别) 时回退:
        输入:  image [1, 3, target, target] float32 + mask [1, H*W] bool
        输出:  heatmaps [1, 1, target, target], image_scores [1]

    前处理 (Resize+CenterCrop+归一化) 由部署端完成 (输入即归一化图);
    后处理 (上采样 + 高斯平滑) 内化在 forward 内, 与 Predictor._upsample_masks 数值一致。
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
            # 固化翻转方向 (Python bool, 导出时写入图; None=旧ckpt逐图判断)
            self.pca_flip = pca_params.get('flip')
            self.pca_inline = True
        else:
            self.pca_inline = False
            self.pca_flip = None

    def forward(self, image, mask=None):
        B = image.shape[0]

        # 输入为部署端已归一化的图像 (前处理外部完成)
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

        heatmaps, image_scores = self._post_process(scores, B)   # patch 级
        return self._upsample_blur(heatmaps), image_scores       # [B,1,target,target], [B]

    def _pca_mask(self, features, B):
        """内联 PCAMaskGenerator.compute_background_mask 逻辑。

        投影 → 阈值 → (固化方向, 不再逐图判断) → 膨胀 + 闭运算。
        pca_flip 在导出时作为 Python bool 固化进图:
          - flip=True : 掩模取补集 ~(proj > thr)
          - flip=False: 直接用 proj > thr
          - flip=None (旧 ckpt): 回退逐图中心判断 (兼容旧模型行为)
        """
        H, W = self.H, self.W
        feat = features.reshape(B, H * W, -1)
        proj = (feat - self.pca_mean) @ self.pca_component   # [B, N]
        mask_2d = (proj > self.pca_threshold).reshape(B, H, W)

        if self.pca_flip is None:
            # 旧 ckpt 无固化方向: 逐图判断 (与旧版行为一致)
            h0, h1 = int(H * self.pca_border), int(H * (1 - self.pca_border))
            w0, w1 = int(W * self.pca_border), int(W * (1 - self.pca_border))
            if h0 < h1 and w0 < w1:
                center = mask_2d.float()[:, h0:h1, w0:w1]
                keep = (center.mean(dim=(1, 2), keepdim=True) > 0.35).float()  # [B,1,1]
                # 翻转时用负投影重阈值 (与 PyTorch 一致), 非取补集
                mask_flip = (-proj > self.pca_threshold).reshape(B, H, W)
                m_f, m_flip_f = mask_2d.float(), mask_flip.float()
                mask_2d = (keep * m_f + (1 - keep) * m_flip_f) > 0.5
        elif self.pca_flip:
            # 固化方向: 与 PCAMaskGenerator 一致, 用负投影重新阈值
            # ((-proj) > thr), 而非取补集 —— 两者对 |proj|<thr 的 patch 语义不同
            mask_2d = (-proj > self.pca_threshold).reshape(B, H, W)

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
    params = {
        'mean': mean,
        'component': comp,
        'threshold': config.pca_threshold,
        'border': config.pca_border,
        'kernel_size': config.pca_kernel_size,
    }
    # 固化翻转方向: 新 ckpt 含 pca_flip → 写死进图; 旧 ckpt → None (逐图判断回退)
    params['flip'] = detector._pca_flip
    if detector._pca_flip is None:
        print("[WARN] ckpt has no pca_flip (old format); mask direction will "
              "be decided per-image inside the ONNX graph (old behavior)")
    return params


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


def _write_deploy_metadata(onnx_path: str, deploy: dict, pca_flip=None):
    """把训练时标定的部署阈值写入 ONNX metadata_props（模型自包含）。

    部署端 (onnx_infer.py) 按 duad.* key 读取, 免再拿验证集标定。
    旧 ckpt 无 deploy 字段时跳过阈值, 仅写输入/输出契约 (本地走 *.threshold.json 兜底)。
    """
    try:
        import onnx
        from onnx import StringStringEntryProto
    except ImportError:
        print("[WARN] onnx package not available, skip writing deploy metadata")
        return

    model = onnx.load(onnx_path)
    props = {mp.key: mp.value for mp in model.metadata_props}

    def put(k, v):
        if v is not None:
            props[k] = str(v)

    if deploy:
        put("duad.category", deploy.get("category"))
        put("duad.image_threshold", deploy.get("image_threshold"))
        put("duad.image_threshold_method", deploy.get("image_threshold_method"))
        put("duad.pixel_threshold", deploy.get("pixel_threshold"))
        put("duad.pixel_f1_max", deploy.get("pixel_f1_max"))
        put("duad.calibrated_at", deploy.get("calibrated_at"))
        put("duad.heatmap_vmin", deploy.get("heatmap_vmin"))
        put("duad.heatmap_vmax", deploy.get("heatmap_vmax"))
        props["duad.deploy"] = json.dumps(deploy, ensure_ascii=False)
    # 后处理已内化, 写明输入输出契约供部署端核对
    put("duad.input", "float32 NCHW normalized image [1,3,target,target] (ImageNet mean/std; client-side Resize+CenterCrop+normalize)")
    put("duad.output", "heatmaps=[B,1,target,target] upsampled+blurred amap; image_scores=[B] patch-max")
    # PCA 翻转方向标记: true/false = 固化 (新版), none = 旧版逐图判断
    if pca_flip is not None:
        props["duad.pca_flip"] = "true" if pca_flip else "false"

    del model.metadata_props[:]
    for k, v in props.items():
        model.metadata_props.append(StringStringEntryProto(key=k, value=v))
    onnx.save(model, onnx_path)
    print(f"[INFO] Deploy thresholds written to ONNX metadata: "
          f"image={deploy.get('image_threshold') if deploy else None}, "
          f"pixel={deploy.get('pixel_threshold') if deploy else None}, "
          f"pca_flip={pca_flip}")


def _constantize_resize_sizes(onnx_path, target_size):
    """把 Resize 的 sizes 输入 (Shape→Slice→Concat 动态链) 替换为常量 initializer。

    torch 导出器对 F.interpolate 恒从输入 Shape 运行时构造 sizes
    (Concat(Slice(Shape(x))[0:2], [target,target])); 输入分辨率固定后该值
    恒等于 [1, C, target, target], 直接固化为常量并清理死节点,
    图内不再残留动态形状运算 (TRT 构建期零 Shape 推导)。
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    shape_of = {}
    for vi in list(model.graph.value_info) + list(model.graph.input):
        dims = vi.type.tensor_type.shape.dim
        if all(d.HasField('dim_value') for d in dims):
            shape_of[vi.name] = [d.dim_value for d in dims]

    new_inits = []
    for n in model.graph.node:
        if n.op_type != 'Resize' or len(n.input) < 4:
            continue
        x_shape = shape_of.get(n.input[0])
        if x_shape is None or len(x_shape) != 4:
            print(f"[WARN] skip constantizing Resize, unknown input shape: {n.name}")
            continue
        sizes = np.array([1, x_shape[1], target_size, target_size], dtype=np.int64)
        init = numpy_helper.from_array(sizes, name=f"{n.input[3]}_const")
        new_inits.append(init)
        n.input[3] = init.name
    if not new_inits:
        print("[WARN] no Resize node found, skip constantizing sizes")
        return

    # 清理死节点: 输出无人引用且非图输出的节点 (迭代至不动点)
    used = {o.name for o in model.graph.output}
    for n in model.graph.node:
        used.update(n.input)
    keep = [n for n in model.graph.node
            if any(o in used for o in n.output)]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    model.graph.initializer.extend(new_inits)
    # sizes 替换后重跑 shape 推断, 让输出形状尽量以静态值自描述
    model = onnx.shape_inference.infer_shapes(model)
    # Resize 上游含 onnx 无法值求值的形状链 (Shape→Concat→Reshape),
    # 推断不出静态输出形状; 输出尺寸即常量化的 sizes, 显式写死
    for o in model.graph.output:
        if o.name == 'heatmaps':
            dims = o.type.tensor_type.shape.dim
            del dims[:]
            for v in (1, 1, target_size, target_size):
                d = dims.add()
                d.dim_value = v
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"[INFO] Constantized {len(new_inits)} Resize sizes to [1, C, {target_size}, {target_size}]")


def export_onnx(ckpt_path, onnx_path, config, target_size=518, opset_version=17,
                detector=None, model=None):
    """导出完整 ONNX 模型 (PCA 内联; 无 pca 数据时回退 mask 外部输入)。

    输入为部署端已归一化的图像 [1, 3, target, target] float32
    (前处理 Resize+CenterCrop+归一化在客户端完成);
    后处理 (上采样 + 高斯平滑) 内化在图内。
    detector/model: 复用外部已构建的实例 (verify 需与导出共用同一模型,
    否则旧 ckpt 无 agg_state 时 gate_mlp 随机初始化, 两边不一致)。
    """
    device = config.device
    if detector is None or model is None:
        detector, model = _build_detector_and_onnx_model(ckpt_path, config, target_size)
    model.to(device).eval()

    # 聚合一致性检查: ONNX 内联 _aggregate_fusion vs 真实 FeatureAggregator
    _check_aggregator_consistency(detector, model, target_size)

    B, H = 1, target_size // 14
    # 已归一化 dummy 图像 (前处理由部署端完成, 图内只做后处理)
    dummy_image = torch.randn(B, 3, target_size, target_size, device=device)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    # batch 固定 1 (部署端单图推理): 不声明任何 dynamic_axes, 全部形状在
    # 导出期折叠为常量 —— 不会生成 TRT 不友好的 If (squeeze 动态维)。
    # 如需动态 batch/分辨率, 用 dynamo=True 导出 (注意其限制)。
    if model.pca_inline:
        # PCA 内联模式: 输入仅 image, mask 在模型内部计算
        torch.onnx.export(
            model, dummy_image, onnx_path,
            input_names=['image'],
            output_names=['heatmaps', 'image_scores'],
            opset_version=opset_version,
        )
        print(f"[INFO] PCA inline mode: mean/1st-PC embedded from checkpoint; "
              f"flip direction = {model.pca_flip}")
    else:
        # 回退: mask 外部输入 (旧 ckpt 无 pca 数据)
        dummy_mask = torch.ones(B, H * H, dtype=torch.bool, device=device)
        torch.onnx.export(
            model, (dummy_image, dummy_mask), onnx_path,
            input_names=['image', 'mask'],
            output_names=['heatmaps', 'image_scores'],
            opset_version=opset_version,
        )
        print("[WARN] No PCA params in checkpoint; falling back to external mask input")

    # Resize sizes 常量化 (消除 Shape→Slice→Concat 动态链) 后再写 metadata
    _constantize_resize_sizes(onnx_path, target_size)

    # 训练时标定的部署阈值随模型写入 metadata（旧 ckpt 无 deploy 字段则跳过）
    _write_deploy_metadata(onnx_path, getattr(detector, "_deploy", None),
                           pca_flip=getattr(model, "pca_flip", None))
    print(f"[OK] ONNX model exported to {onnx_path}")


# ─── 验证 ──────────────────────────────────────────────────────────

def _ort_available():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        print("[WARN] onnxruntime not installed, skip verification")
        return False


def _natural_verify_image(category, data_dirs, target_size):
    """从数据集取一张真实 good 测试图 (Resize 到 target_size, 返回 uint8 CHW)。

    真实图像的 PCA mask 必有前景, 分数不会进入 1e10 哨兵区 (随机噪声图会),
    否则绝对误差比较会被大数舍入淹没而假 FAIL。
    数据集不可用 (路径缺失/无 good 图) 时返回 None。
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    for base in data_dirs:
        # MVTec: {base}/{cat}/test/good/*.png; VisA: {base}/{cat}/Data/Images/Normal/*.JPG
        candidates = [
            Path(base) / category / 'test' / 'good',
            Path(base) / category / 'Data' / 'Images' / 'Normal',
        ]
        for good_dir in candidates:
            if not good_dir.is_dir():
                continue
            imgs = sorted(good_dir.glob('*.png')) + sorted(good_dir.glob('*.JPG')) \
                + sorted(good_dir.glob('*.jpg'))
            if not imgs:
                continue
            pil = Image.open(imgs[0]).convert('RGB')
            pil = pil.resize((target_size, target_size), Image.BILINEAR)
            return np.asarray(pil).transpose(2, 0, 1)  # [3, H, W] uint8
    return None


def _client_preprocess(images_uint8, target_size, device):
    """部署端前处理复现: /255 + ImageNet 归一化 (Resize 已在取图时完成)。

    与训练 get_transform 数值一致; 返回 [1, 3, target, target] float32。
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = images_uint8.to(device).float() / 255.0
    return (x - mean.to(device)) / std.to(device)


def verify_onnx(ckpt_path, onnx_path, config, target_size=518, atol=1e-3,
                category=None, data_dirs=(), detector=None, model=None):
    """验证 ONNX 模型 (支持 PCA 内联 / mask 外部输入两种模式)。

    输入为已归一化图像 (前处理在验证脚本侧复现, 与部署端一致);
    覆盖图内后处理 (上采样+高斯平滑) 路径。PT 侧关闭 TF32, 保证与
    ORT fp32 逐位可比。
    detector/model 必须与导出时共用同一实例 (旧 ckpt 无 agg_state 时
    gate_mlp 随机初始化, 重新构建会得到不同权重导致假 FAIL)。
    """
    if not _ort_available():
        return
    import onnxruntime as ort

    device = config.device
    H = target_size // 14
    N = H * H

    if detector is None or model is None:
        detector, model = _build_detector_and_onnx_model(
            ckpt_path, config, target_size)
    pt_model = model.to(device).eval()

    # uint8 图像: 优先真实 good 图 (分数不进哨兵区), 否则平滑随机回退
    nat = _natural_verify_image(category, data_dirs, target_size)
    if nat is None:
        print("[WARN] no dataset image for verification, using smoothed random image")
        rnd = np.random.RandomState(0).rand(3, target_size, target_size).astype(np.float32)
        rnd = torch.from_numpy(rnd).unsqueeze(0)
        rnd = F.avg_pool2d(rnd, kernel_size=8, stride=1, padding=4)  # 平滑: 更接近真实图像
        nat = (rnd * 255).round().to(torch.uint8).squeeze(0).cpu().numpy()
    images = _client_preprocess(torch.from_numpy(nat).unsqueeze(0), target_size, device)

    session = ort.InferenceSession(
        onnx_path,
        providers=[p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                   if p in ort.get_available_providers()],
    )

    # PT 侧关闭 TF32: 1e10 哨兵值在 TF32 下会有 ~1e6 绝对误差, 与 ORT fp32 不可比
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with torch.no_grad():
        if pt_model.pca_inline:
            pt_heatmaps, pt_scores = pt_model(images)
            ort_hm, ort_sc = session.run(None, {
                'image': images.cpu().numpy(),
            })
        else:
            mask = torch.ones(1, N, dtype=torch.bool, device=device)
            pt_heatmaps, pt_scores = pt_model(images, mask)
            ort_hm, ort_sc = session.run(None, {
                'image': images.cpu().numpy(),
                'mask': mask.cpu().numpy(),
            })

    hm_diff = np.abs(pt_heatmaps.cpu().numpy() - ort_hm).max()
    sc_diff = np.abs(pt_scores.cpu().numpy() - ort_sc).max()

    mode = "PCA inline" if pt_model.pca_inline else "external mask"
    print(f"\n  Verification ({mode}, target={target_size}, input=normalized {target_size}x{target_size}):")
    print(f"    heatmaps shape:  {ort_hm.shape} (上采样+高斯平滑后)")
    print(f"    heatmaps max diff:     {hm_diff:.2e}")
    print(f"    image_scores max diff: {sc_diff:.2e}")
    # 哨兵值场景 (随机回退图 mask 全 False → 分数 ~1e10) 下绝对比较无意义,
    # 退化为相对比较; 真实图 (正常分数量级) 仍是严格绝对比较
    scale = max(1.0, float(np.abs(ort_hm).max()),
                float(pt_heatmaps.cpu().numpy().max()))
    ok = max(hm_diff, sc_diff) < atol * scale
    print(f"  {'[PASS]' if ok else '[FAIL]'} "
          f"ONNX matches PyTorch within {atol:.0e} (scale={scale:.1e})")


# ─── CLI ──────────────────────────────────────────────────────────

def _export_category(args, category, cfg) -> bool:
    """导出单个类别的 ONNX 模型。ckpt 缺失时返回 False 并跳过。

    Returns:
        True 导出成功, False 跳过 (ckpt 不存在)
    """
    ckpt_dir = Path('model_ckpt') / category

    if args.k_shot is not None: # 如果指定了 k_shot, 则使用少样本训练的 checkpoint
        base = f"{category}_k{args.k_shot}_s{args.shot_seed}"
        ckpt_path = ckpt_dir / f"{base}_best_ckpt.pth"
    else:
        base = f"{category}"# 默认使用全量训练的 checkpoint
        ckpt_path = ckpt_dir / f"{category}_best_ckpt.pth"

    if not ckpt_path.exists(): # 检查 checkpoint 是否存在
        print(f"[ERROR] Checkpoint not found: {ckpt_path}, skip '{category}'")
        return False

    # 类别特定 PCA 阈值/边界覆盖 (与训练 main.py 一致, 固化进内联掩码)
    config = build_model_config(cfg, 'cuda' if torch.cuda.is_available() else 'cpu')
    from config import get_category_pca_thresholds, get_category_pca_border_thresholds
    config.pca_threshold = get_category_pca_thresholds(cfg).get(category, config.pca_threshold)
    config.pca_border = get_category_pca_border_thresholds(cfg).get(category, config.pca_border)
    target_size = args.target_size or config.target_size

    onnx_path = Path('model_onnx') / f"{base}_full.onnx"
    print(f"Exporting: {ckpt_path} -> {onnx_path}")
    # 构建一次, 导出与验证共用同一实例 (gate_mlp/参数一致性)
    detector, model = _build_detector_and_onnx_model(
        str(ckpt_path), config, target_size)
    export_onnx(str(ckpt_path), str(onnx_path), config,
                target_size=target_size, opset_version=args.opset,
                detector=detector, model=model)

    if args.verify:
        data_dirs = [cfg['paths']['mvtec_base_dir'],
                     cfg['paths'].get('visa_base_dir', '')]
        verify_onnx(str(ckpt_path), str(onnx_path), config,
                    target_size=target_size,
                    category=category, data_dirs=data_dirs,
                    detector=detector, model=model)
    return True


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Export model to ONNX"
    )
    parser.add_argument('--category', type=str, required=True,
                        help='类别名, 空格分隔多个, 例如 --category "bottle screw"')
    parser.add_argument('--k_shot', type=int, default=None)
    parser.add_argument('--shot_seed', type=int, default=0)
    parser.add_argument('--target_size', type=int, default=None,
                        help='默认从 config.toml 读取')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    # 日志 (格式化 detector 内部的 logging 输出; 本脚本自身用 print)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    # 准备路径 (多类别批量导出)
    categories = args.category.strip().split()
    Path('model_onnx').mkdir(parents=True, exist_ok=True)

    # 加载配置 (PCA 阈值覆盖按类别在 _export_category 内应用)
    cfg = load_config('config.toml')

    exported, skipped = [], []
    for i, category in enumerate(categories):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(categories)}] category: {category}")
        print('='*60)
        ok = _export_category(args, category, cfg)
        (exported if ok else skipped).append(category)

    # 汇总
    print(f"\n{'='*60}")
    print(f"Export summary: {len(exported)} succeeded, {len(skipped)} skipped")
    if exported:
        print("  succeeded:", ", ".join(exported))
    if skipped:
        print("  skipped (no ckpt):", ", ".join(skipped))
    print('='*60)


if __name__ == '__main__':
    main()
