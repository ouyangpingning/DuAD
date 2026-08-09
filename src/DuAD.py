from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from commen_import import *
from utils import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, init_weight, download_dinov2_models, _safe_roc_auc
from sklearn.decomposition import PCA
import scipy.ndimage as ndimage
import cv2
from torchvision.transforms import GaussianBlur

@dataclass
# 模型配置类--用于配置模型的参数
class ModelConfig:
    """模型配置"""
    # 架构参数
    target_size: int = 288
    layer_indices: List[int] = None
    input_planes: int = 768  
    hidden_dim: int = 1024   
    
    # 训练参数
    proj_lr: float = 1e-4
    dsc_lr: float = 1e-4
    gan_epochs: int = 4
    meta_epochs: int = 80
    batch_size: int = 8  # 批次大小
    warmup_epochs: int = 5  # 预热期 meta-epoch 数: 前 N 个 epoch 不参与 best checkpoint 选择 (config.toml 覆盖)
    
    # 噪声参数
    noise_std: float = 0.5
    # 噪声退火参数
    use_noise_annealing: bool = True       # 是否启用噪声强度随epoch退火
    noise_std_max: float = 0.8             # 初始最大噪声强度
    noise_std_min: float = 0.2            # 最终最小噪声强度
    noise_anneal_epochs: int = None        # 退火到最小值所需的epoch数，None表示使用meta_epochs
    noise_anneal_type: str = "linear"      # 退火类型: "linear", "cosine", "exponential"
    
    # PCA掩模参数
    use_pca_mask: bool = False  # 是否使用PCA掩模
    pca_threshold: float = 10.0 # PCA掩模阈值
    pca_border: float = 0.2 # 中心区域边界比例
    pca_kernel_size: int = 3 # 形态学操作核大小
    pca_use_gpu: bool = True           # 是否使用GPU加速PCA
    pca_skip_categories: List[str] = None  # 指定不使用PCA的类别列表（返回全1掩模）

    # 数据增强控制 — 4 种独立增强，可按类别分别指定
    flip_categories: List[str] = None            # 启用随机翻转的类别
    rotate_categories: List[str] = None          # 启用随机旋转的类别
    translate_categories: List[str] = None       # 启用随机平移的类别
    color_jitter_categories: List[str] = None    # 启用颜色抖动的类别

    # Perlin掩模参数
    use_perlin_mask: bool = False      # 是否使用Perlin掩模在PCA基础上进一步限制噪声位置
    perlin_min: int = 0                # Perlin噪声最小尺度
    perlin_max: int = 4                # Perlin噪声最大尺度

    # 双分支损失参数
    perlin_branch_weight: float = 1.0  # Perlin分支BCE损失的权重
    pca_branch_weight: float = 1.0     # PCA分支Hinge损失的权重

    # 消融实验标记
    ablation_tag: str = ""              # 消融变体标识，用于 checkpoint/log 命名

    # 特征聚合方式
    aggregation_type: str = "neighborhood"  # "neighborhood" (当前, _embed_legacy) 或 "channel_concat" (消融 B4, _embed_channel_concat)

    # Scheduler 配置
    use_scheduler: bool = True          # 是否使用学习率调度器
    scheduler_type: str = "cosine"      # 调度器类型: "cosine" (CosineAnnealingLR) 或 "multistep" (MultiStepLR)
    multistep_milestones: List[float] = None  # MultiStepLR 的 milestone (总步数比例), 如 [0.8, 0.9]
    multistep_gamma: float = 0.4        # MultiStepLR 的衰减因子

    # 其他
    patch_size: int = 3
    device: str = "cuda"
    
    def __post_init__(self):
        if self.layer_indices is None:
            self.layer_indices = [2, 5, 8, 11]
        if self.multistep_milestones is None:
            self.multistep_milestones = [0.8, 0.9]


# ═══════════════════════════════════════════════════════════════════
# 特征聚合器 — 所有聚合方法集中在一个类中, 新增方法加私有函数即可
# ═══════════════════════════════════════════════════════════════════

class FeatureAggregator(torch.nn.Module):
    """特征聚合器 — 将 DINOv2 多层特征图聚合为 patch 特征向量。

    支持多种聚合策略, 通过 method 参数切换。新增方法: 添加一个
    _aggregate_xxx() 私有方法, 在 __init__ 的 _METHODS 注册, forward() 加分支。

    Available methods:
        "neighborhood"   — 3×3 邻域 Unfold + Align_dim + Stack + AdaptiveAvgPool1d (默认)
        "channel_concat" — 通道维直接拼接, 无邻域无跨层融合 (消融 B4)
        "fusion"         — 门控融合: neighborhood + channel_concat 自适应加权

    Usage:
        agg = FeatureAggregator(input_dim=384, num_layers=4, method="neighborhood")
        patches = agg(layer_features)  # layer_features: list of [B,384,H,W]

    Args:
        input_dim:  每层特征通道数 (DINOv2 ViT-S/14 = 384)
        num_layers: 使用的层数 (默认 4)
        method:     聚合策略名称
        patch_size: Unfold 窗口大小, 仅 "neighborhood" 使用 (默认 3)
        stride:     Unfold 步长, 仅 "neighborhood" 使用 (默认 1)
    """

    _METHODS = {"neighborhood", "channel_concat", "fusion"}

    def __init__(self, input_dim=384, num_layers=4,
                 method="neighborhood",
                 patch_size=3, stride=1,
                 fusion_hidden_dim=None):
        super().__init__()
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown aggregation method: '{method}'. "
                f"Available: {sorted(self._METHODS)}"
            )
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = input_dim * num_layers  # 384 * 4 = 1536
        self.method = method
        self.patch_size = patch_size
        self.stride = stride

        # 预创建 Unfold 算子，避免每次 forward 重复分配
        padding = (patch_size - 1) // 2
        self.unfolder = torch.nn.Unfold(
            kernel_size=patch_size, stride=stride,
            padding=padding, dilation=1,
        )

        # 门控融合 MLP: 学习每个 patch 在每个通道上信任哪个分支
        if method == "fusion":
            hidden = fusion_hidden_dim or max(self.output_dim // 4, 64)
            self.gate_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.output_dim * 2, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, self.output_dim),
                torch.nn.Sigmoid(),
            )

    def forward(self, layer_features):
        """聚合多层特征 → patch 向量。

        Args:
            layer_features: list of [B, C, H, W], [layer0, layer1, ...]
        Returns:
            patches: [B*H*W, output_dim]
        """
        if self.method == "neighborhood":
            return self._aggregate_neighborhood(layer_features)
        elif self.method == "channel_concat":
            return self._aggregate_channel_concat(layer_features)
        elif self.method == "fusion":
            return self._aggregate_fusion(layer_features)
        raise ValueError(f"Unknown method: {self.method}")

    def extra_repr(self):
        return (f"input_dim={self.input_dim}, num_layers={self.num_layers}, "
                f"output_dim={self.output_dim}, method='{self.method}'")

    # ─────────────────────────────────────────────────────────
    #  聚合策略 (新增方法写在这里)
    # ─────────────────────────────────────────────────────────

    def _aggregate_neighborhood(self, layer_features):
        """邻域聚合 + 跨层 learned pool (等价 _embed_legacy)。

        Unfold 3×3 → Align_dim → Stack → AdaptiveAvgPool1d
        """
        B, C, H, W = layer_features[0].shape
        target_dim = self.output_dim
        output_size = self.output_dim

        align_features = []
        for feat in layer_features:
            unfolded = self._patchify(feat)
            aligned = self._align_dim(unfolded, target_dim)
            align_features.append(aligned)

        # Stack layers → flatten → AdaptiveAvgPool1d
        stacked = torch.stack(align_features, dim=1)              # [B*N, L, D]
        stacked = stacked.reshape(stacked.shape[0], 1, -1)        # [B*N, 1, L*D]
        pooled = F.adaptive_avg_pool1d(stacked, output_size)      # [B*N, 1, D]
        return pooled.reshape(pooled.shape[0], -1)                # [B*N, D]

    def _aggregate_channel_concat(self, layer_features):
        """通道拼接 (等价 _embed_channel_concat)。

        无 patchify, 无 Align_dim, 无跨层 pool — 纯通道 concat。
        """
        concat = torch.cat(layer_features, dim=1)                # [B, D, H, W]
        B, C, H, W = concat.shape
        return concat.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, D]

    def _aggregate_fusion(self, layer_features):
        """门控融合: 将 neighborhood (全局/平滑) 和 channel_concat (局部/细节)
        通过可学习的 per-patch 门控进行自适应融合。

        gate = σ(MLP([F_neighbor; F_channel]))
        F_out = gate ⊙ F_neighbor + (1 - gate) ⊙ F_channel

        动机: neighborhood 类似低通滤波器，丢失细小异常但保留全局结构；
              channel_concat 保留所有细节但缺乏空间上下文。
              门控让模型在每个 patch 上自行决定信任哪个分支。
        """
        F_neighbor = self._aggregate_neighborhood(layer_features)    # [B*N, D]
        F_channel = self._aggregate_channel_concat(layer_features)   # [B*N, D]

        # 拼接两个视图作为门控输入
        gate_input = torch.cat([F_neighbor, F_channel], dim=-1)      # [B*N, 2D]
        gate = self.gate_mlp(gate_input)                             # [B*N, D]

        # 加权融合
        return gate * F_neighbor + (1 - gate) * F_channel            # [B*N, D]

    # ─────────────────────────────────────────────────────────
    #  共享工具 (供聚合策略内部调用)
    # ─────────────────────────────────────────────────────────

    def _patchify(self, feature):
        """Unfold 提取邻域 patch: [B,C,H,W] → [B,N,C,ps,ps]"""
        unfolded = self.unfolder(feature)
        return unfolded.reshape(
            *feature.shape[:2], self.patch_size, self.patch_size, -1
        ).permute(0, 4, 1, 2, 3)

    def _align_dim(self, feature, target_dim):
        """patch 展平 + AdaptiveAvgPool1d: [B,N,C,ps,ps] → [B*N, target_dim]"""
        _f = feature.reshape(-1, *feature.shape[2:])          # [B*N, C, ps, ps]
        _f = _f.reshape(len(_f), -1).unsqueeze(1)              # [B*N, 1, C*ps*ps]
        _f = F.adaptive_avg_pool1d(_f, target_dim)             # [B*N, 1, D]
        return _f.squeeze(1)                                   # [B*N, D]


# 复用组件--特征提取器
class FeatureExtractor(torch.nn.Module):
    """特征提取器 - 封装 DINOv2 和 FeatureAggregator 聚合"""

    def __init__(
        self,
        model_path: str,
        layer_indices: List[int],
        aggregator: FeatureAggregator = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.device = device

        # 聚合器: 可传入自定义实例, 默认用 NeighborhoodAggregation
        if aggregator is None:
            C = 384  # DINOv2 ViT-S/14 per-layer dim
            self.aggregator = FeatureAggregator(
                input_dim=C, num_layers=len(layer_indices), method="neighborhood",
            )
        else:
            self.aggregator = aggregator

        # 加载编码器 (向后兼容: export_onnx.py 直接访问 self.encoder)
        self.encoder = self._load_encoder(model_path).to(self.device)
        self.encoder.eval()

        # 聚合器也移到目标设备 (gate_mlp 等有参数模块需要)
        self.aggregator = self.aggregator.to(self.device)

        # 冻结参数
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _load_encoder(self, model_path: str):
        return download_dinov2_models(
            name='dinov2_vits14_reg',
            source='local',
            model_pth=model_path,
            pretrained=True
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Returns:
            patches_features: [B*H*W, C] 聚合后的特征
            (H, W): 特征图空间尺寸
        """
        layer_features = self._extract_layer_features(images)
        B, C, H, W = layer_features[0].shape
        patches_features = self.aggregator(layer_features)
        return patches_features, (H, W)

    def _extract_layer_features(self, image_tensor):
        """提取 DINOv2 指定中间层的特征 (List of [B, C, H, W])"""
        with torch.no_grad():
            return self.encoder.get_intermediate_layers(
                image_tensor,
                n=self.layer_indices,
                reshape=True,
                return_class_token=False,
            )

class PCAMaskGenerator:
    """
    PCA掩模生成器 - GPU/CPU混合版本
    使用PyTorch SVD实现GPU加速的PCA计算
    """
    
    def __init__(
        self,
        threshold: float = 10.0,
        border_ratio: float = 0.2,
        kernel_size: int = 3,
        use_gpu: bool = True,                 # 是否使用GPU加速
        skip_categories: List[str] = None,    # 指定跳过的类别列表（返回全1掩模）
    ):
        self.threshold = threshold
        self.border_ratio = border_ratio
        self.kernel_size = kernel_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.skip_categories = skip_categories or []
        self.current_category = None  # 当前处理的类别
        # SVD 缓存：fit 一次，后续只做投影
        self._pca_mean = None       # [D] 均值向量
        self._pca_component = None  # [D] 第一主成分方向
        self._pca_category = None   # 缓存对应的类别名 (避免跨类别混用)
    def set_category(self, category: str):
        """
        设置当前处理的类别

        Args:
            category: 类别名称，如 'screw', 'transistor' 等
        """
        if category != self.current_category:
            # 切换类别时清除 SVD 缓存 (新类别需要重新 fit)
            self._pca_mean = None
            self._pca_component = None
            self._pca_category = None
        self.current_category = category

    def __call__(
        self, 
        features: torch.Tensor, 
        grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        生成前景掩模
        
        Args:
            features: 特征张量 [B*H*W, C] 或 [H*W, C]
            grid_size: (H, W) 单张图像的网格尺寸
        Returns: bool tensor [B*H*W] 或 [H*W], True表示前景
        """
        H, W = grid_size
        num_patches = H * W
        
        # 检查当前类别是否在跳过列表中
        if self.current_category and self.current_category in self.skip_categories:
            # 直接返回全1掩模（所有像素都是前景），长度与 features 第一维匹配（支持 batch）
            return torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        
        # 处理 batch 情况
        if features.shape[0] > num_patches:
            B = features.shape[0] // num_patches
            # 重塑为 [B, N, C]
            features_batch = features.reshape(B, num_patches, -1)
            all_masks = []
            
            for i in range(B):
                mask = self.compute_background_mask(
                    features_batch[i], grid_size
                )
                all_masks.append(mask)
            
            return torch.cat(all_masks)
        else:
            return self.compute_background_mask(features, grid_size)
    
    def compute_background_mask(
        self,
        features: torch.Tensor,
        grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        计算背景掩模 - GPU/CPU混合版本

        Args:
            features: [N, C] 特征张量
            grid_size: (H, W) 网格尺寸
        Returns: bool tensor [N], True表示前景
        """
        H, W = grid_size
        device = features.device

        # 确保特征在正确的设备上
        if self.use_gpu and device.type == 'cuda':
            features_tensor = features
        else:
            features_tensor = features.cpu()

        # ---- SVD 路径：PC 值 → 阈值 → 中心检测 → 掩模 ----
        # 计算第一主成分 (首次 SVD + 缓存, 后续 O(N*D) 投影)
        first_pc = self._compute_first_pc_svd(features_tensor)

        # 生成初始掩模
        mask = first_pc > self.threshold

        # 自适应掩模：检查中心区域是否被保留
        mask_2d = mask.reshape(H, W) # reshape为2D格式，便于处理大小为37x37的掩模

        # 提取中心区域
        h_start, h_end = int(H * self.border_ratio), int(H * (1 - self.border_ratio))
        w_start, w_end = int(W * self.border_ratio), int(W * (1 - self.border_ratio))

        # 确保索引有效
        if h_start >= h_end or w_start >= w_end:
            # 如果border比例导致无效区域，直接使用整个图像
            center_mask = mask_2d
        else:
            center_mask = mask_2d[h_start:h_end, w_start:w_end]

        # 如果中心区域前景太少，反转掩模
        if center_mask.sum().item() <= center_mask.numel() * 0.35:
            mask = (-first_pc) > self.threshold
            mask_2d = mask.reshape(H, W)

        # 形态学后处理 (GPU-native)
        mask_processed = self._morphological_process(mask_2d)
        return mask_processed.flatten()

    def _compute_first_pc_svd(self, features: torch.Tensor) -> torch.Tensor:
        """通过 SVD 计算第一主成分投影值。

        首次调用时运行 SVD 并缓存方向向量；后续调用直接用缓存做投影 (O(N*D))，
        避免重复 SVD (O(N*D*min(N,D)))。
        """
        # 检查缓存是否命中 (同类别、同设备)
        if (self._pca_component is not None
                and self._pca_mean is not None
                and self._pca_category == self.current_category):
            cached_mean = self._pca_mean.to(features.device)
            cached_comp = self._pca_component.to(features.device)
            features_centered = features - cached_mean.unsqueeze(0)
            return features_centered @ cached_comp

        # 首次调用：完整 SVD，缓存方向向量
        mean = features.mean(dim=0, keepdim=True)
        features_centered = features - mean
        try:
            U, S, Vh = torch.linalg.svd(features_centered, full_matrices=False)
            self._pca_component = Vh[0, :].detach().cpu()   # [D] 缓存到 CPU
            self._pca_mean = mean.squeeze(0).detach().cpu()  # [D] 缓存到 CPU
            self._pca_category = self.current_category
            return features_centered @ self._pca_component.to(features.device)
        except RuntimeError:
            features_np = features.cpu().numpy()
            pca = PCA(n_components=1, svd_solver='randomized')
            pca.fit(features_np)
            first_pc_np = pca.components_[0]  # [D] 方向向量
            self._pca_component = torch.from_numpy(first_pc_np).float().cpu()
            self._pca_mean = torch.from_numpy(pca.mean_).float().cpu()
            self._pca_category = self.current_category
            return torch.from_numpy(
                pca.transform(features_np).squeeze()
            ).to(features.device)

    def _morphological_process(self, mask_2d: torch.Tensor) -> torch.Tensor:
        """
        形态学后处理 - 使用 PyTorch (GPU-native，避免 GPU↔CPU 数据传输)

        Args:
            mask_2d: [H, W] 二值掩模 (torch.Tensor, bool)
        Returns: [H, W] 处理后的二值掩模 (torch.Tensor, bool)
        """
        k = self.kernel_size
        # 先膨胀，扩大前景区域
        mask_dilated = _dilate_binary(mask_2d, k)
        # 再闭运算（膨胀+腐蚀），填充小孔
        return _close_binary(mask_dilated, k)
    
    def apply_mask(
        self, 
        features: torch.Tensor, 
        mask: torch.Tensor,
        device: str
    ) -> torch.Tensor:
        """
        应用掩模到特征
        
        Args:
            features: 特征张量 [N, C]
            mask: 布尔掩模 [N]
            device: 目标设备
        Returns: 掩模后的特征 [M, C]
        """
        return features[mask.to(device)]


# ---------------------------------------------------------------------------
# PyTorch-native 二值形态学操作（替代 OpenCV，避免 GPU↔CPU 数据传输）
# 原理：二值膨胀 = max pooling，二值腐蚀 = min pooling = -max_pool(-mask)
# ---------------------------------------------------------------------------

def _dilate_binary(mask_2d: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """二值膨胀 (GPU-native)

    Args:
        mask_2d: [H, W] 二值掩模
        kernel_size: 方形结构元素尺寸
    Returns:
        [H, W] 膨胀后的二值掩模
    """
    padding = kernel_size // 2
    x = mask_2d.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    return x.squeeze(0).squeeze(0) > 0.5


def _erode_binary(mask_2d: torch.Tensor, kernel_size: int,
                  iterations: int = 1) -> torch.Tensor:
    """二值腐蚀 (GPU-native)，支持多次迭代

    Args:
        mask_2d: [H, W] 二值掩模
        kernel_size: 方形结构元素尺寸
        iterations: 腐蚀迭代次数
    Returns:
        [H, W] 腐蚀后的二值掩模
    """
    for _ in range(iterations):
        padding = kernel_size // 2
        x = (-mask_2d.float()).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        x = -F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
        mask_2d = x.squeeze(0).squeeze(0) > 0.5
    return mask_2d


def _close_binary(mask_2d: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """二值闭运算 = 膨胀 + 腐蚀 (GPU-native)

    Args:
        mask_2d: [H, W] 二值掩模
        kernel_size: 方形结构元素尺寸
    Returns:
        [H, W] 闭运算后的二值掩模
    """
    mask_2d = _dilate_binary(mask_2d, kernel_size)
    return _erode_binary(mask_2d, kernel_size)


class PerlinMaskGenerator:
    """Perlin 噪声掩模生成器 — 在图像上生成多尺度随机 Perlin 噪声掩模"""

    def __init__(self, min_scale: int = 0, max_scale: int = 4):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img_shape, feat_size, mask_fg):
        """生成 Perlin 噪声二值掩模

        Args:
            img_shape: (C, H_img, W_img)   图像张量形状
            feat_size: int                  特征图尺寸 (H_feat == W_feat)
            mask_fg: np.ndarray [H_img, W_img]   前景约束 (0/1 或 float)
        Returns:
            np.ndarray [feat_size, feat_size]     Perlin 噪声掩模
        """
        mask = np.zeros((feat_size, feat_size))
        while np.max(mask) == 0:
            thr1 = self._generate_thr(img_shape)
            thr2 = self._generate_thr(img_shape)
            temp = torch.rand(1).numpy()[0]
            if temp > 2 / 3:
                perlin_thr = thr1 + thr2
                perlin_thr = np.where(perlin_thr > 0, np.ones_like(perlin_thr),
                                      np.zeros_like(perlin_thr))
            elif temp > 1 / 3:
                perlin_thr = thr1 * thr2
            else:
                perlin_thr = thr1
            perlin_thr = torch.from_numpy(perlin_thr) * mask_fg
            down_y = int(img_shape[1] / feat_size)
            down_x = int(img_shape[2] / feat_size)
            mask = (F.max_pool2d(
                perlin_thr.unsqueeze(0).unsqueeze(0),
                (down_y, down_x)).float().numpy()[0, 0])
        return mask

    def _generate_thr(self, img_shape):
        """单层 Perlin 噪声 + 旋转 + 阈值二值化"""
        perlin_scalex = 2 ** np.random.randint(self.min_scale, self.max_scale)
        perlin_scaley = 2 ** np.random.randint(self.min_scale, self.max_scale)
        perlin_noise = self._rand_perlin_2d_np(
            (img_shape[1], img_shape[2]),
            (perlin_scalex, perlin_scaley))
        angle = np.random.uniform(-90, 90)
        perlin_noise = ndimage.rotate(perlin_noise, angle, reshape=False, order=1)
        return np.where(perlin_noise > 0.5, np.ones_like(perlin_noise),
                        np.zeros_like(perlin_noise))

    @staticmethod
    def _rand_perlin_2d_np(shape, res,
                           fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
        """2D Perlin 噪声 numpy 实现"""
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (math.ceil(shape[0] / res[0]), math.ceil(shape[1] / res[1]))
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

        angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

        def tile_grads(sl1, sl2):
            return np.repeat(np.repeat(
                gradients[sl1[0]:sl1[1], sl2[0]:sl2[1]], d[0], axis=0), d[1], axis=1)

        def dot(grad, shift):
            return (np.stack(
                (grid[:shape[0], :shape[1], 0] + shift[0],
                 grid[:shape[0], :shape[1], 1] + shift[1]),
                axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[:shape[0], :shape[1]])
        return (math.sqrt(2) *
                PerlinMaskGenerator._lerp_np(
                    PerlinMaskGenerator._lerp_np(n00, n10, t[..., 0]),
                    PerlinMaskGenerator._lerp_np(n01, n11, t[..., 0]),
                    t[..., 1]))

    @staticmethod
    def _lerp_np(x, y, w):
        return (y - x) * w + x


# 投影器————将特征维度进行修改
class Projection(torch.nn.Module):
    """
    投影器--纯线性 MLP 构成
    Linear(1536, 1536) → Linear(1536, 1536)
    """
    def __init__(self, in_planes, out_planes=None, n_layers=1):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
        self.apply(init_weight)
    
    def forward(self, x):
        x = self.layers(x)
        return x

# 判别器--得到异常分数
class Discriminator(torch.nn.Module):
    """
    判别器--mlp构成,默认是两个mlp
    """
    def __init__(self, in_planes, n_layers=1, hidden=None):  # in_planes:输入特征维度
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden), # 全连接
                                     # torch.nn.BatchNorm1d(_hidden), # 批量归一化1d
                                     torch.nn.LeakyReLU(0.2) # 激活函数#

                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x, return_features=False):
        features = self.body(x) # n个mlp
        x = self.tail(features) # 最后一个全连接层，把特征维度转换为1
        if return_features:
            return x, features
        return x

# 训练过程类
class Trainer:
    """训练器 - 只负责训练逻辑"""
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor, # 特征提取器
        projection: torch.nn.Module, # 投影器
        discriminator: torch.nn.Module, # 判别器
        config: ModelConfig, # 模型参数类
        logger: Optional[logging.Logger] = None # 日志
    ):
        self.extractor = feature_extractor # 用一个提取特征
        self.projection = projection # 投影器
        self.discriminator = discriminator # 判别器
        self.config = config # 模型参数
        self.logger = logger or logging.getLogger(__name__) # 日志
        
        # PCA掩模生成器（可选）
        self.pca_generator = PCAMaskGenerator(
            threshold=config.pca_threshold,
            border_ratio=config.pca_border,
            kernel_size=config.pca_kernel_size,
            use_gpu=config.pca_use_gpu, # 使用GPU加速
            skip_categories=config.pca_skip_categories # 指定跳过的类别
        ) if config.use_pca_mask else None

        # 优化器
        # 注意: optimizer_proj 只管理 projection；discriminator 由 optimizer_dsc 独立管理
        self.optimizer_proj = torch.optim.AdamW(
            projection.parameters(),
            lr=config.proj_lr * 0.1,
            weight_decay=1e-5
        )
        self.optimizer_dsc = torch.optim.Adam(
            discriminator.parameters(),
            lr=config.dsc_lr,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        total_steps = config.gan_epochs * config.meta_epochs
        if config.use_scheduler:
            if config.scheduler_type == "multistep":
                # MultiStepLR (SuperSimpleNet 风格): 在指定比例处衰减 lr
                milestones = [int(total_steps * m) for m in config.multistep_milestones]
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer_dsc,
                    milestones=milestones,
                    gamma=config.multistep_gamma,
                )
                self.logger.info(
                    f"Scheduler: MultiStepLR (milestones={milestones}, gamma={config.multistep_gamma})"
                )
            else:
                # CosineAnnealingLR (默认)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_dsc,
                    T_max=total_steps,
                    eta_min=config.dsc_lr * 0.4,
                )
                self.logger.info(
                    f"Scheduler: CosineAnnealingLR (T_max={total_steps}, eta_min={config.dsc_lr * 0.4:.2e})"
                )
        else:
            self.scheduler = None
        
        self.global_step = 0
        self.current_meta_epoch = 0
        self.log_interval = 50  # 每隔多少步打印一次特征统计日志
        
    def train_epoch(
        self, 
        dataloader, # 数据加载器
    ) -> Dict[str, float]:
        """训练一个gan_epoch,也就是4个最小的epoch"""
        self.projection.train() # 进入训练模式
        self.discriminator.train() # 进入训练模式
        self.extractor.eval()  # 冻结特征提取器

        all_loss = []
        all_p_true = []
        all_p_fake = []
        
        self.current_meta_epoch += 1
        
        # 重置当前epoch的噪声统计
        for gan_epoch in range(self.config.gan_epochs):
            current_std = self._get_current_noise_std()
            if self.logger and gan_epoch == 0:
                self.logger.info(f"Meta Epoch {self.current_meta_epoch} 当前噪声强度 std={current_std:.4f}")

            pbar = tqdm(dataloader, desc=f"Meta {self.current_meta_epoch} GAN {gan_epoch+1}")

            for batch_idx, (images, _, _, _) in enumerate(pbar):
                loss, p_true, p_fake = self._train_step(images)

                all_loss.append(loss)
                all_p_true.append(p_true)
                all_p_fake.append(p_fake)

                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'p_t': f'{p_true:.3f}',
                    'p_f': f'{p_fake:.3f}'
                })

            if self.scheduler:
                self.scheduler.step()
        
        return {
            'loss': sum(all_loss) / len(all_loss),
            'p_true': sum(all_p_true) / len(all_p_true),
            'p_fake': sum(all_p_fake) / len(all_p_fake)
        }
    
    def _get_current_noise_std(self) -> float:
        """根据当前 meta epoch 计算噪声标准差"""
        if not getattr(self.config, 'use_noise_annealing', False):
            return self.config.noise_std
        
        max_std = getattr(self.config, 'noise_std_max', self.config.noise_std)
        min_std = getattr(self.config, 'noise_std_min', self.config.noise_std)
        total_epochs = getattr(self.config, 'noise_anneal_epochs', None) or self.config.meta_epochs
        total_epochs = max(total_epochs, 1)
        
        # 当前 epoch 索引 (0-based)，并限制在退火范围内
        epoch = max(0, self.current_meta_epoch - 1)
        epoch = min(epoch, total_epochs - 1) if total_epochs > 1 else 0
        ratio = epoch / max(total_epochs - 1, 1)
        
        anneal_type = getattr(self.config, 'noise_anneal_type', 'linear')
        if anneal_type == "linear":
            current = max_std - (max_std - min_std) * ratio
        elif anneal_type == "cosine":
            current = min_std + (max_std - min_std) * (1 + math.cos(math.pi * ratio)) / 2
        elif anneal_type == "exponential":
            current = max_std * ((min_std / max_std) ** ratio)
        else:
            current = self.config.noise_std
        
        return float(max(current, min_std))
    
    def _generate_perlin_masks(
        self,
        images: torch.Tensor,
        H: int,
        W: int,
        pca_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        为batch中的每个图像生成Perlin掩码，在PCA前景掩码基础上进一步限制噪声区域

        通过形态学腐蚀（erode）将PCA掩模向内收缩，使Perlin噪声更靠近前景中心，
        避免PCA边缘不精确时噪声覆盖到背景区域。

        Args:
            images: [B, C, target_size, target_size]
            H, W: 特征网格尺寸
            pca_mask: [B*H*W] bool tensor (PCA前景掩码)

        Returns:
            perlin_mask: [B*H*W] bool tensor
        """
        B = images.shape[0]
        target_size = images.shape[2]
        device = pca_mask.device

        if not hasattr(self, '_perlin_gen'):
            self._perlin_gen = PerlinMaskGenerator(
                min_scale=self.config.perlin_min,
                max_scale=self.config.perlin_max,
            )

        # 重塑PCA掩码为 [B, H, W]
        pca_mask_2d = pca_mask.reshape(B, H, W).float()

        all_masks = []
        for i in range(B):
            # 上采样PCA掩码到图像分辨率
            pca_mask_img = F.interpolate(
                pca_mask_2d[i:i+1].unsqueeze(0),  # [1, 1, H, W]
                size=(target_size, target_size),
                mode='nearest'
            ).squeeze()  # [target_size, target_size]

            # 如果该图像没有前景，直接返回全零掩码
            if pca_mask_img.sum().item() == 0:
                perlin_flat = np.zeros(H * W, dtype=bool)
                all_masks.append(torch.from_numpy(perlin_flat))
                continue

            # 用腐蚀（erode）收缩PCA掩模，让Perlin区域更靠近前景中心
            pca_mask_bool = pca_mask_img > 0.5
            pca_mask_eroded = _erode_binary(pca_mask_bool, kernel_size=5, iterations=2)

            # 如果腐蚀后前景没了，回退到原始PCA掩模
            if pca_mask_eroded.sum().item() == 0:
                pca_mask_eroded = pca_mask_bool

            # 生成Perlin掩码（PerlinMaskGenerator 需要 numpy 输入）
            try:
                perlin_s = self._perlin_gen(
                    img_shape=(images.shape[1], target_size, target_size),
                    feat_size=H,
                    mask_fg=pca_mask_eroded.cpu().numpy().astype(np.float32),
                )
                perlin_flat = (perlin_s > 0).flatten()
            except Exception as e:
                self.logger.warning(f"Perlin mask generation failed for image {i}: {e}")
                perlin_flat = np.ones(H * W, dtype=bool)

            all_masks.append(torch.from_numpy(perlin_flat))

        return torch.cat(all_masks).to(device)

    def _log_tensor_stats(self, name: str, tensor: torch.Tensor):
        """记录张量的统计信息"""
        if tensor.numel() == 0:
            self.logger.info(f"[{name}] shape={tuple(tensor.shape)}, EMPTY TENSOR")
            return
        self.logger.info(
            f"[{name}] shape={tuple(tensor.shape)}, "
            f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
            f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
        )

    def _train_step(self, images: torch.Tensor) -> Tuple[float, float, float]:
        """单步训练"""
        images = images.to(self.config.device)
        
        # 提取特征
        features, (H, W) = self.extractor(images)
        
        # 诊断：检查提取后的特征
        if features.numel() == 0:
            self.logger.error(f"[DIAG] Extractor returned EMPTY features! images shape={images.shape}")
        
        # PCA掩模（可选）
        pca_mask = None
        if self.pca_generator:
            pca_mask = self.pca_generator(features, (H, W))
            pca_ratio = pca_mask.float().mean().item()
            if pca_ratio == 0.0:
                self.logger.warning(f"[DIAG] PCA mask ratio is 0.0! All patches filtered. features shape before mask={features.shape}. Falling back to all-ones mask.")
                pca_mask = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
            features = self.pca_generator.apply_mask(
                features, pca_mask, self.config.device
            )
            if features.numel() == 0:
                self.logger.error(f"[DIAG] After PCA mask, features is EMPTY! pca_mask sum={pca_mask.sum().item()}")
        
        # 投影--只接受PCA前景patch的特征进行投影和对抗训练
        projected = self.projection(features) # 
        
        if projected.numel() == 0:
            self.logger.error(f"[DIAG] Projected is EMPTY! features shape={features.shape if features.numel() > 0 else 'EMPTY'}. Skipping this batch.")
            return 0.0, 0.0, 0.0
        
        current_std = self._get_current_noise_std()

        # ==================== PCA掩模训练模式 ====================
        if pca_mask is not None:
            if self.config.use_perlin_mask:
                # ==================== 双分支训练 (Perlin BCE + PCA Hinge) ====================
                # ---- 分支1: Perlin定位分支 (BCE Loss) ----
                # 在PCA基础上生成Perlin掩码，用于定位噪声位置
                perlin_mask_tensor = self._generate_perlin_masks(images, H, W, pca_mask)
                perlin_mask = pca_mask & perlin_mask_tensor  # Perlin噪声只在PCA前景内的Perlin区域
                # 获取Perlin区域的投影特征
                pca_indices = torch.nonzero(pca_mask, as_tuple=True)[0]
                perlin_indices = torch.nonzero(perlin_mask, as_tuple=True)[0]
                is_perlin = torch.isin(pca_indices, perlin_indices)
                projected_perlin = projected[is_perlin] if perlin_indices.numel() > 0 else projected

                if projected_perlin.size(0) == 0:
                    projected_perlin = projected

                # 在Perlin区域加噪--高斯混合噪声作为基础
                noise_perlin = torch.normal(0, current_std, projected_perlin.shape, device=projected_perlin.device)
                fake_perlin = projected_perlin + noise_perlin

                # 构建Perlin分支的输入: 全部真实特征 + Perlin区域假特征
                scores_perlin = self.discriminator(
                    torch.cat([projected, fake_perlin], dim=0)
                )
                true_scores_perlin = scores_perlin[:len(projected)]
                fake_scores_perlin = scores_perlin[len(projected):]

                # BCE损失: 将判别器输出sigmoid后作为"是真实特征"的概率
                # true_labels=1 (真实), fake_labels=0 (假/噪声)
                true_labels = torch.ones_like(true_scores_perlin)
                fake_labels = torch.zeros_like(fake_scores_perlin)

                bce_loss = F.binary_cross_entropy_with_logits(
                    true_scores_perlin, true_labels, reduction='mean'
                ) + F.binary_cross_entropy_with_logits(
                    fake_scores_perlin, fake_labels, reduction='mean'
                )

                # ---- 分支2: PCA对抗分支 (Hinge Loss) ----
                # 在整个PCA前景patch上施加标准对抗训练，不使用Perlin限制
                noise_pca = torch.normal(0, current_std, projected.shape, device=projected.device)
                fake_pca = projected + noise_pca

                # 构建PCA分支的输入: 全部真实特征 + 全部假特征
                scores_pca = self.discriminator(
                    torch.cat([projected, fake_pca], dim=0)
                )
                true_scores_pca = scores_pca[:len(projected)]
                fake_scores_pca = scores_pca[len(projected):]

                # 非对称Hinge: 真实 > 1.0, 假 < 0.0 (与BCE目标对齐)
                true_loss_pca = torch.clip(-true_scores_pca + 1.0, min=0).mean()
                fake_loss_pca = torch.clip(fake_scores_pca, min=0).mean()
                hinge_loss = true_loss_pca + fake_loss_pca

                # ---- 合并损失 ----
                w_perlin = getattr(self.config, 'perlin_branch_weight', 1.0)
                w_pca = getattr(self.config, 'pca_branch_weight', 1.0)
                loss = w_perlin * bce_loss + w_pca * hinge_loss

                # 反向传播
                self.optimizer_proj.zero_grad()
                self.optimizer_dsc.zero_grad()
                loss.backward()
                self.optimizer_proj.step()
                self.optimizer_dsc.step()
                # -------------------
                # 计算指标
                with torch.no_grad():
                    p_true = (true_scores_pca >= 1.0).float().mean().item()
                    p_fake = (fake_scores_pca < 0.0).float().mean().item()

                    # 日志
                    if self.global_step % self.log_interval == 0:
                        self.logger.info(f"--- Step {self.global_step} 双分支训练 (noise_std={current_std:.4f}, noise=gaussian) ---")
                        self.logger.info(f"Perlin分支 - BCE: {bce_loss.item():.4f}, mask_ratio: {perlin_mask.float().mean().item():.3f}")
                        self.logger.info(f"PCA分支   - Hinge: {hinge_loss.item():.4f}, true_loss: {true_loss_pca.item():.4f}, fake_loss: {fake_loss_pca.item():.4f}")
                        self.logger.info(f"合并损失  - total: {loss.item():.4f} (w_perlin={w_perlin}, w_pca={w_pca})")
                        self.logger.info(f"下面分别是：投影特征、Perlin分支假特征、PCA分支假特征的统计信息：")
                        self._log_tensor_stats("Projected", projected)
                        self._log_tensor_stats("PerlinFake", fake_perlin)
                        self._log_tensor_stats("PCAFake", fake_pca)
                        self.logger.info("-" * 60)
                self.global_step += 1
                return loss.item(), p_true, p_fake
            else:
                # ==================== 单分支PCA训练 (PCA Hinge only, 无Perlin) ====================
                # 仅在PCA前景patch上进行标准对抗训练
                noise = torch.normal(0, current_std, projected.shape, device=projected.device)
                fake = projected + noise

                scores = self.discriminator(torch.cat([projected, fake], dim=0))
                true_scores = scores[:len(projected)]
                fake_scores = scores[len(projected):]

                true_loss = torch.clip(-true_scores + 1.0, min=0).mean()
                fake_loss = torch.clip(fake_scores, min=0).mean()
                loss = true_loss + fake_loss

                self.optimizer_proj.zero_grad()
                self.optimizer_dsc.zero_grad()
                loss.backward()
                self.optimizer_proj.step()
                self.optimizer_dsc.step()

                with torch.no_grad():
                    p_true = (true_scores >= 1.0).float().mean().item()
                    p_fake = (fake_scores < 0.0).float().mean().item()

                    if self.global_step % self.log_interval == 0:
                        self.logger.info(f"--- Step {self.global_step} 单分支PCA训练 (noise_std={current_std:.4f}, noise=gaussian) ---")
                        self.logger.info(f"Hinge Loss: {loss.item():.4f}, true_loss: {true_loss.item():.4f}, fake_loss: {fake_loss.item():.4f}")
                        self._log_tensor_stats("Projected", projected)
                        self._log_tensor_stats("Fake", fake)
                        self.logger.info("-" * 60)
                self.global_step += 1
                return loss.item(), p_true, p_fake
        else:
            # ---- 单分支训练 (无PCA掩码，标准Hinge Loss) ----
            noise = torch.normal(0, current_std, projected.shape, device=projected.device)
            fake = projected + noise

            scores = self.discriminator(torch.cat([projected, fake], dim=0))
            true_scores = scores[:len(projected)]
            fake_scores = scores[len(projected):]

            true_loss = torch.clip(-true_scores + 1.0, min=0).mean()
            fake_loss = torch.clip(fake_scores, min=0).mean()
            loss = true_loss + fake_loss

            self.optimizer_proj.zero_grad()
            self.optimizer_dsc.zero_grad()
            loss.backward()
            self.optimizer_proj.step()
            self.optimizer_dsc.step()

            with torch.no_grad():
                p_true = (true_scores >= 1.0).float().mean().item()
                p_fake = (fake_scores < 0.0).float().mean().item()

                if self.global_step % self.log_interval == 0:
                    self.logger.info(f"--- Step {self.global_step} 单分支训练 (noise_std={current_std:.4f}, noise=gaussian) ---")
                    self.logger.info(f"Hinge Loss: {loss.item():.4f}, true_loss: {true_loss.item():.4f}, fake_loss: {fake_loss.item():.4f}")
                    self._log_tensor_stats("Projected", projected)
                    self._log_tensor_stats("Fake", fake)
                    self.logger.info("-" * 60)
            self.global_step += 1
            return loss.item(), p_true, p_fake
    
class Predictor:
    """预测器 - 只负责推理逻辑"""
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,  # 特征提取器
        projection: torch.nn.Module,#  投影器
        discriminator: torch.nn.Module, # 判别器
        config: ModelConfig, # 模型配置层
        logger: Optional[logging.Logger] = None
    ):
        self.extractor = feature_extractor
        self.projection = projection
        self.discriminator = discriminator
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        # PCA掩模
        self.pca_generator = PCAMaskGenerator(
            threshold=config.pca_threshold,
            border_ratio=config.pca_border,
            kernel_size=config.pca_kernel_size,
            use_gpu=config.pca_use_gpu, # 使用GPU加速
            skip_categories=config.pca_skip_categories # 指定跳过的类别
        ) if config.use_pca_mask else None

        # 预创建高斯模糊算子，避免推理时逐图 cv2.GaussianBlur CPU 往返
        # kernel_size=25 等价于 cv2.GaussianBlur(..., (0,0), sigmaX=4) 的自动计算
        self.blur = GaussianBlur(kernel_size=25, sigma=4)

        self._eval_mode()
    
    def _eval_mode(self):
        """设置为评估模式"""
        self.extractor.eval()
        self.projection.eval()
        self.discriminator.eval()
        
    @torch.no_grad()
    def predict(
        self, 
        dataloader, # 数据加载器
        aggregation: str = "max"  # "max" or "topk"
    ) -> Tuple[List[float], List[np.ndarray], List, List]:
        """
        预测异常分数
        
        Returns:
            image_scores: 图像级异常分数
            masks: 像素级异常掩码
            labels_gt: 真实标签
            masks_gt: 真实掩码
        """
        all_scores = []
        all_masks = []
        all_labels = []
        all_masks_gt = []
        
        for images, masks_gt, labels, _ in tqdm(dataloader, desc="Predicting"):
            batch_size = images.shape[0]
            images = images.to(self.config.device)
            
            # 提取特征
            features, (H, W) = self.extractor(images)
            
            # PCA掩模
            mask_tensor = None
            if self.pca_generator:
                mask_tensor = self.pca_generator(features, (H, W))
                features_masked = self.pca_generator.apply_mask(
                    features, mask_tensor, self.config.device
                )
            else:
                features_masked = features
            
            # 投影和判别
            projected = self.projection(features_masked)
            patch_scores = -self.discriminator(projected)
            
            # 还原完整特征图（如果用了PCA）-- 因为背景部分的分数没有计算也就是丢掉了这部分patch，所以用最小分数填充
            if self.pca_generator and mask_tensor is not None:
                full_scores = torch.ones(batch_size * H * W, 1, device=self.config.device)
                full_scores *= patch_scores.min()
                full_scores[mask_tensor] = patch_scores
                patch_scores = full_scores
            
            # 重塑为图像形式 (保留在 GPU)
            patch_scores = patch_scores.reshape(batch_size, H, W)   # [B, H, W]

            # 上采样到目标尺寸 (全程 GPU)
            masks = self._upsample_masks(patch_scores)

            # 计算图像级分数 (GPU)
            if aggregation == "max":
                img_scores = patch_scores.reshape(batch_size, -1).max(dim=1).values
            elif aggregation == "topk":
                flat = patch_scores.reshape(batch_size, -1)
                img_scores = flat.topk(k=10, dim=1).values.mean(dim=1)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            # 批量转 CPU (唯一一次)
            all_scores.extend(img_scores.cpu().tolist())
            all_masks.extend([m for m in masks.cpu().numpy()])
            all_labels.extend(labels.numpy().tolist())
            all_masks_gt.extend(masks_gt.numpy().tolist())
        
        return all_scores, all_masks, all_labels, all_masks_gt
    
    def _upsample_masks(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """上采样到目标尺寸 + GPU 批量高斯平滑。

        Args:
            patch_scores: [B, H, W] GPU tensor
        Returns:
            [B, 1, target_size, target_size] GPU tensor
        """
        scores_tensor = patch_scores.unsqueeze(1).float()  # [B, 1, H, W]

        upsampled = F.interpolate(
            scores_tensor,
            size=(self.config.target_size, self.config.target_size),
            mode='bilinear',
            align_corners=False
        )  # [B, 1, target_size, target_size]

        return self.blur(upsampled)  # [B, 1, target_size, target_size]
    
# 主要代码层
class DINOv2AnomalyDetector:
    """
    简化后的主类 - 只负责 orchestration
    """
    
    def __init__(
        self,
        model_path: str, 
        config: Optional[ModelConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or ModelConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = model_path  # 保存路径以便 load 时重建组件
        
        # 初始化组件
        # 特征提取器 - 封装 DINOv2 和 FeatureAggregator
        agg_type = getattr(self.config, 'aggregation_type', 'neighborhood')
        C_input = self.config.input_planes // len(self.config.layer_indices)  # 384
        aggregator = FeatureAggregator(
            input_dim=C_input,
            num_layers=len(self.config.layer_indices),
            method=agg_type,
            patch_size=self.config.patch_size,
        )
        self.feature_extractor = FeatureExtractor(
            model_path=model_path,
            layer_indices=self.config.layer_indices,
            aggregator=aggregator,
            device=self.config.device,
        )
        

        # 投影器和判别器的输入维度是特征维度（C * n_layers）
        self.projection = Projection(
            in_planes= self.config.input_planes, # 特征维度,
            n_layers=2,
        ).to(self.config.device) # cuda
        # 判别器的输入维度也是特征维度，输出是1维异常分数
        self.discriminator = Discriminator(
            in_planes=self.config.hidden_dim, # 特征维度
            n_layers=2,
            hidden=self.config.hidden_dim
        ).to(self.config.device) # cuda
        
        # 初始化和预测器
        self.trainer = None # 训练器实例化放在fit方法中，避免不必要的资源占用
        self.predictor = None # 预测器实例化放在predict方法中，避免不必要的资源占用
        self.current_category = None # 当前处理的类别，用于PCA掩模的类别特定控制
        self._pca_mean = None  # checkpoint 恢复的 PCA SVD 参数 (CPU)
        self._pca_component = None

        # 记录初始化信息
        self._log_init()
    
    def _log_init(self):
        """记录初始化信息"""
        self.logger.info("=" * 60)
        self.logger.info("DINOv2 Anomaly Detector Initialization")
        self.logger.info("=" * 60)
        for key, value in vars(self.config).items():
            self.logger.info(f"  {key}: {value}")
        
        # 检查模型参数所在设备
        encoder_device = next(self.feature_extractor.encoder.parameters()).device
        proj_device = next(self.projection.parameters()).device
        dsc_device = next(self.discriminator.parameters()).device
        self.logger.info(f"  Encoder device: {encoder_device}")
        self.logger.info(f"  Projection device: {proj_device}")
        self.logger.info(f"  Discriminator device: {dsc_device}")
        self.logger.info("=" * 60)
    
    def set_category(self, category: str):
        """
        设置当前处理的类别，用于PCA掩模的类别特定控制
        
        Args:
            category: 类别名称，如 'screw', 'transistor' 等
        """
        self.current_category = category
        if self.trainer and self.trainer.pca_generator:
            self.trainer.pca_generator.set_category(category)
        if self.predictor and self.predictor.pca_generator:
            self.predictor.pca_generator.set_category(category)
        self.logger.info(f"Set current category: {category}")

    def fit(self, train_dataloader) -> Dict[str, float]:
        """训练一个 meta epoch"""
        if self.trainer is None:
            self.trainer = Trainer(
                self.feature_extractor, self.projection,
                self.discriminator, self.config, self.logger
            )
            if self.current_category is not None and self.trainer.pca_generator:
                self.trainer.pca_generator.set_category(self.current_category)
                self._restore_pca(self.trainer.pca_generator)
        return self.trainer.train_epoch(train_dataloader)
    
    def predict(
        self,
        test_dataloader,
        aggregation: str = "max"
    ) -> Tuple[List[float], List[np.ndarray], List, List]:
        """预测异常"""
        if self.predictor is None:
            self.predictor = Predictor(
                self.feature_extractor,
                self.projection,
                self.discriminator,
                self.config,
                self.logger
            )
            if self.current_category is not None and self.predictor.pca_generator:
                self.predictor.pca_generator.set_category(self.current_category)
                self._restore_pca(self.predictor.pca_generator)

        return self.predictor.predict(test_dataloader, aggregation)
    
    def save(self, path: str, epoch: int = 0, scores: dict = None):
        """保存模型权重（Projection + Discriminator + 聚合器 gate_mlp + PCA SVD 参数）"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            'proj_state': self.projection.state_dict(),
            'dsc_state': self.discriminator.state_dict(),
            # fusion 聚合的门控 MLP 虽不训练, 但必须持久化:
            # 否则每次重建 detector 都会得到新的随机初始化, 推理/导出无法复现
            'agg_state': self.feature_extractor.aggregator.state_dict(),
            'epoch': epoch,
            'scores': scores,
        }
        # PCA SVD 参数 (训练第一个 batch 的聚合特征求得, 持久化后推理无需重算)
        pca = None
        if self.trainer is not None and self.trainer.pca_generator is not None:
            pca = self.trainer.pca_generator
        elif self.predictor is not None and self.predictor.pca_generator is not None:
            pca = self.predictor.pca_generator
        if pca is not None:
            state['pca_mean'] = pca._pca_mean        # CPU tensor; skip 类别为 None
            state['pca_component'] = pca._pca_component
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load(self, path: str):
        """加载模型权重，清空 Trainer/Predictor 使其使用新权重重建"""
        state = torch.load(path, map_location=self.config.device)
        self.projection.load_state_dict(state['proj_state'])
        self.discriminator.load_state_dict(state['dsc_state'])
        # 兼容旧 checkpoint (无 agg_state): 保持随机初始化 gate_mlp
        if 'agg_state' in state:
            self.feature_extractor.aggregator.load_state_dict(state['agg_state'])
        else:
            self.logger.warning(
                "Checkpoint has no agg_state (old format); gate_mlp stays "
                "randomly initialized and will differ from the training run"
            )
        # PCA SVD 参数: 重建 Trainer/Predictor 时注入
        self._pca_mean = state.get('pca_mean')
        self._pca_component = state.get('pca_component')
        self.trainer = None
        self.predictor = None
        self.logger.info(f"Checkpoint loaded from {path}")
        return state.get('epoch', 0), state.get('scores', None), None, -1

    def _restore_pca(self, pca_generator):
        """把 checkpoint 里的 PCA SVD 参数注入到新创建的 pca_generator。

        必须在 set_category() 之后调用 (set_category 会清空 SVD 缓存)。
        skip 类别 (缓存为 None) 不注入, 推理时直接返回全 1 掩码。
        """
        if pca_generator is not None and self._pca_component is not None:
            pca_generator._pca_mean = self._pca_mean.to(self.config.device)
            pca_generator._pca_component = self._pca_component.to(self.config.device)
            pca_generator._pca_category = self.current_category
    
    def evaluate(
                self,
                scores: List[float],
                segmentations: List[np.ndarray],
                labels_gt: List,
                masks_gt: List[np.ndarray],
                compute_full_metrics: bool = False
                ) -> Dict[str, float]:
        """
        评估性能 - 交叉归一化集成(Cross-Normalization Ensemble)
        
        对每张图像，使用数据集中每个样本的 min/max 进行归一化，
        然后将所有归一化结果累加求平均，得到最终的归一化分数。
        
        Args:
            compute_full_metrics: 若为 False,仅计算 AUROC(训练时快速评估):
                                  若为 True,计算全部指标(AP、F1、PRO 等)。
        """
        # ========== 图像级 AUROC（逐图 min-max 归一化，与 SimpleNet 对齐）==========
        scores_arr = np.squeeze(np.array(scores))
        img_min_scores = scores_arr.min(axis=-1)
        img_max_scores = scores_arr.max(axis=-1)
        scores_norm = (scores_arr - img_min_scores) / (img_max_scores - img_min_scores)
        
        img_metrics = compute_imagewise_retrieval_metrics(scores_norm, labels_gt)
        
        # 快速模式：只返回 AUROC
        if not compute_full_metrics:
            if len(masks_gt) > 0:
                seg_arr = np.array(segmentations)  # [B, H, W]
                seg_mins = (
                seg_arr.reshape(len(seg_arr), -1) # (83, 288* 288)
                .min(axis=-1)# (83,1)
                .reshape(-1, 1, 1, 1) # (83,1,1,1)
                )
                seg_maxs = (
                seg_arr.reshape(len(seg_arr), -1)
                .max(axis=-1)# (83,1)
                .reshape(-1, 1, 1, 1)
                ) # (83,1,1,1)
                ranges = np.maximum(seg_maxs - seg_mins, 1e-2)
                seg_norm = (seg_arr * (1.0 / ranges).sum() - (seg_mins / ranges).sum()) / len(segmentations)
                pixel_auroc = _safe_roc_auc(
                    np.array(masks_gt).ravel().astype(int), seg_norm.ravel()
                )
                return {'image_auroc': img_metrics['auroc'], 'pixel_auroc': pixel_auroc}
            return {'image_auroc': img_metrics['auroc'], 'pixel_auroc': -1}
        
        # ========== 完整模式：计算全部指标 ==========
        if len(masks_gt) > 0:
            seg_arr = np.array(segmentations)  # [B, H, W]
            seg_mins = (
                seg_arr.reshape(len(seg_arr), -1) # (83, 288* 288)
                .min(axis=-1)# (83,1)
                .reshape(-1, 1, 1, 1) # (83,1,1,1)
                )
            seg_maxs = (
                seg_arr.reshape(len(seg_arr), -1)
                .max(axis=-1)# (83,1)
                .reshape(-1, 1, 1, 1)
                ) # (83,1,1,1)
            
            # 交叉归一化：用每个样本的 min/max 归一化所有图像，累加后平均
            # 注：该操作对 AUROC/AP/F1 有利，但会破坏阈值型指标(PRO)所需的逐图动态范围
            ranges = np.maximum(seg_maxs - seg_mins, 1e-2)
            seg_norm = (seg_arr * (1.0 / ranges).sum() - (seg_mins / ranges).sum()) / len(segmentations)
            
            # 为 PRO 单独计算逐图 min-max 归一化（保留每张图自身的对比度）
            pixel_metrics = compute_pixelwise_retrieval_metrics(seg_norm, masks_gt)
           
            
            return {
                'image_auroc': img_metrics['auroc'],
                'image_ap': img_metrics['ap'],
                'image_f1': img_metrics['f1'],
                'pixel_auroc': pixel_metrics['auroc'],
                'pixel_ap': pixel_metrics['ap'],
                'pixel_f1': pixel_metrics['f1'],
                'pixel_pro': pixel_metrics['pro']
            }
        
        return {
            'image_auroc': img_metrics['auroc'],
            'image_ap': img_metrics['ap'],
            'image_f1': img_metrics['f1'],
            'pixel_auroc': -1,
            'pixel_ap': -1,
            'pixel_f1': -1,
            'pixel_pro': -1
        }


