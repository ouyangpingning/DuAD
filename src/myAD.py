from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from commen_import import *
from utils import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, _embed_legacy, init_weight, download_dinov2_models, _safe_roc_auc
from sklearn.decomposition import PCA
import cv2

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

    # 数据增强控制
    augment_categories: List[str] = None  # 指定使用图像级数据增强的类别列表（少样本模式下生效）

    # 颜色数据增强控制（解决 toothbrush 等多颜色类别的少样本颜色偏差问题）
    color_augment_categories: List[str] = None  # 指定使用颜色增强的类别列表，如 ["toothbrush"]

    # Perlin掩模参数
    use_perlin_mask: bool = False      # 是否使用Perlin掩模在PCA基础上进一步限制噪声位置
    perlin_min: int = 0                # Perlin噪声最小尺度
    perlin_max: int = 4                # Perlin噪声最大尺度

    # 双分支损失参数
    perlin_branch_weight: float = 1.0  # Perlin分支BCE损失的权重
    pca_branch_weight: float = 1.0     # PCA分支Hinge损失的权重

    # PCA Student 参数
    use_pca_student: bool = False       # 是否使用 PCA Student 替代 SVD
    pca_student_hidden_dims: List[int] = None  # MLP 隐藏层维度
    pca_student_lr: float = 0.001       # Adam 学习率
    pca_student_epochs: int = 50        # 训练轮数
    pca_student_batch_size: int = 4096  # mini-batch 大小 (patch 级)

    # 其他
    patch_size: int = 3
    use_scheduler: bool = True
    device: str = "cuda"
    
    def __post_init__(self):
        if self.layer_indices is None:
            self.layer_indices = [2, 5, 8, 11]
        if self.pca_student_hidden_dims is None:
            self.pca_student_hidden_dims = [512, 128]

# 复用组件--特征提取器
class FeatureExtractor(torch.nn.Module):
    """特征提取器 - 封装 DINOv2 和特征聚合"""
    
    def __init__(
        self,
        model_path: str, # 模型的路径
        layer_indices: List[int], # 要使用的对应层
        patch_size: int = 3, # 每个patch块的分辨率
        device: str = "cuda", # 使用的设备
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.patch_size = patch_size
        self.device = device

        # 加载编码器
        self.encoder = self._load_encoder(model_path).to(self.device)
        self.encoder.eval()

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
    
    # 前向传播过程--dinov2提取特征 + _embed_legacy特征聚合
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        返回: (patches_features, (H, W))
        patches_features: [B*H*W, C] 聚合后的特征
        """
        # 提取多层特征
        layer_features = self._extract_layer_features(images)

        B, C, H, W = layer_features[0].shape

        # 聚合多层特征
        features_dict = {idx: feat for idx, feat in enumerate(layer_features)}
        patches_features, _ = _embed_legacy(
            features_dict,
            layers=list(range(len(self.layer_indices))),
            patchsize=self.patch_size,
            stride=1,
            target_patches=H * W,
            target_dim=C * len(self.layer_indices), # 384 * 4 = 1536
            output_size=C * len(self.layer_indices) # 384 * 4 = 1536
        )  # [B*H*W, C]

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
        pca_student = None,                    # PCAStudent 实例，替代 SVD
    ):
        self.threshold = threshold
        self.border_ratio = border_ratio
        self.kernel_size = kernel_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.skip_categories = skip_categories or []
        self.current_category = None  # 当前处理的类别
        self.pca_student = pca_student  # PCA Student 模型（可选）
    def set_category(self, category: str):
        """
        设置当前处理的类别
        
        Args:
            category: 类别名称，如 'screw', 'transistor' 等
        """
        self.current_category = category

    def set_pca_student(self, student):
        """设置或清除 PCA Student 模型"""
        self.pca_student = student

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

        # ---- PCA Student 路径：概率 → 0.5 阈值 → 形态学 ----
        if self.pca_student is not None:
            probs = self._compute_first_pc_torch(features_tensor)  # [0, 1]
            mask = probs > 0.5
            mask_2d = mask.reshape(H, W)
            mask_np = mask_2d.cpu().numpy() if mask_2d.is_cuda else mask_2d.numpy()
            mask_processed = self._morphological_process(mask_np)
            return torch.from_numpy(mask_processed.flatten()).to(device)

        # ---- SVD 路径：PC 值 → 阈值 → 中心检测 → 掩模 ----
        # 计算第一主成分（GPU/CPU自动选择）
        first_pc = self._compute_first_pc_torch(features_tensor)

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

        # 形态学后处理（需要转到CPU，因为OpenCV不支持GPU）
        mask_np = mask_2d.cpu().numpy() if mask_2d.is_cuda else mask_2d.numpy()
        mask_processed = self._morphological_process(mask_np)
        return torch.from_numpy(mask_processed.flatten()).to(device)

    def _compute_first_pc_torch(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算前景概率 (PCA Student) 或 PC 投影值 (SVD 回退)

        PCA Student 路径: 输出 sigmoid 概率 [0, 1]
        SVD 回退路径: 输出 PC 投影值 (无界标量)
        """
        if self.pca_student is not None:
            self.pca_student.eval()
            with torch.no_grad():
                return torch.sigmoid(self.pca_student(features).squeeze(-1))

        return self._compute_first_pc_svd(features)

    @staticmethod
    def _compute_first_pc_svd(features: torch.Tensor) -> torch.Tensor:
        """通过 SVD 计算第一主成分投影值"""
        mean = features.mean(dim=0, keepdim=True)
        features_centered = features - mean
        try:
            U, S, Vh = torch.linalg.svd(features_centered, full_matrices=False)
            first_component = Vh[0, :]
            return features_centered @ first_component
        except RuntimeError:
            features_np = features.cpu().numpy()
            pca = PCA(n_components=1, svd_solver='randomized')
            first_pc_np = pca.fit_transform(features_np).squeeze()
            return torch.from_numpy(first_pc_np).to(features.device)

    def _morphological_process(self, mask_2d: np.ndarray) -> np.ndarray:
        """
        形态学后处理 - 使用OpenCV
        
        Args:
            mask_2d: [H, W] 二值掩模
        Returns: [H, W] 处理后的二值掩模
        """
        # 转换为uint8 (0/1)
        mask_uint8 = mask_2d.astype(np.uint8)
        
        # 创建结构元素
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # 先膨胀，扩大前景区域
        mask_dilated = cv2.dilate(mask_uint8, kernel)
        
        # 再闭运算（膨胀+腐蚀），填充小孔
        mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)
        
        # 转换回bool
        return mask_closed.astype(bool)
    
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


class PCAStudent(torch.nn.Module):
    """
    PCA Student — MLP 从 DINOv2 特征预测二值前景掩模

    架构: Linear(input_dim, H1) -> ReLU -> Linear(H1, H2) -> ReLU -> Linear(H2, 1)
    输出 raw logits，用 BCEWithLogitsLoss 训练，推理时经 sigmoid 转前景概率。
    """

    def __init__(self, input_dim: int = 1536, hidden_dims: list = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [512, 128]

        layers = []
        prev_dim = input_dim
        for h in self.hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU(inplace=True))
            prev_dim = h
        layers.append(torch.nn.Linear(prev_dim, 1))
        self.net = torch.nn.Sequential(*layers)
        self.apply(init_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回 raw logits [N, 1]"""
        return self.net(features)


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

        # 收集需要训练的参数：投影器 + 判别器
        trainable_params = list(projection.parameters()) + list(discriminator.parameters())
        
        # 优化器
        self.optimizer_proj = torch.optim.AdamW(
            trainable_params, 
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dsc,
            T_max=total_steps,
            eta_min=config.dsc_lr * 0.4
        ) if config.use_scheduler else None
        
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
        from perlin import perlin_mask
        
        B = images.shape[0]
        target_size = images.shape[2]
        device = pca_mask.device
        
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
            
            pca_mask_np = pca_mask_img.cpu().numpy().astype(np.uint8)
            
            # 如果该图像没有前景，直接返回全零掩码
            if pca_mask_np.sum() == 0:
                perlin_flat = np.zeros(H * W, dtype=bool)
                all_masks.append(torch.from_numpy(perlin_flat))
                continue
            
            # 用腐蚀（erode）收缩PCA掩模，让Perlin区域更靠近前景中心
            kernel = np.ones((5, 5), np.uint8)
            pca_mask_eroded = cv2.erode(pca_mask_np, kernel, iterations=2)
            
            # 如果腐蚀后前景没了，回退到原始PCA掩模
            if pca_mask_eroded.sum() == 0:
                pca_mask_eroded = pca_mask_np
            
            # 生成Perlin掩码
            try:
                perlin_s = perlin_mask(
                    img_shape=(images.shape[1], target_size, target_size),
                    feat_size=H,
                    min=self.config.perlin_min,
                    max=self.config.perlin_max,
                    mask_fg=pca_mask_eroded.astype(np.float32),
                    flag=0
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

        # ==================== 双分支训练模式 ====================
        if pca_mask is not None:
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
    
    def train_pca_student(self, train_dataloader) -> None:
        """
        在 GAN 训练之前训练 PCA Student。
        Phase 1: 用 SVD 生成的二值掩模作为 ground truth
        Phase 2: 用 BCEWithLogitsLoss 训练 MLP
        训练完成后自动挂接到 self.pca_generator。
        """
        self.logger.info("=" * 60)
        self.logger.info("Training PCA Student (MLP, BCE)...")
        self.logger.info(f"  hidden_dims={self.config.pca_student_hidden_dims}, "
                         f"lr={self.config.pca_student_lr}, "
                         f"epochs={self.config.pca_student_epochs}")
        self.logger.info("=" * 60)

        self.pca_student = PCAStudent(
            input_dim=self.config.input_planes,
            hidden_dims=self.config.pca_student_hidden_dims,
        ).to(self.config.device)

        optimizer = torch.optim.Adam(
            self.pca_student.parameters(),
            lr=self.config.pca_student_lr,
            weight_decay=1e-5,
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        # --- Phase 1: 收集特征和 SVD 二值掩模 targets ---
        self.logger.info("Phase 1/2: Extracting features and computing SVD mask targets...")
        all_features = []
        all_targets = []

        temp_pca_gen = PCAMaskGenerator(
            threshold=self.config.pca_threshold,
            border_ratio=self.config.pca_border,
            kernel_size=self.config.pca_kernel_size,
            use_gpu=self.config.pca_use_gpu,
            skip_categories=self.config.pca_skip_categories,
            pca_student=None,
        )
        if self.pca_generator is not None:
            temp_pca_gen.set_category(self.pca_generator.current_category)

        self.extractor.eval()
        with torch.no_grad():
            for images, _, _, _ in tqdm(
                train_dataloader,
                desc="Collecting PCA targets",
                leave=False,
            ):
                images = images.to(self.config.device)
                features, (H, W) = self.extractor(images)
                targets = temp_pca_gen(features, (H, W)).float()
                all_features.append(features.cpu())
                all_targets.append(targets.cpu())

        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        n_patches = all_features.shape[0]
        fg_ratio = all_targets.float().mean().item()
        self.logger.info(f"Collected {n_patches} patches, "
                         f"feature_dim={all_features.shape[1]}.")
        self.logger.info(f"Foreground ratio: {fg_ratio:.3f} "
                         f"({int(fg_ratio * n_patches)} patches)")

        # --- Phase 2: 训练 Student ---
        self.logger.info(f"Phase 2/2: Training for {self.config.pca_student_epochs} "
                         f"epochs...")

        dataset = torch.utils.data.TensorDataset(all_features, all_targets)
        loader = DataLoader(
            dataset,
            batch_size=self.config.pca_student_batch_size,
            shuffle=True,
            drop_last=False,
        )

        best_loss = float("inf")
        for epoch in range(self.config.pca_student_epochs):
            self.pca_student.train()
            total_loss = 0.0
            num_batches = 0

            for batch_feat, batch_target in loader:
                batch_feat = batch_feat.to(self.config.device)
                batch_target = batch_target.to(self.config.device)

                pred = self.pca_student(batch_feat).squeeze(-1)
                loss = criterion(pred, batch_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.pca_student.eval()
                with torch.no_grad():
                    subset_n = min(4096, n_patches)
                    sub_feat = all_features[:subset_n].to(self.config.device)
                    sub_target = all_targets[:subset_n].to(self.config.device)
                    sub_pred = self.pca_student(sub_feat).squeeze(-1)

                    pred_mask = (torch.sigmoid(sub_pred) > 0.5).float()
                    target_mask = sub_target
                    intersection = (pred_mask * target_mask).sum()
                    union = (pred_mask + target_mask).clamp(0, 1).sum()
                    iou = (intersection / max(union, 1)).item()
                    acc = (pred_mask == target_mask).float().mean().item()

                    self.logger.info(
                        f"  PCA Student Epoch {epoch+1:3d}/"
                        f"{self.config.pca_student_epochs}: "
                        f"BCE={avg_loss:.6f}, IoU={iou:.4f}, Acc={acc:.4f}"
                    )

            best_loss = min(best_loss, avg_loss)

        self.pca_student.eval()
        self.logger.info(f"PCA Student training complete. "
                         f"Best BCE={best_loss:.6f}.")

        # 挂接到自己的 PCA generator
        if self.pca_generator is not None:
            self.pca_generator.set_pca_student(self.pca_student)
            self.logger.info("PCA Student attached to Trainer's PCA generator.")



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
            
            # 重塑为图像形式
            patch_scores = patch_scores.cpu().numpy()
            patch_scores = patch_scores.reshape(batch_size, H, W)
            
            # 上采样到目标尺寸
            masks = self._upsample_masks(patch_scores)
            
            # 计算图像级分数
            if aggregation == "max":
                img_scores = self._aggregate_max(patch_scores)
            elif aggregation == "topk":
                img_scores = self._aggregate_topk(patch_scores, k=10)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            all_scores.extend(img_scores.tolist())
            all_masks.extend(masks)
            all_labels.extend(labels.numpy().tolist())
            all_masks_gt.extend(masks_gt.numpy().tolist())
        
        return all_scores, all_masks, all_labels, all_masks_gt
    
    def _upsample_masks(self, patch_scores: np.ndarray) -> List[np.ndarray]:
        """上采样到目标尺寸"""
        B, H, W = patch_scores.shape
        scores_tensor = torch.from_numpy(patch_scores).unsqueeze(1).float()
        
        upsampled = F.interpolate(
            scores_tensor,
            size=(self.config.target_size, self.config.target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # 高斯平滑
        masks = upsampled.numpy()
        return [
            cv2.GaussianBlur(m, (0, 0), sigmaX=4)
            for m in masks
        ]
    
    def _aggregate_max(self, patch_scores: np.ndarray) -> np.ndarray:
        """最大值聚合"""
        return patch_scores.reshape(patch_scores.shape[0], -1).max(axis=1)
    
    def _aggregate_topk(self, patch_scores: np.ndarray, k: int = 10) -> np.ndarray:
        """Top-K平均聚合"""
        B = patch_scores.shape[0]
        flat = patch_scores.reshape(B, -1)
        topk = np.partition(flat, -k, axis=1)[:, -k:]
        return topk.mean(axis=1)

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
        # 特征提取器 - 封装 DINOv2 和特征聚合
        self.feature_extractor = FeatureExtractor(
            model_path=model_path,
            layer_indices=self.config.layer_indices,
            patch_size=self.config.patch_size,
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
        self.pca_student = None  # PCA Student 模型（可选，替代 SVD 推理）

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

    def train_pca_student(self, train_dataloader) -> None:
        """训练 PCA Student（每次调用都重新训练），委托给 Trainer 执行"""
        if not self.config.use_pca_student:
            self.logger.info("PCA Student is disabled (use_pca_student=False). Skipping.")
            return

        if not self.config.use_pca_mask:
            self.logger.warning("PCA Student requires PCA mask (use_pca_mask=True). Skipping.")
            return

        if self.config.pca_skip_categories and self.current_category in self.config.pca_skip_categories:
            self.logger.info(
                f"Category '{self.current_category}' is in skip_categories. "
                f"Skipping PCA Student training (not needed, mask is always all-ones)."
            )
            return

        if self.trainer is None:
            self.trainer = Trainer(
                self.feature_extractor, self.projection,
                self.discriminator, self.config, self.logger
            )
            if self.current_category is not None and self.trainer.pca_generator:
                self.trainer.pca_generator.set_category(self.current_category)

        self.trainer.train_pca_student(train_dataloader)
        self.pca_student = self.trainer.pca_student

    def fit(self, train_dataloader) -> Dict[str, float]:
        """训练一个 meta epoch"""
        if self.trainer is None:
            self.trainer = Trainer(
                self.feature_extractor, self.projection,
                self.discriminator, self.config, self.logger
            )
            if self.current_category is not None and self.trainer.pca_generator:
                self.trainer.pca_generator.set_category(self.current_category)
            if self.pca_student is not None and self.trainer.pca_generator is not None:
                self.trainer.pca_generator.set_pca_student(self.pca_student)
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
            # 挂接 PCA Student 到新创建的 predictor
            if self.pca_student is not None and self.predictor.pca_generator is not None:
                self.predictor.pca_generator.set_pca_student(self.pca_student)

        return self.predictor.predict(test_dataloader, aggregation)
    
    def save(self, path: str, epoch: int = 0, scores: dict = None):
        """保存模型权重（仅 Projection + Discriminator）"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            'proj_state': self.projection.state_dict(),
            'dsc_state': self.discriminator.state_dict(),
            'epoch': epoch,
            'scores': scores,
        }
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load(self, path: str):
        """加载模型权重，清空 Trainer/Predictor 使其使用新权重重建"""
        state = torch.load(path, map_location=self.config.device)
        self.projection.load_state_dict(state['proj_state'])
        self.discriminator.load_state_dict(state['dsc_state'])
        self.trainer = None
        self.predictor = None
        self.logger.info(f"Checkpoint loaded from {path}")
        return state.get('epoch', 0), state.get('scores', None), None, -1
    
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


