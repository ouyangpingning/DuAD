"""SimpleNet 核心实现。

参考 src/myAD.py 的架构设计:
  SimpleNetConfig  → 配置 dataclass
  SimpleNetFeatureExtractor → 特征提取 (WideResNet50-2 + _embed)
  SimpleNetTrainer  → 训练逻辑
  SimpleNetPredictor → 推理逻辑
  SimpleNet         → 主协调器 (facade)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import sys
import os

# 复用 src 的公共组件
_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _src_dir)
from commen_import import *
import scipy.ndimage as ndimage
from utils import (
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
    setup_logger,
    init_weight,
)
from myAD import Projection
from simplenet.wide_resnet import wide_resnet50_2, _embed


class SimpleNetDiscriminator(torch.nn.Module):
    """SimpleNet 原版判别器 — 与 src/myAD.py 的区别是包含 BatchNorm1d。

    原版 SimpleNet (CVPR 2023) 在 discriminator body 中使用了 BatchNorm1d，
    这对稳定训练和最终性能有实际影响。
    """
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super().__init__()
        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


@dataclass
class SimpleNetConfig:
    """SimpleNet 模型配置"""
    # 架构参数
    target_size: int = 288
    layer_indices: List[int] = None    # WideResNet 层索引 [1, 2] = layer2, layer3
    input_planes: int = 1536           # 投影器输入维度
    hidden_dim: int = 1024             # 判别器隐藏层维度

    # 训练参数
    meta_epochs: int = 40
    gan_epochs: int = 4
    batch_size: int = 8
    proj_lr: float = 1e-3
    dsc_lr: float = 2e-4
    dsc_margin: float = 0.5
    use_scheduler: bool = True

    # 噪声参数
    noise_std: float = 0.015
    mix_noise: int = 1                # 噪声强度种类数

    # 数据增强控制（消融实验：少样本时启用，与 DuAD 对齐）
    augment_categories: List[str] = None
    color_augment_categories: List[str] = None

    # 其他
    patch_size: int = 3
    resize: int = 329                 # 先 resize 到此尺寸
    isize: int = 288                  # 再中心裁剪到此尺寸
    target_patches: int = 1296        # _embed 对齐到的 patch 数
    device: str = "cuda"

    def __post_init__(self):
        if self.layer_indices is None:
            self.layer_indices = [1, 2]


def create_noise(true_feats: torch.Tensor, mix_noise: int, noise_std: float, device: str):
    """为输入特征生成高斯噪声。

    支持多种噪声强度: 按 1.1^k 倍 std 创建 K 种，随机分配给每个 patch。
    当 mix_noise=1 时退化为单一强度。
    """
    noise_idxs = torch.randint(0, mix_noise, torch.Size([true_feats.shape[0]]))
    noise_one_hot = F.one_hot(noise_idxs, num_classes=mix_noise).to(device)

    noise = torch.stack([
        torch.normal(0, noise_std * 1.1 ** k, true_feats.shape)
        for k in range(mix_noise)
    ], dim=1).to(device)

    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
    return noise


class SimpleNetFeatureExtractor(torch.nn.Module):
    """特征提取器 - WideResNet50-2 + _embed 聚合"""

    def __init__(
        self,
        layer_indices: List[int],
        patch_size: int = 3,
        target_patches: int = 1296,
        target_dim: int = 1536,
        output_size: int = 1536,
        device: str = "cuda",
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.patch_size = patch_size
        self.target_patches = target_patches
        self.target_dim = target_dim
        self.output_size = output_size
        self.device = device

        self.encoder = wide_resnet50_2(pretrained=True).to(self.device)
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]]]:
        """返回 (patches_features, patch_shapes)"""
        features = self.encoder(images)
        patches_features, patch_shapes = _embed(
            features=features,
            layers=self.layer_indices,
            patchsize=self.patch_size,
            stride=1,
            target_patches=self.target_patches,
            target_dim=self.target_dim,
            output_size=self.output_size,
        )
        return patches_features, patch_shapes


class SimpleNetTrainer:
    """训练器 - Hinge loss 对抗训练"""

    def __init__(
        self,
        feature_extractor: SimpleNetFeatureExtractor,
        projection: torch.nn.Module,
        discriminator: torch.nn.Module,
        config: SimpleNetConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.extractor = feature_extractor
        self.projection = projection
        self.discriminator = discriminator
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        trainable_params = list(projection.parameters()) + list(discriminator.parameters())

        self.proj_opt = torch.optim.AdamW(
            trainable_params,
            lr=config.proj_lr * 0.1,
            weight_decay=1e-4,
        )
        self.dsc_opt = torch.optim.Adam(
            discriminator.parameters(),
            lr=config.dsc_lr,
            weight_decay=1e-5,
        )

        self.scheduler = None
        if config.use_scheduler:
            total_steps = config.gan_epochs * config.meta_epochs
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.dsc_opt,
                T_max=total_steps,
                eta_min=config.dsc_lr * 0.4,
            )

        self.global_step = 0

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.projection.train()
        self.discriminator.train()
        self.extractor.eval()

        all_loss = []
        all_p_true = []
        all_p_fake = []

        for gan_epoch in range(self.config.gan_epochs):
            pbar = tqdm(dataloader, desc=f"GAN {gan_epoch + 1}/{self.config.gan_epochs}")

            for images, _, _, _ in pbar:
                loss, p_true, p_fake = self._train_step(images)
                all_loss.append(loss)
                all_p_true.append(p_true)
                all_p_fake.append(p_fake)
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'p_t': f'{p_true:.3f}',
                    'p_f': f'{p_fake:.3f}',
                })

            if self.scheduler:
                self.scheduler.step()

        return {
            'loss': sum(all_loss) / max(len(all_loss), 1),
            'p_true': sum(all_p_true) / max(len(all_p_true), 1),
            'p_fake': sum(all_p_fake) / max(len(all_p_fake), 1),
        }

    def _train_step(self, images: torch.Tensor) -> Tuple[float, float, float]:
        images = images.to(self.config.device)

        features, _ = self.extractor(images)
        projected = self.projection(features)

        noise = create_noise(
            projected, self.config.mix_noise,
            self.config.noise_std, self.config.device,
        )
        fake = projected + noise

        scores = self.discriminator(torch.cat([projected, fake], dim=0))
        true_scores = scores[:len(projected)]
        fake_scores = scores[len(projected):]

        th = self.config.dsc_margin
        true_loss = torch.clip(-true_scores + th, min=0).mean()
        fake_loss = torch.clip(fake_scores + th, min=0).mean()
        loss = true_loss + fake_loss

        self.proj_opt.zero_grad()
        self.dsc_opt.zero_grad()
        loss.backward()
        self.proj_opt.step()
        self.dsc_opt.step()

        self.global_step += 1

        with torch.no_grad():
            p_true = (true_scores >= th).float().mean().item()
            p_fake = (fake_scores < -th).float().mean().item()

        return loss.item(), p_true, p_fake


class SimpleNetPredictor:
    """预测器 - 生成异常分数和分割掩码"""

    def __init__(
        self,
        feature_extractor: SimpleNetFeatureExtractor,
        projection: torch.nn.Module,
        discriminator: torch.nn.Module,
        config: SimpleNetConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.extractor = feature_extractor
        self.projection = projection
        self.discriminator = discriminator
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._eval_mode()

    def _eval_mode(self):
        self.extractor.eval()
        self.projection.eval()
        self.discriminator.eval()

    @torch.no_grad()
    def predict(self, dataloader) -> Tuple[List[float], List[np.ndarray], List, List]:
        all_scores = []
        all_masks = []
        all_labels = []
        all_masks_gt = []

        for images, masks_gt, labels, _ in tqdm(dataloader, desc="Predicting"):
            batch_size = images.shape[0]
            images = images.to(self.config.device)

            features, patch_shapes = self.extractor(images)
            projected = self.projection(features)
            patch_scores = -self.discriminator(projected)

            patch_scores = patch_scores.cpu().numpy()

            # 图像级分数: max over all patches
            img_scores = patch_scores.reshape(batch_size, -1)
            img_scores = img_scores.max(axis=1)

            # 像素级分数: reshape 到空间网格
            grid_h, grid_w = patch_shapes[0]
            patch_scores_2d = patch_scores.reshape(batch_size, grid_h, grid_w)

            masks = self._upsample_masks(patch_scores_2d)

            all_scores.extend(img_scores.tolist())
            all_masks.extend(masks)
            all_labels.extend(labels.numpy().tolist())
            all_masks_gt.extend(masks_gt.numpy().tolist())

        return all_scores, all_masks, all_labels, all_masks_gt

    def _upsample_masks(self, patch_scores: np.ndarray) -> List[np.ndarray]:
        B, H, W = patch_scores.shape
        scores_tensor = torch.from_numpy(patch_scores).unsqueeze(1).float()

        upsampled = F.interpolate(
            scores_tensor,
            size=(self.config.target_size, self.config.target_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

        masks = upsampled.numpy()
        return [
            ndimage.gaussian_filter(m, sigma=4)
            for m in masks
        ]


class SimpleNet:
    """SimpleNet 主协调器"""

    def __init__(
        self,
        config: Optional[SimpleNetConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or SimpleNetConfig()
        self.logger = logger or logging.getLogger(__name__)

        self.feature_extractor = SimpleNetFeatureExtractor(
            layer_indices=self.config.layer_indices,
            patch_size=self.config.patch_size,
            target_patches=self.config.target_patches,
            target_dim=self.config.input_planes,
            output_size=self.config.input_planes,
            device=self.config.device,
        )

        self.projection = Projection(
            in_planes=self.config.input_planes,
            n_layers=1,
            layer_type=0,
        ).to(self.config.device)

        self.discriminator = SimpleNetDiscriminator(
            in_planes=self.config.input_planes,
            n_layers=2,
            hidden=self.config.hidden_dim,
        ).to(self.config.device)

        self.trainer = None
        self.predictor = None

        self._log_init()

    def _log_init(self):
        self.logger.info("=" * 60)
        self.logger.info("SimpleNet Initialization (WideResNet50-2)")
        self.logger.info("=" * 60)
        for key, value in vars(self.config).items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)

    def fit(self, train_dataloader) -> Dict[str, float]:
        if self.trainer is None:
            self.trainer = SimpleNetTrainer(
                self.feature_extractor,
                self.projection,
                self.discriminator,
                self.config,
                self.logger,
            )
        return self.trainer.train_epoch(train_dataloader)

    def predict(self, test_dataloader) -> Tuple[List[float], List[np.ndarray], List, List]:
        if self.predictor is None:
            self.predictor = SimpleNetPredictor(
                self.feature_extractor,
                self.projection,
                self.discriminator,
                self.config,
                self.logger,
            )
        return self.predictor.predict(test_dataloader)

    def save(self, path: str, epoch: int = 0, scores: dict = None):
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
        state = torch.load(path, map_location=self.config.device)
        self.projection.load_state_dict(state['proj_state'])
        self.discriminator.load_state_dict(state['dsc_state'])
        self.trainer = None
        self.predictor = None
        self.logger.info(f"Checkpoint loaded from {path}")
        return state.get('epoch', 0), state.get('scores', None)

    def evaluate(
        self,
        scores: List[float],
        segmentations: List[np.ndarray],
        labels_gt: List,
        masks_gt: List[np.ndarray],
        compute_full_metrics: bool = False,
    ) -> Dict[str, float]:
        """评估。训练期快速模式只算 AUROC，完整模式算全部指标。"""
        # 图像级: 逐图 min-max 归一化
        scores_arr = np.squeeze(np.array(scores))
        img_min = scores_arr.min(axis=-1)
        img_max = scores_arr.max(axis=-1)
        scores_norm = (scores_arr - img_min) / (img_max - img_min)

        img_metrics = compute_imagewise_retrieval_metrics(scores_norm, labels_gt)

        if not compute_full_metrics:
            if len(masks_gt) > 0:
                seg_arr = np.array(segmentations)
                seg_mins = seg_arr.reshape(len(seg_arr), -1).min(axis=-1).reshape(-1, 1, 1, 1)
                seg_maxs = seg_arr.reshape(len(seg_arr), -1).max(axis=-1).reshape(-1, 1, 1, 1)
                ranges = np.maximum(seg_maxs - seg_mins, 1e-2)
                seg_norm = (seg_arr * (1.0 / ranges).sum() - (seg_mins / ranges).sum()) / len(segmentations)
                pixel_auroc = metrics.roc_auc_score(
                    np.array(masks_gt).ravel().astype(int), seg_norm.ravel()
                )
                return {'image_auroc': img_metrics['auroc'], 'pixel_auroc': pixel_auroc}
            return {'image_auroc': img_metrics['auroc'], 'pixel_auroc': -1}

        # 完整指标
        if len(masks_gt) > 0:
            seg_arr = np.array(segmentations)
            seg_mins = seg_arr.reshape(len(seg_arr), -1).min(axis=-1).reshape(-1, 1, 1, 1)
            seg_maxs = seg_arr.reshape(len(seg_arr), -1).max(axis=-1).reshape(-1, 1, 1, 1)
            ranges = np.maximum(seg_maxs - seg_mins, 1e-2)
            seg_norm = (seg_arr * (1.0 / ranges).sum() - (seg_mins / ranges).sum()) / len(segmentations)
            pixel_metrics = compute_pixelwise_retrieval_metrics(seg_norm, masks_gt)

            return {
                'image_auroc': img_metrics['auroc'],
                'image_ap': img_metrics['ap'],
                'image_f1': img_metrics['f1'],
                'pixel_auroc': pixel_metrics['auroc'],
                'pixel_ap': pixel_metrics['ap'],
                'pixel_f1': pixel_metrics['f1'],
                'pixel_pro': pixel_metrics['pro'],
            }

        return {'image_auroc': img_metrics['auroc'], 'image_ap': img_metrics['ap'],
                'image_f1': img_metrics['f1'], 'pixel_auroc': -1, 'pixel_ap': -1,
                'pixel_f1': -1, 'pixel_pro': -1}
