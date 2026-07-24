import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commen_import import *
from utils import clean_GPU_Cache, setup_logger
from dataset import get_dataloader, get_transform
from DuAD import DINOv2AnomalyDetector, ModelConfig, PCAMaskGenerator, PerlinMaskGenerator
from config import load_config, build_model_config, get_category_pca_thresholds, get_category_pca_border_thresholds, get_paths
from sklearn.decomposition import PCA
import cv2
import click
import random


# ==============================================================================
# CategoryVisualizer — 将每个类别的所有可视化逻辑封装为独立方法
# ==============================================================================

class CategoryVisualizer:
    """单个类别的可视化编排器（Strategy + Template Method 模式）。

    每个 visualize_*() 方法是一个独立的可视化策略；
    run_all() 是模板方法，按固定顺序依次调用各策略。
    """

    def __init__(
        self,
        atype: str,
        vis_suffix: str,
        detector: DINOv2AnomalyDetector,
        config: ModelConfig,
        train_loader,
        test_loader,
        device: torch.device,
        output_dir: str,
        logger,
        *,
        do_flip: bool = False,
        do_rotate: bool = False,
        do_translate: bool = False,
        do_color_jitter: bool = False,
        k_shot: int = None,
        num_samples: int = 4,
        skip_inference: bool = False,
        query_seed: int = 42,
        compare_detector: DINOv2AnomalyDetector = None,
        compare_label: str = "Compare",
    ):
        self.atype = atype
        self.vis_suffix = vis_suffix
        self.detector = detector
        self.compare_detector = compare_detector
        self.compare_label = compare_label
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.do_flip = do_flip
        self.do_rotate = do_rotate
        self.do_translate = do_translate
        self.do_color_jitter = do_color_jitter
        self.k_shot = k_shot
        self.num_samples = num_samples
        self.skip_inference = skip_inference
        self.query_seed = query_seed

        # 缓存：避免重复从 DataLoader 取第一张训练图
        self._cached_train_sample = None

    # =====================================================================
    # 共享工具方法
    # =====================================================================

    @staticmethod
    def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
        """ImageNet 反归一化: tensor [C,H,W] → numpy [H,W,C] (值域 [0,1])"""
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        return np.clip(img_np, 0, 1)

    @staticmethod
    def _setup_chinese_font():
        """配置 matplotlib 中文字体（全局，仅需调用一次）"""
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False

    def _get_train_sample(self) -> torch.Tensor:
        """获取训练集第一张图 [1, C, H, W]，结果缓存复用"""
        if self._cached_train_sample is None:
            train_iter = iter(self.train_loader)
            images, _, _, _ = next(train_iter)
            self._cached_train_sample = images[0:1].to(self.device)
        return self._cached_train_sample

    def _create_pca_generator(self) -> PCAMaskGenerator:
        """工厂方法：用当前 config 创建 PCAMaskGenerator"""
        pca_gen = PCAMaskGenerator(
            threshold=self.config.pca_threshold,
            border_ratio=self.config.pca_border,
            kernel_size=self.config.pca_kernel_size,
            use_gpu=self.config.pca_use_gpu,
            skip_categories=self.config.pca_skip_categories,
        )
        pca_gen.set_category(self.atype)
        return pca_gen

    # =====================================================================
    # 1. 数据增强可视化（仅少样本模式）
    # =====================================================================

    def visualize_augmented(self):
        """可视化训练数据增强效果（2×4 网格）"""
        train_iter = iter(self.train_loader)
        aug_images, _, _, _ = next(train_iter)
        B = min(aug_images.shape[0], 8)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i in range(B):
            img = self._denormalize(aug_images[i])
            axes[i].imshow(img)
            axes[i].set_title(f'Aug #{i + 1}')
            axes[i].axis('off')
        for i in range(B, 8):
            axes[i].axis('off')

        plt.suptitle(
            f'{self.atype} - Augmented Training Images (K={self.k_shot})',
            fontsize=14)
        plt.tight_layout()

        save_dir = f"{self.output_dir}/augmented/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{self.atype}{self.vis_suffix}_augmented.png"
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Augmented image visualization saved to: {save_path}")

    # =====================================================================
    # 2. 异常热力图（随机抽样 + 推理 + F1 阈值过滤）
    # =====================================================================

    def visualize_anomaly_heatmap(self):
        """随机抽取 num_samples 张测试图，逐张推理并绘制热力图网格。

        当 self.compare_detector 不为 None 时，生成 N×4 对比网格:
        [原图 | GT Mask | Heatmap A | Heatmap B]
        否则生成 N×3 网格。
        """
        has_compare = self.compare_detector is not None
        self.logger.info(
            f"Collecting test samples for random visualization "
            f"(n={self.num_samples}, compare={'yes' if has_compare else 'no'})...")

        # --- 收集全部测试样本 ---
        all_samples = []
        for images, masks_gt, labels, _ in self.test_loader:
            for i in range(len(labels)):
                all_samples.append({
                    'image': images[i],
                    'mask_gt': masks_gt[i],
                    'label': labels[i].item(),
                })

        anomaly_samples = [s for s in all_samples if s['label'] == 1]
        normal_samples = [s for s in all_samples if s['label'] == 0]
        self.logger.info(
            f"Test set: {len(anomaly_samples)} anomalies, "
            f"{len(normal_samples)} normals ({len(all_samples)} total)")

        # --- 随机抽样：优先异常，不足时正常补齐 ---
        random.seed(self.query_seed)
        if len(anomaly_samples) >= self.num_samples:
            selected = random.sample(anomaly_samples, self.num_samples)
        else:
            selected = anomaly_samples.copy()
            needed = self.num_samples - len(selected)
            if len(normal_samples) >= needed:
                selected += random.sample(normal_samples, needed)
            else:
                selected += normal_samples
        random.shuffle(selected)

        n_anomaly = sum(1 for s in selected if s['label'] == 1)
        n_normal = sum(1 for s in selected if s['label'] == 0)
        self.logger.info(
            f"Selected {len(selected)} samples "
            f"({n_anomaly} anomalies, {n_normal} normals)")

        # --- 准备推理环境 (detector A) ---
        self.detector.feature_extractor.eval()
        self.detector.projection.eval()
        self.detector.discriminator.eval()
        pca_gen = (self._create_pca_generator()
            if self.config.use_pca_mask else None)

        # --- 准备推理环境 (detector B, 对比模式) ---
        if has_compare:
            self.compare_detector.feature_extractor.eval()
            self.compare_detector.projection.eval()
            self.compare_detector.discriminator.eval()
            # 对比检测器共享同一个 PCA 生成器以保证掩模一致
            pca_gen_b = (self._create_pca_generator()
                if self.config.use_pca_mask else None)
        else:
            pca_gen_b = None

        target_size = self.config.target_size

        # --- 逐张推理 ---
        results = []
        for s in selected:
            sample_img = s['image'].unsqueeze(0).to(self.device)
            sample_gt_mask = s['mask_gt'].cpu().numpy().squeeze()
            if sample_gt_mask.ndim > 2:
                sample_gt_mask = sample_gt_mask.squeeze()

            # 反归一化原图 (两个检测器共享)
            img_np = self._denormalize(sample_img[0])

            # 推理: detector A
            heatmap_a = self._infer_heatmap(
                self.detector, sample_img, pca_gen,
                target_size, s['label'], sample_gt_mask)

            # 推理: detector B (对比模式)
            heatmap_b = None
            if has_compare:
                heatmap_b = self._infer_heatmap(
                    self.compare_detector, sample_img, pca_gen_b,
                    target_size, s['label'], sample_gt_mask)

            results.append({
                'img_np': img_np,
                'gt_mask': sample_gt_mask,
                'heatmap_a': heatmap_a,
                'heatmap_b': heatmap_b,
                'label': s['label'],
            })

        # --- N×K 网格渲染 (K=3 单模型, K=4 对比模式) ---
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        n_cols = 4 if has_compare else 3
        n = len(results)
        fig, axes = plt.subplots(n, n_cols, figsize=(6 * n_cols, 6 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, r in enumerate(results):
            label_str = 'Anomaly' if r['label'] == 1 else 'Normal'

            # Col 0: 原图
            axes[i, 0].imshow(r['img_np'])
            axes[i, 0].set_title(f'[{label_str}] Original', fontsize=11)
            axes[i, 0].axis('off')

            # Col 1: GT 掩码叠加
            gt_mask_up = cv2.resize(
                r['gt_mask'].astype(np.float32),
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST)
            gt_mask_binary = gt_mask_up > 0.5
            axes[i, 1].imshow(r['img_np'])
            overlay = np.zeros((target_size, target_size, 4), dtype=np.float32)
            overlay[gt_mask_binary] = [1, 0, 0, 0.5]
            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f'[{label_str}] GT Mask', fontsize=11)
            axes[i, 1].axis('off')

            # Col 2: Heatmap A (主检测器) — 不单独加 colorbar
            axes[i, 2].imshow(r['img_np'], alpha=1.0)
            axes[i, 2].imshow(r['heatmap_a'], cmap='plasma', alpha=0.8)
            axes[i, 2].set_title(f'[{label_str}] Heatmap A', fontsize=11)
            axes[i, 2].axis('off')

            # Col 3: Heatmap B (对比检测器) — 最右列，加 colorbar
            if has_compare and r['heatmap_b'] is not None:
                axes[i, 3].imshow(r['img_np'], alpha=1.0)
                im_last = axes[i, 3].imshow(r['heatmap_b'], cmap='plasma', alpha=0.8)
                axes[i, 3].set_title(
                    f'[{label_str}] Heatmap B ({self.compare_label})',
                    fontsize=11)
                axes[i, 3].axis('off')
                divider = make_axes_locatable(axes[i, 3])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im_last, cax=cax)
            elif not has_compare:
                # 单模型模式: Col 2 就是最右列，在此加 colorbar
                divider = make_axes_locatable(axes[i, 2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                # 重新获取 im 引用 (上一个 imshow 返回的对象)
                im_last = axes[i, 2].images[-1]
                plt.colorbar(im_last, cax=cax)

        title = (f'{self.atype}{self.vis_suffix} — Anomaly Detection '
                 f'({n_anomaly} anomalies + {n_normal} normals)')
        if has_compare:
            title += f' | Compare: {self.compare_label}'
        plt.suptitle(title, fontsize=15, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        suffix = "_compare" if has_compare else ""
        save_path = f"{self.output_dir}/{self.atype}{self.vis_suffix}_heatmap{suffix}.png"
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Anomaly heatmap saved to: {save_path}")

    def _infer_heatmap(self, detector, sample_img, pca_gen,
                       target_size, label, gt_mask):
        """对单张图运行检测器推理，返回归一化后的 heatmap [H, W]。

        Args:
            detector: DINOv2AnomalyDetector 实例
            sample_img: [1, C, H, W] tensor (已在目标设备上)
            pca_gen: PCAMaskGenerator 或 None
            target_size: 输出 heatmap 尺寸
            label: 0 (正常) 或 1 (异常)
            gt_mask: ground truth mask [H, W] numpy, 仅 label==1 时用于 F1 阈值
        Returns:
            heatmap: [target_size, target_size] numpy float32, NaN=背景/低于F1阈值
        """
        with torch.no_grad():
            features, (H, W) = detector.feature_extractor(sample_img)

            mask_tensor = None
            if pca_gen:
                mask_tensor = pca_gen(features, (H, W))
                features_masked = pca_gen.apply_mask(
                    features, mask_tensor, self.device)
            else:
                features_masked = features

            projected = detector.projection(features_masked)
            patch_scores = -detector.discriminator(projected)

            # 构建完整 H×W 分数图（背景填 0，后续会被 NaN 掉）
            fg_mask_np = None
            if pca_gen and mask_tensor is not None:
                fg_mask_np = mask_tensor.cpu().numpy().reshape(H, W)
                full_scores = torch.zeros(H * W, 1, device=self.device)
                full_scores[mask_tensor] = patch_scores
                patch_scores = full_scores

            patch_scores = patch_scores.cpu().numpy().reshape(H, W)
            heatmap = cv2.resize(
                patch_scores.astype(np.float32), (target_size, target_size))
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4)

            # 背景 NaN
            if fg_mask_np is not None:
                bg_mask = (~fg_mask_np).astype(np.float32)
                bg_mask_up = cv2.resize(
                    bg_mask, (target_size, target_size),
                    interpolation=cv2.INTER_NEAREST)
                bg_mask_up = cv2.GaussianBlur(bg_mask_up, (0, 0), sigmaX=4)
                fg_mask_up = ~(bg_mask_up > 0.5)
                heatmap[bg_mask_up > 0.5] = np.nan
            else:
                fg_mask_up = np.ones((target_size, target_size), dtype=bool)

            # 百分位归一化（仅前景）
            fg_values = heatmap[~np.isnan(heatmap)]
            if len(fg_values) > 0:
                vmin = np.percentile(fg_values, 2)
                vmax = np.percentile(fg_values, 98)
                if vmax - vmin > 1e-8:
                    heatmap = np.clip(heatmap, vmin, vmax)
                    heatmap = (heatmap - vmin) / (vmax - vmin)

            # F1 阈值过滤（仅异常样本）
            if label == 1:
                self._apply_f1_threshold(
                    heatmap, gt_mask, fg_mask_up, target_size)

        return heatmap

    @staticmethod
    def _apply_f1_threshold(
        heatmap: np.ndarray,
        gt_mask: np.ndarray,
        fg_mask: np.ndarray,
        target_size: int,
    ) -> None:
        """在异常样本上计算最优 F1 阈值，将低于阈值的像素置为 NaN（原地修改 heatmap）"""
        gt_mask_up = cv2.resize(
            gt_mask.astype(np.float32), (target_size, target_size))
        gt_mask_up = (gt_mask_up > 0.5)

        fg_heatmap = heatmap[fg_mask]
        fg_gt = gt_mask_up[fg_mask]
        if fg_gt.sum() == 0:
            return

        thresholds = np.linspace(0.05, 0.95, 60)
        best_f1 = 0.0
        best_thresh = 0.0
        for t in thresholds:
            pred = fg_heatmap > t
            tp = (pred & fg_gt).sum()
            fp = (pred & ~fg_gt).sum()
            fn = (~pred & fg_gt).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (2 * precision * recall / max(precision + recall, 1e-8))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        heatmap[heatmap <= best_thresh] = np.nan

    # =====================================================================
    # 3. PCA 掩模可视化
    # =====================================================================

    def visualize_pca_mask(self) -> dict:
        """生成 PCA 掩模图；返回 dict 供 visualize_perlin_mask 复用。

        Returns:
            {'svd_mask': Tensor, 'svd_mask_up': np.ndarray,
             'H': int, 'W': int,
             'sample_image': Tensor [1,C,H_img,W_img],
             'img_np': np.ndarray}
        """
        os.makedirs(f"{self.output_dir}/pca_mask/", exist_ok=True)
        sample_image = self._get_train_sample()
        target_size = self.config.target_size

        self.detector.feature_extractor.eval()
        with torch.no_grad():
            features, (H, W) = self.detector.feature_extractor(sample_image)

            # SVD 掩模
            pca_gen_svd = self._create_pca_generator()
            svd_mask = pca_gen_svd(features, (H, W))
            svd_mask_2d = svd_mask.cpu().numpy().reshape(H, W)

            # SVD 第一主成分投影值 (sklearn PCA for visualization)
            features_np = features.cpu().numpy()
            pca_sk = PCA(n_components=1, svd_solver='randomized')
            first_pc = pca_sk.fit_transform(features_np).squeeze()
            first_pc_2d = first_pc.reshape(H, W)

            # 准备图像（min-max 归一化）
            img_np = sample_image[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            # 上采样
            first_pc_up = cv2.resize(first_pc_2d, (target_size, target_size))
            svd_mask_up = cv2.resize(
                svd_mask_2d.astype(np.float32), (target_size, target_size))

        # 3 列可视化: 原图 | 第一主成分 | PCA掩模
        self._visualize_pca_mask_fallback(
            image=img_np,
            mask=svd_mask_up,
            first_pc=first_pc_up,
            save_path=f"{self.output_dir}/pca_mask/"
                     f"{self.atype}{self.vis_suffix}_pca_mask.png",
        )

        self.logger.info(
            f"PCA mask visualization saved to: "
            f"{self.output_dir}/pca_mask/"
            f"{self.atype}{self.vis_suffix}_pca_mask.png")

        return {
            'svd_mask': svd_mask,
            'svd_mask_up': svd_mask_up,
            'H': H,
            'W': W,
            'sample_image': sample_image,
            'img_np': img_np,
        }

    # =====================================================================
    # 4. Perlin 掩模可视化
    # =====================================================================

    def visualize_perlin_mask(self, pca_data: dict):
        """基于 PCA 掩模生成 Perlin 噪声掩模可视化。

        Args:
            pca_data: visualize_pca_mask() 的返回值
        """
        os.makedirs(f"{self.output_dir}/perlin_mask/", exist_ok=True)

        svd_mask = pca_data['svd_mask']
        svd_mask_up = pca_data['svd_mask_up']
        H = pca_data['H']
        W = pca_data['W']
        sample_image = pca_data['sample_image']
        img_np = pca_data['img_np']
        target_size = self.config.target_size

        perlin_gen = PerlinMaskGenerator(
            min_scale=self.config.perlin_min,
            max_scale=self.config.perlin_max,
        )

        with torch.no_grad():
            pca_mask_img = F.interpolate(
                svd_mask.reshape(1, 1, H, W).float(),
                size=(target_size, target_size),
                mode='nearest'
            ).squeeze().cpu().numpy()

            try:
                perlin_s = perlin_gen(
                    img_shape=(sample_image.shape[1], target_size, target_size),
                    feat_size=H,
                    mask_fg=pca_mask_img,
                )
                perlin_2d = perlin_s
                perlin_up = cv2.resize(
                    perlin_2d.astype(np.float32), (target_size, target_size))
            except Exception as e:
                self.logger.warning(f"Perlin mask generation failed: {e}")
                perlin_2d = np.zeros((H, W), dtype=np.float32)
                perlin_up = cv2.resize(perlin_2d, (target_size, target_size))

            fig, axes = plt.subplots(1, 4, figsize=(20, 6))

            axes[0].imshow(img_np)
            axes[0].set_title('原图')
            axes[0].axis('off')

            axes[1].imshow(svd_mask_up, cmap='gray')
            axes[1].set_title('PCA掩模')
            axes[1].axis('off')

            axes[2].imshow(perlin_up, cmap='gray')
            axes[2].set_title('Perlin掩模')
            axes[2].axis('off')

            axes[3].imshow(img_np)
            overlay = np.zeros((*perlin_up.shape, 4))
            overlay[perlin_up > 0.5] = [0, 1, 0, 0.3]
            axes[3].imshow(overlay)
            axes[3].set_title('Perlin掩模叠加')
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(
                f"{self.output_dir}/perlin_mask/"
                f"{self.atype}{self.vis_suffix}_perlin_mask.png",
                dpi=600, bbox_inches='tight')
            plt.close()

        self.logger.info(
            f"Perlin mask visualization saved to: "
            f"{self.output_dir}/perlin_mask/"
            f"{self.atype}{self.vis_suffix}_perlin_mask.png")

    # =====================================================================
    # 5. DINOv2 特征激活图
    # =====================================================================

    def visualize_feature_map(self):
        """生成 DINOv2 特征 L2 范数激活热力图"""
        os.makedirs(f"{self.output_dir}/feature_map/", exist_ok=True)
        sample_image = self._get_train_sample()
        target_size = self.config.target_size

        self.detector.feature_extractor.eval()
        with torch.no_grad():
            features, (H, W) = self.detector.feature_extractor(sample_image)
            feat_map = features.reshape(1, H, W, -1)
            activation = torch.norm(
                feat_map, p=2, dim=-1).squeeze(0).cpu().numpy()
            activation = ((activation - activation.min()) /
                          (activation.max() - activation.min() + 1e-8))
            activation_up = cv2.resize(activation, (target_size, target_size))

        img_np = self._denormalize(sample_image[0])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_np)
        axes[0].set_title('原图')
        axes[0].axis('off')
        axes[1].imshow(activation_up, cmap='jet')
        axes[1].set_title('DINOv2特征激活热力图')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/feature_map/"
            f"{self.atype}{self.vis_suffix}_feature_map.png",
            dpi=600, bbox_inches='tight')
        plt.close()
        self.logger.info(
            f"Feature map visualization saved to: "
            f"{self.output_dir}/feature_map/"
            f"{self.atype}{self.vis_suffix}_feature_map.png")

    # =====================================================================
    # 模板方法：按固定顺序执行所有可视化
    # =====================================================================

    @staticmethod
    def _visualize_pca_mask_fallback(image, mask, first_pc, save_path=None):
        """3 列 PCA 掩模可视化: 原图 | 第一主成分 | PCA掩模"""
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        axes[0].imshow(image)
        axes[0].set_title('原图')
        axes[0].axis('off')

        im = axes[1].imshow(first_pc, cmap='viridis')
        axes[1].set_title('第一主成分')
        axes[1].axis('off')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        axes[2].imshow(image)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0.5] = [1, 0, 0, 0.3]
        axes[2].imshow(overlay)
        axes[2].set_title('PCA掩模')
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=600)
            plt.close()
        else:
            plt.show()

    def run_all(self):
        """依次执行当前类别的所有可视化（模板方法）。"""
        self._setup_chinese_font()

        # 1. 数据增强图（仅少样本模式）
        if self.do_flip or self.do_rotate or self.do_translate or self.do_color_jitter:
            self.logger.info("Visualizing augmented training images...")
            self.visualize_augmented()

        # 2. 异常热力图（受 --skip_inference 控制）
        if not self.skip_inference:
            self.visualize_anomaly_heatmap()
        else:
            self.logger.info(
                "Skipping model inference (--skip_inference).")

        # 3. PCA 掩模（同时为 Perlin 准备复用数据）
        pca_data = None
        if self.config.use_pca_mask:
            self.logger.info("Generating PCA mask visualization...")
            pca_data = self.visualize_pca_mask()

        # 4. Perlin 掩模（依赖 PCA 数据）
        if (self.config.use_perlin_mask
                and self.config.use_pca_mask
                and pca_data is not None):
            self.logger.info("Generating Perlin mask visualization...")
            self.visualize_perlin_mask(pca_data)

        # 5. DINOv2 特征激活图
        self.logger.info("Generating DINOv2 feature map visualization...")
        self.visualize_feature_map()

        self.logger.info(
            f"Category {self.atype} processing completed.\n")


# ==============================================================================
# Click CLI — 精简后的 main()
# ==============================================================================

@click.command()
@click.option(
    '--categories',
    type=str,
    default="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper",
    show_default=True,
    help='要可视化的类别列表，空格分隔，例如 "pill screw toothbrush transistor wood"'
)
@click.option(
    '--k_shot',
    type=int,
    default=None,
    help='少样本数量，None表示使用全样本模型。例如 --k_shot 4'
)
@click.option(
    '--shot_seed',
    type=int,
    default=0,
    help='少样本采样种子，需与训练时一致。例如 --shot_seed 42'
)
@click.option(
    '--dataset',
    type=click.Choice(['mvtec', 'visa']),
    default='mvtec',
    show_default=True,
    help='数据集选择: mvtec (MVTec AD) 或 visa (VisA)'
)
@click.option(
    '--skip_inference',
    is_flag=True,
    default=False,
    help='跳过模型推理和异常热力图可视化，仅生成 PCA掩模 / Perlin掩模 / 特征图 / 数据增强图'
)
@click.option(
    '--num_samples',
    type=int,
    default=4,
    show_default=True,
    help='随机抽取的测试样本数量（优先抽取异常样本）'
)
@click.option(
    '--query_seed',
    type=int,
    default=42,
    show_default=True,
    help='随机抽取测试图片的种子，固定后可复现每次 heatmap 的 query 图片'
)
@click.option(
    '--compare_seed',
    type=int,
    default=None,
    help='对比模式的第二个 shot_seed。指定后生成 N×4 对比热力图 (主模型 vs 对比模型)'
)
@click.option(
    '--compare_k_shot',
    type=int,
    default=None,
    help='对比模式的 K-Shot 值（与主模型不同的 K）。需配合 --compare_shot_seed 使用'
)
@click.option(
    '--compare_shot_seed',
    type=int,
    default=None,
    help='对比模式的 shot_seed（默认与主模型 --shot_seed 相同）'
)
def main(categories, k_shot, shot_seed, dataset, skip_inference, num_samples, query_seed, compare_seed, compare_k_shot, compare_shot_seed):
    categories = categories.strip().split()
    print(f"处理类别: {categories}")
    print(f"数据集: {dataset}")
    if k_shot is not None:
        print(f"少样本模式: K={k_shot}, seed={shot_seed}")
    if skip_inference:
        print("跳过模型推理（--skip_inference），仅生成分析可视化")
    print(f"Query seed: {query_seed}")
    # 对比模式触发条件: compare_k_shot 或 compare_seed 任一指定
    compare_enabled = (compare_k_shot is not None) or (compare_seed is not None)
    if compare_enabled:
        cmp_k = compare_k_shot if compare_k_shot is not None else k_shot
        cmp_s = compare_shot_seed if compare_shot_seed is not None else (compare_seed if compare_seed is not None else shot_seed)
        print(f"对比模式: K={cmp_k}, seed={cmp_s}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 全局配置加载 ----
    cfg = load_config("config.toml")
    paths = get_paths(cfg)
    category_pca_thresholds = get_category_pca_thresholds(cfg)
    category_pca_border_thresholds = get_category_pca_border_thresholds(cfg)

    base_dir = (paths.get("visa_base_dir", paths["mvtec_base_dir"])
                if dataset == "visa" else paths["mvtec_base_dir"])
    ckpt_dir = paths["ckpt_dir"]
    log_dir = paths["log_dir"]
    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    dinov2_model_dir = paths["dinov2_model_dir"]

    # ---- 逐类别处理 ----
    for current_atype in categories:
        # 路径
        if k_shot is not None:
            ckpt_path = os.path.join(
                ckpt_dir, current_atype,
                f"{current_atype}_k{k_shot}_s{shot_seed}_best_ckpt.pth")
            vis_suffix = f"_k{k_shot}_s{shot_seed}"
        else:
            ckpt_path = os.path.join(
                ckpt_dir, current_atype,
                f"{current_atype}_best_ckpt.pth")
            vis_suffix = ""

        clean_GPU_Cache()

        # 日志
        cat_log_dir = os.path.join(log_dir, dataset, current_atype)
        os.makedirs(cat_log_dir, exist_ok=True)
        logger = setup_logger(
            current_atype, cat_log_dir, logging.DEBUG, log_console=False)
        logger.info(f"Processing category: {current_atype}")

        # 配置
        config = build_model_config(cfg, str(device))
        if current_atype in category_pca_thresholds:
            config.pca_threshold = category_pca_thresholds[current_atype]
            logger.info(
                f"Category-specific PCA threshold for {current_atype}: "
                f"{config.pca_threshold}")
        if current_atype in category_pca_border_thresholds:
            config.pca_border = category_pca_border_thresholds[current_atype]
            logger.info(
                f"Category-specific PCA border threshold for {current_atype}: "
                f"{config.pca_border}")

        # 数据增强控制 — 4 种独立增强
        def _enabled(cat_list):
            if k_shot is None:
                return False
            if cat_list is None:
                return False
            return current_atype in cat_list

        do_flip = _enabled(config.flip_categories)
        do_rotate = _enabled(config.rotate_categories)
        do_translate = _enabled(config.translate_categories)
        do_color_jitter = _enabled(config.color_jitter_categories)
        logger.info(f"Augmentation: flip={do_flip}, rotate={do_rotate}, "
                    f"translate={do_translate}, color_jitter={do_color_jitter}")

        # DataLoader
        train_transform, test_transform, gt_transform = get_transform(
            size=config.target_size, isize=config.target_size,
            flip=do_flip, rotate=do_rotate,
            translate=do_translate, color_jitter=do_color_jitter,
        )
        # DataLoader（Facade 统一入口）
        train_loader, test_loader = get_dataloader(
            root_dir=base_dir,
            category=current_atype,
            dataset_type=dataset,
            train_transform=train_transform,
            test_transform=test_transform,
            gt_transform=gt_transform,
            batch_size=config.batch_size,
            num_workers=4,
            k_shot=k_shot,
            shot_seed=shot_seed,
        )

        # 检测器
        detector = DINOv2AnomalyDetector(
            model_path=dinov2_model_dir,
            config=config,
            logger=logger
        )
        detector.set_category(current_atype)

        # 加载 checkpoint（仅推理模式需要）
        if not skip_inference:
            if os.path.exists(ckpt_path):
                epoch, scores, _, _ = detector.load(ckpt_path)
                logger.info(
                    f"Loaded checkpoint from epoch {epoch}, scores: {scores}")
            else:
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue

        # ---- 对比检测器 (可选) ----
        compare_detector = None
        compare_label = ""
        if compare_enabled and not skip_inference:
            cmp_k = compare_k_shot if compare_k_shot is not None else k_shot
            cmp_s = compare_shot_seed if compare_shot_seed is not None else (compare_seed if compare_seed is not None else shot_seed)
            compare_ckpt_path = os.path.join(
                ckpt_dir, current_atype,
                f"{current_atype}_k{cmp_k}_s{cmp_s}_best_ckpt.pth")
            if not os.path.exists(compare_ckpt_path):
                logger.warning(
                    f"Compare checkpoint not found: {compare_ckpt_path}, "
                    f"skipping comparison")
            else:
                compare_detector = DINOv2AnomalyDetector(
                    model_path=dinov2_model_dir,
                    config=config,
                    logger=logger
                )
                compare_detector.set_category(current_atype)
                epoch_b, scores_b, _, _ = compare_detector.load(compare_ckpt_path)
                logger.info(
                    f"Loaded compare checkpoint (K={cmp_k}, seed={cmp_s}) "
                    f"from epoch {epoch_b}, scores: {scores_b}")
                compare_label = f"K{cmp_k}_s{cmp_s}"

        # ---- 委托给 CategoryVisualizer ----
        viz = CategoryVisualizer(
            atype=current_atype,
            vis_suffix=vis_suffix,
            detector=detector,
            config=config,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            logger=logger,
            do_flip=do_flip,
            do_rotate=do_rotate,
            do_translate=do_translate,
            do_color_jitter=do_color_jitter,
            k_shot=k_shot,
            num_samples=num_samples,
            skip_inference=skip_inference,
            query_seed=query_seed,
            compare_detector=compare_detector,
            compare_label=compare_label,
        )
        viz.run_all()


if __name__ == "__main__":
    main()
