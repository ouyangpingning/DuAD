from commen_import import *
from utils import clean_GPU_Cache, setup_logger
from dataset import get_mvtec_dataloader, get_transform
from myAD import DINOv2AnomalyDetector, ModelConfig, Visualizer, PCAMaskGenerator
from config import load_config, build_model_config, get_category_pca_thresholds, get_category_pca_border_thresholds, get_paths
from sklearn.decomposition import PCA
import cv2
import click


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
def main(categories, k_shot, shot_seed):
    categories = categories.strip().split()
    print(f"处理类别: {categories}")
    if k_shot is not None:
        print(f"少样本模式: K={k_shot}, seed={shot_seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 从 config.toml 加载统一参数
    cfg = load_config("config.toml")
    paths = get_paths(cfg)
    category_pca_thresholds = get_category_pca_thresholds(cfg)
    category_pca_border_thresholds = get_category_pca_border_thresholds(cfg)
    base_dir = paths["base_dir"]
    ckpt_dir = paths["ckpt_dir"]
    log_dir = paths["log_dir"]
    output_dir = paths["output_dir"]
    dinov2_model_dir = paths["dinov2_model_dir"]

    for current_atype in categories:
        if k_shot is not None:
            ckpt_path = os.path.join(ckpt_dir, current_atype, f"{current_atype}_k{k_shot}_s{shot_seed}_best_ckpt.pth")
            vis_suffix = f"_k{k_shot}_s{shot_seed}"
        else:
            ckpt_path = os.path.join(ckpt_dir, current_atype, f"{current_atype}_best_ckpt.pth")
            vis_suffix = ""

        clean_GPU_Cache()

        cat_log_dir = os.path.join(log_dir, current_atype)
        os.makedirs(cat_log_dir, exist_ok=True)
        logger = setup_logger(current_atype, cat_log_dir, logging.DEBUG, log_console=False)
        logger.info(f"Processing category: {current_atype}")

        config = build_model_config(cfg, str(device))
        target_size = config.target_size
        batch_size = config.batch_size
        augment_categories = config.augment_categories
        color_augment_categories = config.color_augment_categories
        perlin_min = config.perlin_min
        perlin_max = config.perlin_max

        do_augment = (k_shot is not None) and (augment_categories is None or current_atype in augment_categories)
        do_color_augment = (k_shot is not None) and (color_augment_categories is None or current_atype in color_augment_categories)

        train_transform, test_transform, gt_transform = get_transform(size=target_size, isize=target_size, augment=do_augment, color_augment=do_color_augment)
        train_transform_dataloader, test_transform_dataloader = get_mvtec_dataloader(
            root_dir=base_dir,
            Atype=current_atype,
            train_transform=train_transform,
            test_transform=test_transform,
            gt_transform=gt_transform,
            batch_size=batch_size,
            num_workers=4,
            k_shot=k_shot,
            shot_seed=shot_seed,
        )

        # ---- 可视化数据增强效果（仅少样本模式） ----
        if do_augment or do_color_augment:
            logger.info("Visualizing augmented training images...")
            train_iter = iter(train_transform_dataloader)
            aug_images, _, _, _ = next(train_iter)  # [B, C, H, W]
            B = min(aug_images.shape[0], 8)

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            for i in range(B):
                img = aug_images[i].cpu().permute(1, 2, 0).numpy()
                # 反归一化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
                axes[i].set_title(f'Aug #{i+1}')
                axes[i].axis('off')
            for i in range(B, 8):
                axes[i].axis('off')

            plt.suptitle(f'{current_atype} - Augmented Training Images (K={k_shot})', fontsize=14)
            plt.tight_layout()
            os.makedirs(f"{output_dir}/augmented/", exist_ok=True)
            aug_save_path = f"{output_dir}/augmented/{current_atype}{vis_suffix}_augmented.png"
            plt.savefig(aug_save_path, dpi=150)
            plt.close()
            logger.info(f"Augmented image visualization saved to: {aug_save_path}")
        
        if current_atype in category_pca_thresholds:
            config.pca_threshold = category_pca_thresholds[current_atype]
            logger.info(f"Category-specific PCA threshold for {current_atype}: {config.pca_threshold}")
        if current_atype in category_pca_border_thresholds:
            config.pca_border = category_pca_border_thresholds[current_atype]
            logger.info(f"Category-specific PCA border threshold for {current_atype}: {config.pca_border}")

        # 创建检测器
        detector = DINOv2AnomalyDetector(
            model_path=dinov2_model_dir,
            config=config,
            logger=logger
        )
        detector.set_category(current_atype)

        # 预测并可视化所有测试样本
        if os.path.exists(ckpt_path):
            epoch, scores, _, _ = detector.load(ckpt_path)
            logger.info(f"Loaded checkpoint from epoch {epoch}, scores: {scores}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        logger.info("Running prediction...")
        scores, segmentations, labels_gt, masks_gt = detector.predict(test_transform_dataloader)

        segmentations = np.array(segmentations)
        masks_gt = np.array(masks_gt)

        logger.info("Generating visualization...")
        Visualizer.visualize_masks(
            masks=segmentations,
            scores=scores,
            save_path=f"{output_dir}/{current_atype}{vis_suffix}_test.png"
        )
        logger.info(f"Test visualization saved to: {output_dir}/{current_atype}{vis_suffix}_test.png")
        
        # 可视化PCA掩模
        if config.use_pca_mask:
            logger.info("Generating PCA mask visualization...")
            # 从train dataloader获取第一张正常样本
            train_iter = iter(train_transform_dataloader)
            first_train_images, _, _, _ = next(train_iter)
            sample_image = first_train_images[0:1].to(device)  # [1, C, H, W]
            
            # 提取特征并生成PCA掩模
            detector.feature_extractor.eval()
            with torch.no_grad():
                features, (H, W) = detector.feature_extractor(sample_image)
                
                # 创建PCA掩模生成器
                pca_gen = PCAMaskGenerator(
                    threshold=config.pca_threshold,
                    border_ratio=config.pca_border,
                    kernel_size=config.pca_kernel_size,
                    use_gpu=config.pca_use_gpu, # 是否使用GPU加速PCA计算
                    skip_categories=config.pca_skip_categories # 传递跳过类别列表
                )
                # 设置当前类别
                pca_gen.set_category(current_atype)
                
                # 计算第一主成分
                features_np = features.cpu().numpy()
                pca = PCA(n_components=1, svd_solver='randomized')
                first_pc = pca.fit_transform(features_np).squeeze() # 得到投影值
                first_pc_2d = first_pc.reshape(H, W)
                
                # 生成掩模
                mask = pca_gen(features, (H, W))
                mask_2d = mask.cpu().numpy().reshape(H, W)
                
                # 准备图像
                img_np = sample_image[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                
                # 上采样到原始尺寸
                first_pc_up = cv2.resize(first_pc_2d, (target_size, target_size))
                mask_up = cv2.resize(mask_2d.astype(np.float32), (target_size, target_size))
            
            # 使用 Visualizer 可视化PCA掩模
            Visualizer.visualize_pca_mask(
                image=img_np,
                mask=mask_up,
                first_pc=first_pc_up,
                save_path=f"{output_dir}/pca_mask/{current_atype}{vis_suffix}_pca_mask.png"
            )
            logger.info(f"PCA mask visualization saved to: {output_dir}/pca_mask/{current_atype}{vis_suffix}_pca_mask.png")

        # 可视化Perlin掩模
        if config.use_perlin_mask and config.use_pca_mask:
            logger.info("Generating Perlin mask visualization...")
            from perlin import perlin_mask
            
            with torch.no_grad():
                # 上采样PCA掩码到图像分辨率作为Perlin的前景约束
                pca_mask_img = F.interpolate(
                    mask.reshape(1, 1, H, W).float(),
                    size=(target_size, target_size),
                    mode='nearest'
                ).squeeze().cpu().numpy()
                try:
                    perlin_s = perlin_mask(
                        img_shape=(sample_image.shape[1], target_size, target_size),
                        feat_size=H,
                        min=perlin_min,
                        max=perlin_max,
                        mask_fg=pca_mask_img,
                        flag=0
                    )
                    perlin_2d = perlin_s  # [H, W]
                    perlin_up = cv2.resize(perlin_2d.astype(np.float32), (target_size, target_size))
                except Exception as e:
                    logger.warning(f"Perlin mask generation failed: {e}")
                    perlin_2d = np.zeros((H, W), dtype=np.float32)
                    perlin_up = cv2.resize(perlin_2d, (target_size, target_size))
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # 指定黑体，可根据系统替换

                plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
                # 构建可视化
                fig, axes = plt.subplots(1, 4, figsize=(20, 6))
                
                axes[0].imshow(img_np)
                axes[0].set_title('原图')
                axes[0].axis('off')
                
                axes[1].imshow(mask_up, cmap='gray')
                axes[1].set_title('PCA掩模')
                axes[1].axis('off')
                
                axes[2].imshow(perlin_up, cmap='gray')
                axes[2].set_title('Perlin掩模')
                axes[2].axis('off')
                
                axes[3].imshow(img_np)
                overlay = np.zeros((*perlin_up.shape, 4))
                overlay[perlin_up > 0.5] = [0, 1, 0, 0.3]  # 绿色表示Perlin掩码
                axes[3].imshow(overlay)
                axes[3].set_title('Perlin掩模叠加')
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/perlin_mask/{current_atype}{vis_suffix}_perlin_mask.png", dpi=300)
                plt.close()
                logger.info(f"Perlin mask visualization saved to: {output_dir}/perlin_mask/{current_atype}{vis_suffix}_perlin_mask.png")

        # 可视化DINOv2特征图
        logger.info("Generating DINOv2 feature map visualization...")
        train_iter = iter(train_transform_dataloader)
        first_train_images, _, _, _ = next(train_iter)
        sample_image = first_train_images[0:1].to(device)

        detector.feature_extractor.eval()
        with torch.no_grad():
            features, (H, W) = detector.feature_extractor(sample_image)
            feat_map = features.reshape(1, H, W, -1)
            activation = torch.norm(feat_map, p=2, dim=-1).squeeze(0).cpu().numpy()
            activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
            activation_up = cv2.resize(activation, (target_size, target_size))
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

        plt.rcParams['axes.unicode_minus'] = False
        img_np = sample_image[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_np)
        axes[0].set_title('原图')
        axes[0].axis('off')
        axes[1].imshow(activation_up, cmap='jet')
        axes[1].set_title('DINOv2特征激活热力图')
        axes[1].axis('off')
        plt.tight_layout()
        os.makedirs(f"{output_dir}/feature_map/", exist_ok=True)
        plt.savefig(f"{output_dir}/feature_map/{current_atype}{vis_suffix}_feature_map.png", dpi=150)
        plt.close()
        logger.info(f"Feature map visualization saved to: {output_dir}/feature_map/{current_atype}{vis_suffix}_feature_map.png")

        logger.info(f"Category {current_atype} processing completed.\n")


if __name__ == "__main__":
    main()