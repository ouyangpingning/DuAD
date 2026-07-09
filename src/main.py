# main代码全部进行重构
from commen_import import *
from dataset import get_dataloader, get_transform
from DuAD import DINOv2AnomalyDetector, ModelConfig
from utils import setup_logger, set_seed, clean_GPU_Cache
from config import load_config, build_model_config, get_category_pca_thresholds, get_category_pca_border_thresholds, get_paths
import click

# 主要训练函数
def train_category(
    atype: str,
    base_dir: str,
    ckpt_dir: str,
    log_dir: str,
    dinov2_model_dir: str,
    config: ModelConfig,
    device: torch.device,
    k_shot: int = None,
    shot_seed: int = 0,
    category_pca_thresholds: dict = None,
    category_pca_border_thresholds: dict = None,
    dataset_type: str = "mvtec",
) -> dict:
    """
    训练单个类别

    Returns:
        dict: 包含最佳分数、最佳epoch、模型路径等信息
    """
    # 更新 config 的 device 确保与传入的 device 一致
    config.device = str(device)

    # 设置日志（按类别分目录，全样本/少样本不同 seed 独立命名）
    cat_log_dir = os.path.join(log_dir, dataset_type, atype)
    os.makedirs(cat_log_dir, exist_ok=True)
    if k_shot is not None:
        log_name = f"{atype}_k{k_shot}_s{shot_seed}{config.ablation_tag}"
    else:
        log_name = f"{atype}{config.ablation_tag}"
    logger = setup_logger(log_name, cat_log_dir, logging.DEBUG, log_console=False)
    logger.info(f"{'='*60}")
    logger.info(f"Start training category: {atype}")
    logger.info(f"Device: {device}")
    if k_shot is not None:
        logger.info(f"Few-shot mode: K={k_shot}, seed={shot_seed}")
    logger.info(f"{'='*60}")
    
    # 保存全局默认值，离开前恢复，防止跨类别参数泄漏
    _default_pca_threshold = config.pca_threshold
    _default_pca_border = config.pca_border

    if category_pca_thresholds and atype in category_pca_thresholds:
        config.pca_threshold = category_pca_thresholds[atype]
        logger.info(f"Category-specific PCA threshold for {atype}: {config.pca_threshold}")
    if category_pca_border_thresholds and atype in category_pca_border_thresholds:
        config.pca_border = category_pca_border_thresholds[atype]
        logger.info(f"Category-specific PCA border threshold for {atype}: {config.pca_border}")
    
    # 设置训练随机种子
    set_seed(0)
    
    # 获取数据增强器 — 4 种独立增强，少样本时按类别启用
    def _enabled(cat_list):
        if k_shot is None:
            return False
        if cat_list is None:
            return False
        return atype in cat_list

    enable_flip = _enabled(config.flip_categories)
    enable_rotate = _enabled(config.rotate_categories)
    enable_translate = _enabled(config.translate_categories)
    enable_color_jitter = _enabled(config.color_jitter_categories)
    logger.info(f"Augmentation: flip={enable_flip}, rotate={enable_rotate}, "
                f"translate={enable_translate}, color_jitter={enable_color_jitter}")

    train_transform, test_transform, gt_transform = get_transform(
        size=config.target_size,
        isize=config.target_size,
        flip=enable_flip,
        rotate=enable_rotate,
        translate=enable_translate,
        color_jitter=enable_color_jitter,
    )
    # 训练和测试数据加载器（Facade 统一入口）
    train_loader, test_loader = get_dataloader(
        root_dir=base_dir,
        category=atype,
        dataset_type=dataset_type,
        train_transform=train_transform,
        test_transform=test_transform,
        gt_transform=gt_transform,
        batch_size=config.batch_size,
        num_workers=4,
        k_shot=k_shot,
        shot_seed=shot_seed,
    )
    
    # 初始化模型
    model = DINOv2AnomalyDetector(
        model_path=dinov2_model_dir,
        config=config,
        logger=logger
    )
    
    # 设置好当前的类别
    model.set_category(atype)

    # 定义检查点路径（按类别分目录，少样本模式下带 K/seed 标识）
    cat_ckpt_dir = os.path.join(ckpt_dir, atype)
    os.makedirs(cat_ckpt_dir, exist_ok=True)
    if k_shot is not None:
        best_ckpt_path = os.path.join(cat_ckpt_dir, f"{atype}_k{k_shot}_s{shot_seed}{config.ablation_tag}_best_ckpt.pth")
    else:
        best_ckpt_path = os.path.join(cat_ckpt_dir, f"{atype}{config.ablation_tag}_best_ckpt.pth")

    # 训练 PCA Student（每次按需训练，不持久化）
    model.train_pca_student(train_loader)

    # 最佳分数追踪
    best_score = {
        'image_auroc': 0.0,
        'pixel_auroc': 0.0,
        'image_ap': 0.0,
        'image_f1': 0.0,
        'pixel_ap': 0.0,
        'pixel_f1': 0.0,
        'pixel_pro': 0.0
    }
    best_epoch = -1

    for epoch in range(config.meta_epochs):
        logger.info(50 * "=" + f" Meta Epoch: {epoch}/{config.meta_epochs} " + 50 * "=")
        
        # === 训练阶段 ===
        # 训练一个 meta_epoch（内部包含 gan_epochs 次迭代）
        train_metrics = model.fit(train_loader)
        logger.info(f"  Train Summary - loss: {train_metrics['loss']:.4f}, "
                   f"p_true: {train_metrics['p_true']:.3f}, "
                   f"p_fake: {train_metrics['p_fake']:.3f}")
        
        # === 评估阶段（快速模式：只算 AUROC）===
        scores, masks, labels_gt, masks_gt = model.predict(test_loader, aggregation="max") 
        eval_metrics = model.evaluate(scores, masks, labels_gt, masks_gt, compute_full_metrics=False)
        
        current_score = {
            'image_auroc': eval_metrics['image_auroc'],
            'pixel_auroc': eval_metrics['pixel_auroc'],
        }
        
        logger.info(f"  Eval - Image AUROC: {current_score['image_auroc']:.4f}, "
                   f"Pixel AUROC: {current_score['pixel_auroc']:.4f}")
        
        # === 保存阶段 ===
        # 预热期: 前 warmup_epochs 个 meta-epoch 跳过 best checkpoint 选择,
        # 避免随机初始化参数主导早期评分 (少样本下尤其明显)
        warmup_epochs = getattr(config, 'warmup_epochs', 5)
        if epoch < warmup_epochs:
            logger.info(f"  [Warmup] epoch {epoch+1}/{warmup_epochs}, 跳过 best 检查")
        else:
            is_best = False
            # macaroni2 使用组合分数 (0.5×Image + 0.5×Pixel)，其他类别以 Image AUROC 为主
            if atype == 'macaroni2':
                current_combined = 0.5 * current_score['image_auroc'] + 0.5 * current_score['pixel_auroc']
                best_combined = 0.5 * best_score['image_auroc'] + 0.5 * best_score['pixel_auroc']
                if current_combined > best_combined:
                    is_best = True
            else:
                if current_score['image_auroc'] > best_score['image_auroc']:
                    is_best = True
                elif (current_score['image_auroc'] == best_score['image_auroc'] and
                      current_score['pixel_auroc'] > best_score['pixel_auroc']):
                    is_best = True

            if is_best:
                best_score = current_score.copy()
                best_epoch = epoch
                model.save(best_ckpt_path, epoch=epoch, scores=best_score)

                logger.info('@' * 50)
                logger.info(f"NEW BEST! Epoch: {epoch+1}")
                logger.info(f"  Image AUROC: {best_score['image_auroc']:.4f}")
                logger.info(f"  Pixel AUROC: {best_score['pixel_auroc']:.4f}")
                if atype == 'macaroni2':
                    combined = 0.5 * best_score['image_auroc'] + 0.5 * best_score['pixel_auroc']
                    logger.info(f"  Combined (0.5I+0.5P): {combined:.4f}")
                logger.info('@' * 50)

    # === 最终完整评估（加载 best checkpoint 计算全部指标）===
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading best checkpoint for full evaluation...")
    model.load(best_ckpt_path)
    scores, masks, labels_gt, masks_gt = model.predict(test_loader, aggregation="max")
    full_metrics = model.evaluate(scores, masks, labels_gt, masks_gt, compute_full_metrics=True)
    
    best_score_full = {
        'image_auroc': full_metrics['image_auroc'],
        'pixel_auroc': full_metrics['pixel_auroc'],
        'image_ap': full_metrics.get('image_ap', 0.0),
        'image_f1': full_metrics.get('image_f1', 0.0),
        'pixel_ap': full_metrics.get('pixel_ap', 0.0),
        'pixel_f1': full_metrics.get('pixel_f1', 0.0),
        'pixel_pro': full_metrics.get('pixel_pro', 0.0),
    }
    
    # 训练完成总结
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Completed for {atype}")
    logger.info(f"Best Epoch: {best_epoch+1}")
    logger.info(f"Best Image AUROC: {best_score['image_auroc']:.4f}")
    logger.info(f"Best Pixel AUROC: {best_score['pixel_auroc']:.4f}")
    logger.info(f"Full Evaluation on Best Model:")
    logger.info(f"  Image AUROC: {best_score_full['image_auroc']:.4f}")
    logger.info(f"  Image AP:    {best_score_full['image_ap']:.4f}")
    logger.info(f"  Image F1:    {best_score_full['image_f1']:.4f}")
    logger.info(f"  Pixel AUROC: {best_score_full['pixel_auroc']:.4f}")
    logger.info(f"  Pixel AP:    {best_score_full['pixel_ap']:.4f}")
    logger.info(f"  Pixel F1:    {best_score_full['pixel_f1']:.4f}")
    logger.info(f"  Pixel PRO:   {best_score_full['pixel_pro']:.4f}")
    logger.info(f"Best model saved to: {best_ckpt_path}")
    logger.info(f"{'='*60}")

    # 恢复全局默认值，防止泄漏到下一个类别
    config.pca_threshold = _default_pca_threshold
    config.pca_border = _default_pca_border

    return {
        'category': atype,
        'best_epoch': best_epoch,
        'best_score': best_score_full,
        'best_ckpt_path': best_ckpt_path
    }


@click.command()
@click.option(
    '--categories',
    type=str,
    default="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper",
    show_default=True,
    help='要训练的类别列表，空格分隔，例如 "pill screw toothbrush transistor wood"'
)
@click.option(
    '--k_shot',
    type=int,
    default=None,
    help='少样本数量，None表示使用全部训练样本。例如 --k_shot 4'
)
@click.option(
    '--shot_seed',
    type=int,
    default=0,
    help='少样本采样时的随机种子，用于多seed取平均。例如 --shot_seed 42'
)
@click.option(
    '--dataset',
    type=click.Choice(['mvtec', 'visa']),
    default='mvtec',
    show_default=True,
    help='数据集选择: mvtec (MVTec AD) 或 visa (VisA)'
)
# ===== 消融实验 CLI flags =====
@click.option(
    '--no_pca_mask',
    is_flag=True,
    default=False,
    help='消融: 关闭 PCA 掩模 (自动也关闭 Perlin，回退到单分支 Hinge)'
)
@click.option(
    '--no_perlin_mask',
    is_flag=True,
    default=False,
    help='消融: 关闭 Perlin 掩模 (保留 PCA，单分支 Hinge)'
)
@click.option(
    '--no_augment',
    is_flag=True,
    default=False,
    help='消融: 关闭所有数据增强'
)
@click.option(
    '--aggregation',
    type=click.Choice(['neighborhood', 'channel_concat']),
    default=None,
    help='特征聚合方式 (消融实验): neighborhood (默认, 邻域聚合) 或 channel_concat (通道拼接)'
)
@click.option(
    '--use_pca_student',
    is_flag=True,
    default=False,
    help='消融: 使用 PCA Student MLP 生成掩模 (替代 SVD)'
)
def main(categories, k_shot, shot_seed, dataset, no_pca_mask, no_perlin_mask, no_augment, aggregation, use_pca_student):
    """主函数"""
    # 将 click 返回的字符串按空格分割为列表
    categories = categories.strip().split()
    
    # 设备设置
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 类别列表
    # all_categories = ["bottle" ,"cable" ,"capsule" ,"carpet" , "grid" , "hazelnut" , "leather", "metal_nut"  ,"pill", "screw" , "tile" , "toothbrush",  "transistor",  "wood",  "zipper"]
    print(f"本次训练类别: {categories}")
    print(f"数据集: {dataset}")
    if k_shot is not None:
        print(f"少样本模式: K={k_shot}, seed={shot_seed}")



    # 从 config.toml 加载统一参数
    cfg = load_config("config.toml")
    paths = get_paths(cfg)
    # 根据数据集类型选择对应的数据根目录
    if dataset == "visa":
        base_dir = paths.get("visa_base_dir", paths["mvtec_base_dir"])
    else:
        base_dir = paths["mvtec_base_dir"]
    ckpt_dir = paths["ckpt_dir"]
    log_dir = paths["log_dir"]
    dinov2_model_dir = paths["dinov2_model_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)

    config = build_model_config(cfg, str(device))
    category_pca_thresholds = get_category_pca_thresholds(cfg)
    category_pca_border_thresholds = get_category_pca_border_thresholds(cfg)

    # ===== 消融实验 CLI override =====
    ablation_tags = []
    if no_pca_mask:
        config.use_pca_mask = False
        config.use_perlin_mask = False  # Perlin 依赖 PCA，一起关掉
        ablation_tags.append("noPCA")
        print("[ABLATION] PCA mask DISABLED (also disabling Perlin)")
    if no_perlin_mask:
        config.use_perlin_mask = False
        ablation_tags.append("noPerlin")
        print("[ABLATION] Perlin mask DISABLED (PCA branch only)")
    if no_augment:
        config.flip_categories = []
        config.rotate_categories = []
        config.translate_categories = []
        config.color_jitter_categories = []
        ablation_tags.append("noAug")
        print("[ABLATION] All data augmentations DISABLED")
    if ablation_tags:
        config.ablation_tag = "_" + "_".join(ablation_tags)
        print(f"[ABLATION] Active tags: {', '.join(ablation_tags)}")
    if aggregation is not None:
        config.aggregation_type = aggregation
        config.ablation_tag += f"_agg-{aggregation}" if config.ablation_tag else f"_agg-{aggregation}"
        print(f"[ABLATION] Aggregation type: {aggregation}")
    if use_pca_student:
        config.use_pca_student = True
        config.ablation_tag += "_pcaStudent" if config.ablation_tag else "_pcaStudent"
        print("[ABLATION] PCA Student ENABLED (MLP mask instead of SVD)")
    if ablation_tags or aggregation is not None or use_pca_student:
        print(f"[ABLATION] Checkpoint tag: {config.ablation_tag}")
    # ================================

    # 记录总体结果
    all_results = []
    
    # 遍历所有类别
    for atype in categories:
        # 清理GPU缓存
        clean_GPU_Cache()

        # 训练当前类别
        result = train_category(
            atype=atype,
            base_dir=base_dir,
            ckpt_dir=ckpt_dir,
            log_dir=log_dir,
            dinov2_model_dir=dinov2_model_dir,
            config=config,
            device=device,
            k_shot=k_shot,
            shot_seed=shot_seed,
            category_pca_thresholds=category_pca_thresholds,
            category_pca_border_thresholds=category_pca_border_thresholds,
            dataset_type=dataset,
        )
        
        all_results.append(result)
    
    # 打印总体总结
    print(f"\n{'='*70}")
    print("ALL CATEGORIES TRAINING SUMMARY")
    print(f"{'='*70}")
    for res in all_results:
        print(f"\nCategory: {res['category']}")
        print(f"  Best Epoch: {res['best_epoch']+1}")
        print(f"  Image AUROC: {res['best_score']['image_auroc']:.4f}")
        print(f"  Image AP:    {res['best_score']['image_ap']:.4f}")
        print(f"  Image F1:    {res['best_score']['image_f1']:.4f}")
        print(f"  Pixel AUROC: {res['best_score']['pixel_auroc']:.4f}")
        print(f"  Pixel AP:    {res['best_score']['pixel_ap']:.4f}")
        print(f"  Pixel F1:    {res['best_score']['pixel_f1']:.4f}")
        print(f"  Pixel PRO:   {res['best_score']['pixel_pro']:.4f}")
        print(f"  Model: {res['best_ckpt_path']}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()