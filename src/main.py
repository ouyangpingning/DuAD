# main代码全部进行重构
from commen_import import *
from dataset import get_mvtec_dataloader, get_transform
from myAD import DINOv2AnomalyDetector, ModelConfig
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
) -> dict:
    """
    训练单个类别

    Returns:
        dict: 包含最佳分数、最佳epoch、模型路径等信息
    """
    # 更新 config 的 device 确保与传入的 device 一致
    config.device = str(device)

    # 设置日志（按类别分目录，全样本/少样本不同 seed 独立命名）
    cat_log_dir = os.path.join(log_dir, atype)
    os.makedirs(cat_log_dir, exist_ok=True)
    if k_shot is not None:
        log_name = f"{atype}_k{k_shot}_s{shot_seed}"
    else:
        log_name = atype
    logger = setup_logger(log_name, cat_log_dir, logging.DEBUG, log_console=False)
    logger.info(f"{'='*60}")
    logger.info(f"Start training category: {atype}")
    logger.info(f"Device: {device}")
    if k_shot is not None:
        logger.info(f"Few-shot mode: K={k_shot}, seed={shot_seed}")
    logger.info(f"{'='*60}")
    
    if category_pca_thresholds and atype in category_pca_thresholds:
        config.pca_threshold = category_pca_thresholds[atype]
        logger.info(f"Category-specific PCA threshold for {atype}: {config.pca_threshold}")
    if category_pca_border_thresholds and atype in category_pca_border_thresholds:
        config.pca_border = category_pca_border_thresholds[atype]
        logger.info(f"Category-specific PCA border threshold for {atype}: {config.pca_border}")
    
    # 设置训练随机种子
    set_seed(0)
    
    # 获取数据增强器（少样本时启用随机翻转、旋转）
    # augment_categories 控制哪些类别启用图像级增强，None 表示全部启用
    enable_augment = False
    if k_shot is not None:
        if config.augment_categories is None:
            enable_augment = True
        elif atype in config.augment_categories:
            enable_augment = True
    logger.info(f"Image augmentation: {enable_augment}")

    # 颜色数据增强（用于 toothbrush 等多颜色类别，解决少样本颜色偏差）
    enable_color_augment = False
    if k_shot is not None:
        if config.color_augment_categories is not None:
            enable_color_augment = atype in config.color_augment_categories
    logger.info(f"Color augmentation: {enable_color_augment}")

    train_transform, test_transform, gt_transform = get_transform(
        size=config.target_size,
        isize=config.target_size,
        augment=enable_augment,
        color_augment=enable_color_augment,
    )
    # 训练和测试数据加载器
    train_loader, test_loader = get_mvtec_dataloader(
        root_dir=base_dir, # 数据地址
        Atype=atype,
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
        best_ckpt_path = os.path.join(cat_ckpt_dir, f"{atype}_k{k_shot}_s{shot_seed}_best_ckpt.pth")
        # PCA Student 与 seed 无关 — 同 K 值所有 seed 共享
        pca_student_path = os.path.join(cat_ckpt_dir, f"{atype}_k{k_shot}_pca_student_best.pth")
    else:
        best_ckpt_path = os.path.join(cat_ckpt_dir, f"{atype}_best_ckpt.pth")
        pca_student_path = os.path.join(cat_ckpt_dir, f"{atype}_pca_student_best.pth")

    # 训练/加载 PCA Student — 文件锁协调并行 tmux，断点续训时从 latest_ckpt 恢复
    if model.load_pca_student(pca_student_path):
        logger.info(f"Loaded existing PCA Student from {pca_student_path}, skipping training.")
    else:
        import time as _time
        pca_lock_path = pca_student_path + ".lock"
        try:
            _fd = os.open(pca_lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(_fd)
            logger.info("PCA Student lock acquired, training...")
            model.train_pca_student(train_loader)
            model.save_pca_student(pca_student_path)
            os.remove(pca_lock_path)
            logger.info("PCA Student lock released.")
        except FileExistsError:
            logger.info("Another process is training PCA Student, waiting (timeout 600s)...")
            for _ in range(600):
                _time.sleep(1)
                if model.load_pca_student(pca_student_path):
                    logger.info("PCA Student loaded after waiting.")
                    break
            else:
                logger.warning("Timeout waiting for PCA Student, training locally.")
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
        # 检查是否为最佳
        is_best = False
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

def main(categories, k_shot, shot_seed):
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
    if k_shot is not None:
        print(f"少样本模式: K={k_shot}, seed={shot_seed}")


    
    # 从 config.toml 加载统一参数
    cfg = load_config("config.toml")
    paths = get_paths(cfg)
    base_dir = paths["base_dir"]
    ckpt_dir = paths["ckpt_dir"]
    log_dir = paths["log_dir"]
    dinov2_model_dir = paths["dinov2_model_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)

    config = build_model_config(cfg, str(device))
    category_pca_thresholds = get_category_pca_thresholds(cfg)
    category_pca_border_thresholds = get_category_pca_border_thresholds(cfg)

    # 少样本时固定噪声强度，不使用退火
    if k_shot is not None:
        config.use_noise_annealing = False

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