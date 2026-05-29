"""SimpleNet 训练入口。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commen_import import *
from dataset import get_mvtec_dataloader, get_transform
from utils import setup_logger, set_seed, clean_GPU_Cache
from simplenet.simplenet import SimpleNet, SimpleNetConfig
from simplenet.config import load_config, build_simplenet_config, get_paths
import click


def train_category(
    atype: str,
    base_dir: str,
    ckpt_dir: str,
    log_dir: str,
    config: SimpleNetConfig,
    device: torch.device,
    k_shot: int = None,
    shot_seed: int = 0,
) -> dict:
    config.device = str(device)

    cat_log_dir = os.path.join(log_dir, atype)
    os.makedirs(cat_log_dir, exist_ok=True)
    if k_shot is not None:
        log_name = f"{atype}_k{k_shot}_s{shot_seed}"
    else:
        log_name = atype
    logger = setup_logger(log_name, cat_log_dir, logging.DEBUG, log_console=False)
    logger.info(f"{'=' * 60}")
    logger.info(f"Start training category: {atype} [SimpleNet]")
    logger.info(f"Device: {device}")
    if k_shot is not None:
        logger.info(f"Few-shot mode: K={k_shot}, seed={shot_seed}")
    logger.info(f"{'=' * 60}")

    set_seed(0)

    train_transform, test_transform, gt_transform = get_transform(
        size=config.resize,
        isize=config.isize,
    )
    train_loader, test_loader = get_mvtec_dataloader(
        root_dir=base_dir,
        Atype=atype,
        train_transform=train_transform,
        test_transform=test_transform,
        gt_transform=gt_transform,
        batch_size=config.batch_size,
        num_workers=4,
        k_shot=k_shot,
        shot_seed=shot_seed,
    )

    model = SimpleNet(config=config, logger=logger)

    cat_ckpt_dir = os.path.join(ckpt_dir, atype)
    os.makedirs(cat_ckpt_dir, exist_ok=True)
    if k_shot is not None:
        best_ckpt_path = os.path.join(cat_ckpt_dir, f"{atype}_k{k_shot}_s{shot_seed}_best_ckpt.pth")
    else:
        best_ckpt_path = os.path.join(cat_ckpt_dir, f"{atype}_best_ckpt.pth")

    best_score = {'image_auroc': 0.0, 'pixel_auroc': 0.0}
    best_epoch = -1

    for epoch in range(config.meta_epochs):
        logger.info(50 * "=" + f" Meta Epoch: {epoch}/{config.meta_epochs} " + 50 * "=")

        train_metrics = model.fit(train_loader)
        logger.info(f"  Train Summary - loss: {train_metrics['loss']:.4f}, "
                    f"p_true: {train_metrics['p_true']:.3f}, "
                    f"p_fake: {train_metrics['p_fake']:.3f}")

        scores, masks, labels_gt, masks_gt = model.predict(test_loader)
        eval_metrics = model.evaluate(scores, masks, labels_gt, masks_gt, compute_full_metrics=False)

        current_score = {
            'image_auroc': eval_metrics['image_auroc'],
            'pixel_auroc': eval_metrics['pixel_auroc'],
        }

        logger.info(f"  Eval - Image AUROC: {current_score['image_auroc']:.4f}, "
                    f"Pixel AUROC: {current_score['pixel_auroc']:.4f}")

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
            logger.info(f"NEW BEST! Epoch: {epoch + 1}")
            logger.info(f"  Image AUROC: {best_score['image_auroc']:.4f}")
            logger.info(f"  Pixel AUROC: {best_score['pixel_auroc']:.4f}")
            logger.info('@' * 50)

    # 最终完整评估
    logger.info(f"\n{'=' * 60}")
    logger.info("Loading best checkpoint for full evaluation...")
    model.load(best_ckpt_path)
    scores, masks, labels_gt, masks_gt = model.predict(test_loader)
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

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training Completed for {atype} [SimpleNet]")
    logger.info(f"Best Epoch: {best_epoch + 1}")
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
    logger.info(f"{'=' * 60}")

    return {
        'category': atype,
        'best_epoch': best_epoch,
        'best_score': best_score_full,
        'best_ckpt_path': best_ckpt_path,
    }


@click.command()
@click.option(
    '--categories',
    type=str,
    default="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper",
    show_default=True,
    help='要训练的类别列表，空格分隔',
)
@click.option(
    '--k_shot',
    type=int,
    default=None,
    help='少样本数量，None 表示使用全部训练样本',
)
@click.option(
    '--shot_seed',
    type=int,
    default=0,
    help='少样本采样时的随机种子',
)
def main(categories, k_shot, shot_seed):
    categories = categories.strip().split()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Categories: {categories}")
    if k_shot is not None:
        print(f"Few-shot mode: K={k_shot}, seed={shot_seed}")

    # 从 simplenet 目录下加载配置
    config_path = os.path.join(os.path.dirname(__file__), "config.toml")
    cfg = load_config(config_path)
    paths = get_paths(cfg)
    base_dir = paths["base_dir"]
    ckpt_dir = paths["ckpt_dir"]
    log_dir = paths["log_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    config = build_simplenet_config(cfg, str(device))

    all_results = []

    for atype in categories:
        clean_GPU_Cache()
        result = train_category(
            atype=atype,
            base_dir=base_dir,
            ckpt_dir=ckpt_dir,
            log_dir=log_dir,
            config=config,
            device=device,
            k_shot=k_shot,
            shot_seed=shot_seed,
        )
        all_results.append(result)

    print(f"\n{'=' * 70}")
    print("ALL CATEGORIES TRAINING SUMMARY [SimpleNet]")
    print(f"{'=' * 70}")
    for res in all_results:
        print(f"\nCategory: {res['category']}")
        print(f"  Best Epoch: {res['best_epoch'] + 1}")
        print(f"  Image AUROC: {res['best_score']['image_auroc']:.4f}")
        print(f"  Image AP:    {res['best_score']['image_ap']:.4f}")
        print(f"  Image F1:    {res['best_score']['image_f1']:.4f}")
        print(f"  Pixel AUROC: {res['best_score']['pixel_auroc']:.4f}")
        print(f"  Pixel AP:    {res['best_score']['pixel_ap']:.4f}")
        print(f"  Pixel F1:    {res['best_score']['pixel_f1']:.4f}")
        print(f"  Pixel PRO:   {res['best_score']['pixel_pro']:.4f}")
        print(f"  Model: {res['best_ckpt_path']}")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
