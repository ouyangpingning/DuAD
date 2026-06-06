"""
数据集统一抽象层（Facade Pattern）。

新增数据集只需：
1. 在 dataset/ 下添加新文件（如 dataset/bean.py）
2. 在此 __init__.py 中导出对应的 get_xxx_dataloader
3. 在 _LOADER_MAP 中注册映射
4. 调用方统一使用 from dataset import get_dataloader
"""

from .mvtec import get_mvtec_dataloader, get_transform, RandomRotationReplicate, RandomTranslationReplicate, MvTecDataset
from .visa import get_visa_dataloader, VisADataset


# ── 数据集注册表 ──────────────────────────────────────────────
# key → (factory_function, extra_defaults)
_LOADER_MAP = {
    "mvtec": (get_mvtec_dataloader, {}),
    "visa":  (get_visa_dataloader,  {"csv_name": "1cls"}),
}


def get_dataloader(
    root_dir: str,
    category: str,
    dataset_type: str = "mvtec",
    *,
    train_transform=None,
    test_transform=None,
    gt_transform=None,
    batch_size: int = 8,
    num_workers: int = 0,
    k_shot: int = None,
    shot_seed: int = 0,
    **extra_kwargs,
):
    """
    统一的数据加载器入口（Facade）。

    Args:
        root_dir:       数据集根目录
        category:       类别名称
        dataset_type:   数据集标识: "mvtec" | "visa" | 未来扩展
        train_transform: 训练图像 transform
        test_transform:  测试图像 transform
        gt_transform:    真值掩码 transform
        batch_size:      批大小
        num_workers:     DataLoader 工作进程数
        k_shot:          少样本 K 值，None = 全样本
        shot_seed:       少样本采样种子
        **extra_kwargs:  传递给具体 loader 的额外参数 (如 csv_name)

    Returns:
        (train_loader, test_loader)
    """
    if dataset_type not in _LOADER_MAP:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Available: {list(_LOADER_MAP.keys())}"
        )

    factory, defaults = _LOADER_MAP[dataset_type]
    kwargs = {**defaults, **extra_kwargs}

    return factory(
        root_dir=root_dir,
        category=category,
        train_transform=train_transform,
        test_transform=test_transform,
        gt_transform=gt_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        k_shot=k_shot,
        shot_seed=shot_seed,
        **kwargs,
    )
