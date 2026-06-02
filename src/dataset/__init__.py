"""
数据集统一抽象层。

新增数据集只需：
1. 在 dataset/ 下添加新文件（如 dataset/bean.py）
2. 在此 __init__.py 中导出对应的 get_xxx_dataloader
3. 调用方统一使用 from dataset import get_xxx_dataloader
"""

from .mvtec import get_mvtec_dataloader, get_transform, RandomRotationReplicate, MvTecDataset
from .visa import get_visa_dataloader, VisADataset
