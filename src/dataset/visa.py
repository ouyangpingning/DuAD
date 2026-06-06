from commen_import import *
from .mvtec import get_transform
import pandas as pd


class VisADataset(Dataset):
    """VisA 数据集加载器。

    VisA 数据集结构:
        {root}/
        ├── split_csv/
        │   ├── 1cls.csv
        │   ├── 2cls_fewshot.csv
        │   └── 2cls_highshot.csv
        ├── candle/
        │   ├── Data/Images/Normal/*.JPG
        │   ├── Data/Images/Anomaly/*.JPG
        │   ├── Data/Masks/Anomaly/*.png
        │   └── image_anno.csv
        └── ...

    CSV 格式: object,split,label,image,mask
    """
    def __init__(self, root, csv_path, category, transform, gt_transform, split):
        """
        Args:
            root:       VisA 数据集根目录 (e.g., /root/siton-tmp/VisA)
            csv_path:   split CSV 文件路径
            category:   类别名称 (e.g., "candle")
            transform:  图像 transform
            gt_transform: 掩码 transform
            split:      "train" 或 "test"
        """
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.category = category
        self.transform = transform
        self.gt_transform = gt_transform
        self.split = split

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_paths = []
        gt_paths = []
        labels = []
        types = []

        df = pd.read_csv(self.csv_path)
        # 过滤出当前类别和 split 的数据
        mask = (df["object"] == self.category) & (df["split"] == self.split)
        df_cat = df[mask]

        for _, row in df_cat.iterrows():
            img_path = self.root / row["image"]
            img_paths.append(str(img_path))

            if row["label"] == "normal":
                gt_paths.append(0)  # 正常样本无掩码
                labels.append(0)
                types.append("good")
            else:
                mask_path = self.root / row["mask"]
                gt_paths.append(str(mask_path))
                labels.append(1)
                types.append("Anomaly")  # VisA 只有一种异常类型

        assert len(img_paths) == len(gt_paths), \
            f"img_paths ({len(img_paths)}) != gt_paths ({len(gt_paths)})"

        return img_paths, gt_paths, labels, types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, gt, label, adtype = (
            self.img_paths[index],
            self.gt_paths[index],
            self.labels[index],
            self.types[index],
        )
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if gt == 0:  # 正常
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
            # VisA 掩模像素值为 1（异常）/ 0（背景），经 ToDtype(scale=True) 后变成 1/255 ≈ 0.004
            # 需要二值化还原为 0/1，与 MVTec 掩模格式保持一致
            gt = (gt > 0).float()
        assert img.size()[1:] == gt.size()[1:], \
            f"img size {img.size()[1:]} != gt size {gt.size()[1:]}"
        return img, gt, label, adtype


def get_visa_dataloader(root_dir: str = "/root/siton-tmp/VisA",
                         category: str = "candle",
                         csv_name: str = "1cls",
                         train_transform=None,
                         test_transform=None,
                         gt_transform=None,
                         batch_size=8,
                         num_workers=0,
                         k_shot: int = None,
                         shot_seed: int = 0):
    """
    Args:
        root_dir:    VisA 数据集根目录
        category:    VisA 数据集的类别
        csv_name:    split CSV 文件名 (不含 .csv 后缀), 默认 "1cls"
        train_transform: 训练数据增强
        test_transform:  测试数据增强
        gt_transform:    掩码增强
        batch_size: 批量大小
        num_workers: 数据加载线程数
        k_shot:     少样本数量, None 表示使用全部训练样本
        shot_seed:  少样本采样随机种子

    Returns:
        train_dataloader, test_dataloader
    """
    if train_transform is not None and test_transform is not None and gt_transform is not None:
        train_transform_, test_transform_, gt_transform_ = train_transform, test_transform, gt_transform
    else:
        train_transform_, test_transform_, gt_transform_ = get_transform(244, 244)

    csv_path = Path(root_dir) / "split_csv" / f"{csv_name}.csv"

    # 正常样本 dataset (train split)
    visa_normal = VisADataset(
        root_dir, csv_path, category,
        train_transform_, gt_transform_, "train",
    )
    # 测试样本 dataset (test split, 包含 normal + anomaly)
    visa_test = VisADataset(
        root_dir, csv_path, category,
        test_transform_, gt_transform_, "test",
    )

    # 少样本设置
    if k_shot is not None:
        rng = random.Random(shot_seed)
        indices = rng.sample(range(len(visa_normal)), min(k_shot, len(visa_normal)))
        visa_normal = Subset(visa_normal, indices)

    # 数据加载
    if k_shot is not None:
        sampler = torch.utils.data.RandomSampler(visa_normal, replacement=True, num_samples=32)
        train_dataloader = DataLoader(visa_normal, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_dataloader = DataLoader(visa_normal, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(visa_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader
