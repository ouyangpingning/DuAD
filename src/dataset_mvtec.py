from commen_import import *
import cv2


class RandomRotationReplicate:
    """随机旋转变换，使用 OpenCV BORDER_REPLICATE 避免黑边。"""
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        # img: torch tensor (C, H, W), uint8 [0, 255]
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())
        if angle == 0:
            return img

        img_np = img.permute(1, 2, 0).cpu().numpy()
        h, w = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img_np, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return torch.from_numpy(np.ascontiguousarray(rotated)).permute(2, 0, 1)


# 数据增强
def get_transform(size, isize, mean=None, std=None, augment=False, color_augment=False):
    mean_train = [0.485, 0.456, 0.406] if mean is None else mean
    std_train = [0.229, 0.224, 0.225] if std is None else std

    if augment:
        extra_aug = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            RandomRotationReplicate(degrees=180),
        ]
    else:
        extra_aug = []

    # 颜色增强：解决多颜色类别（如 toothbrush）少样本时的颜色偏差问题
    if color_augment:
        extra_aug = [v2.ColorJitter(hue=0.3)] + extra_aug

    train_transforms = v2.Compose(
        [
            v2.ToImage(), # 转换为张量仅在输入为PIL时有用
            v2.Resize((size, size)),
            v2.CenterCrop(isize),
        ] + extra_aug + [
            v2.ToDtype(torch.float32, scale=True), # totensor
            v2.Normalize(mean=mean_train, std=std_train),
        ]
    )
    test_transforms = v2.Compose(
        [   
            v2.ToImage(), # 转换为张量仅在输入为PIL时有用
            v2.Resize((size, size)),
            v2.CenterCrop(isize),
            v2.ToDtype(torch.float32, scale=True), # totensor
            v2.Normalize(mean=mean_train, std=std_train),
        ]
    )
    gt_transformes = v2.Compose(
        [   
            v2.ToImage(),
            v2.Resize((size, size)),
            v2.CenterCrop(isize),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return train_transforms, test_transforms, gt_transformes


class MvTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, type):
        if type == "train":
            self.img_path = Path(root) / "train"
        else:
            self.img_path = Path(root) / "test"
            self.gt_path = Path(root) / "ground_truth"
        self.transform = transform
        self.gt_transform = gt_transform

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_paths = []
        gt_paths = []
        labels = []
        types = []

        ADtypes = [path for path in self.img_path.iterdir() if path.is_dir()]
        # G:\repository\remote-repository\data\mvtec_anomaly_detection\bottle\test
        for adtype in ADtypes:
            if adtype.name == "good":  # Nomaly
                _img_paths = sorted(adtype.glob("*.png"))
                img_paths.extend(_img_paths)
                gt_paths.extend([0] * len(_img_paths))
                labels.extend([0] * len(_img_paths))
                types.extend(["good"] * len(_img_paths))
            else:  # Anomaly
                _img_paths = sorted(adtype.glob("*.png"))  # 异常图像路径
                _gt_paths = sorted(
                    (self.gt_path / adtype.name).glob("*.png")
                )  # 异常图像掩码路径
                img_paths.extend(_img_paths)
                gt_paths.extend(_gt_paths)
                labels.extend([1] * len(_img_paths))
                types.extend([adtype.name] * len(_img_paths))
        assert len(img_paths) == len(gt_paths)

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
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])  # 正常掩码
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        assert img.size()[1:] == gt.size()[1:]
        return img, gt, label, adtype  # 这里img是图片、gt是掩码、label是标签0/1、adtype是异常类型(str)
    

def get_mvtec_dataloader(root_dir:str = "/run/media/lxb/Soft/repository/remote-repository/data/mvtec_anomaly_detection",
                         Atype:str="bottle",
                         train_transform=None,
                         test_transform=None,
                         gt_transform=None,
                         batch_size=8,
                         num_workers=0,
                         k_shot: int = None,
                         shot_seed: int = 0):
    """
        root_dir: mvtec数据集的根目录 \n
        Atype: mvtec数据集的类别 \n
        data_transform: 数据增强 \n
        gt_transform: 掩码增强 \n
        batch_size: 批量大小 \n
        num_workers: 数据加载的线程数 \n
        k_shot: 少样本数量, None表示使用全部训练样本 \n
        shot_seed: 少样本采样时的随机种子

        return : train_transform_dataloader, test_transform_dataloader
    """
    if train_transform is not None and test_transform is not None and gt_transform is not None: # 如果有数据增强
        train_transform_, test_transform_, gt_transform_ = train_transform, test_transform, gt_transform
    else: # 如果没有数据增强
        train_transform_, test_transform_, gt_transform_ = get_transform(244,244) # 使用默认的数据增强

    # 正常样本的dataset
    MvTec_nomal = MvTecDataset(
        f"{root_dir}/{Atype}",
        train_transform_,
        gt_transform_,
        "train",
    )
    # 异常样本的dataset
    MvTec_anomal = MvTecDataset(
        f"{root_dir}/{Atype}",
        test_transform_,
        gt_transform_,
        "test",
    )

    # 少样本设置：从训练集中随机采样 K 张正常图像
    if k_shot is not None:
        rng = random.Random(shot_seed)
        indices = rng.sample(range(len(MvTec_nomal)), min(k_shot, len(MvTec_nomal)))
        MvTec_nomal = Subset(MvTec_nomal, indices)

    # 数据加载
    if k_shot is not None:
        # 少样本时使用带放回采样，保证每个 batch 都填满 batch_size
        sampler = torch.utils.data.RandomSampler(MvTec_nomal, replacement=True, num_samples=32)
        train_transform_dataloader = DataLoader(MvTec_nomal, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_transform_dataloader = DataLoader(MvTec_nomal, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_transform_dataloader = DataLoader(MvTec_anomal, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_transform_dataloader, test_transform_dataloader
