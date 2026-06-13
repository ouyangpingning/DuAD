from commen_import import *
import cv2
import pandas as pd
from skimage import measure


def _safe_roc_auc(y_true, y_score):
    """计算 AUROC，当 y_true 只有一个类别时返回默认值，避免 UndefinedMetricWarning。"""
    classes = np.unique(np.asarray(y_true))
    if len(classes) < 2:
        return 0.5  # 单类时无法定义 AUROC，返回随机水平
    return metrics.roc_auc_score(y_true, y_score)


# 用于模型的评估
def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
    #  模型对每张图像的异常得分,形状：[N]，N为图像数量,模型输出的异常分数（得分越高，异常可能性越大）
    # # 每张图像的真实标签,形状：[N]，与上面一一对应,二元标签：1=异常，0=正常
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    auroc = _safe_roc_auc(
        anomaly_ground_truth_labels,  # 真实标签
        anomaly_prediction_weights # 预测得分
    )
    ap = metrics.average_precision_score(
        anomaly_ground_truth_labels,
        anomaly_prediction_weights
    )
    
    precision, recall, pr_thresholds = metrics.precision_recall_curve(
        anomaly_ground_truth_labels,
        anomaly_prediction_weights
    )
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    f1 = float(np.max(f1_scores)) if len(f1_scores) > 0 else 0.0

    return {
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
    }

def compute_pro(masks, amaps, num_th=200):
    # 处理可能存在的多余维度 (N, 1, H, W) -> (N, H, W)
    masks = np.squeeze(np.array(masks))
    amaps = np.squeeze(np.array(amaps))

    df_rows = []
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df_rows.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th})

    df = pd.DataFrame(df_rows)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    if df["fpr"].max() > 0:
        df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = _safe_roc_auc(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    ap = metrics.average_precision_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, pr_thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    f1 = float(np.max(F1_scores)) if len(F1_scores) > 0 else 0.0
    
    pro = compute_pro(ground_truth_masks, anomaly_segmentations)

    return {
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
        "pro": pro,
    }


def set_seed(seed: int):
    # Python hash 随机性
    os.environ['PYTHONHASHSEED'] = str(seed)
    # # python 内置随机
    random.seed(seed)
    # # numpy 随机
    np.random.seed(seed)
    # PyTorch CPU 随机
    torch.manual_seed(seed)
    # PyTorch CUDA 随机（单/多 GPU）
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # cuDNN 可确定性设置（可能降低性能）
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # 新版 PyTorch 可选的强制确定性（若不支持会抛错）
    try:
        torch.use_deterministic_algorithms(False) # 关闭确定性模式
    except Exception:
        pass

def clean_GPU_Cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()# 清空GPU缓存，释放显存，进行下一次的测试

def init_weight(m):
    """
    初始化权重
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight) # 使用xavier参数初始化
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

def setup_logger(experiment_name, log_dir='logs', level=logging.INFO, log_console:bool=True ,log_file:bool = True):
    """
        experiment_name : 日志的名字 \n
        log_dir : 存放的目录文件夹地址  \n
        level : 日志的等级  \n
        log_console: 是否使用控制台来输出日志 \n
        log_file : 是否使用文件来保存日志 \n
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # 创建logger
    logger = colorlog.getLogger(experiment_name)
    logger.setLevel(level)
    
    # 清除现有处理器（避免重复）
    if logger.handlers:
        logger.handlers.clear()

    if log_console:
        # 控制台处理器
        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(level)
        
        # 简洁格式（用于控制台）
        simple_formatter = colorlog.ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            reset=True,
            style='%'
        )
        # 设置终端处理器格式
        console_handler.setFormatter(simple_formatter)
        # 添加终端处理器到日志对象
        logger.addHandler(console_handler)

    if log_file: # 使用日志
        # 文件处理器 - 完整日志 - 文件日志得使用logging库
        file_handler = logging.FileHandler(
            log_dir / f'{experiment_name}_full.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
            
        # 详细格式（用于文件）
        detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        )

        file_handler.setFormatter(detailed_formatter)
        # 添加处理器
        logger.addHandler(file_handler)

    return logger

def download_dinov2_models(name: str, source: str = 'local', model_pth: str = "../facebookresearch_dinov2_main", pretrained: bool = True):
    """
    下载dinov2模型
    Args:
        name: 模型名称 \n
        source: 模型来源，'local'或'github' \n
        model_pth: 模型路径 \n
        pretrained: 是否加载预训练权重 \n
    Returns:
        pre_model: 下载的模型 \n
    目前支持的模型有：
        'dinov2_vitl14' : DINOv2 Vision Transformer Large 14 \n
        'dinov2_vitl14_reg' : DINOv2 Vision Transformer Large 14 with regularization \n
        'dinov2_vits14_reg' : DINOv2 Vision Transformer Small 14 with regularization \n
    """
    # 下载VITL/14
    if name == 'dinov2_vitl14':
        source = 'github' if source == 'github' else 'local'
        dinov2_vitl14 = torch.hub.load(model_pth, 'dinov2_vitl14',source=source,pretrained = pretrained)
        pre_model = dinov2_vitl14.eval()
    elif name == 'dinov2_vitl14_reg':
        source = 'github' if source == 'github' else 'local'
        dinov2_vitl14_reg = torch.hub.load(model_pth, 'dinov2_vitl14_reg',source=source,pretrained = pretrained)
        pre_model = dinov2_vitl14_reg.eval()
    elif name == 'dinov2_vits14_reg':
        source = 'github' if source == 'github' else 'local'
        dinov2_vitl14_reg = torch.hub.load(model_pth, 'dinov2_vits14_reg',source=source,pretrained = pretrained)
        pre_model = dinov2_vitl14_reg.eval()
    # print(pre_model)
    return pre_model

def _embed_legacy(features,layers:list,patchsize:int, stride:int,target_patches:int, target_dim:int ,output_size:int):
    """
        features: 所有features的集合 \n
        layers : 按顺序写入要embed的feature_map \n
        patchsize: 一个patch的大小, 决定滑窗裁剪操作 \n
        stride: 步进值, 决定滑窗裁剪操作 \n
        target_patches: 对齐patch个数,没有给定的话会使用fetures_layer的分辨率的最大乘积 \n
        target_dim: 要对齐的维度, 一般是最大输入特获图通道数 \n 
        output_size : 最后要输出的每个patch的维度,通过一维池化来给定
        
        简化版:去掉了Align_patches步骤,假设所有层特征具有相同的空间分辨率
    """

    def patchify(feature:torch.Tensor, patchsize:int, stride:int):
        """Returns feature embeddings for images."""
        # input: shape[B, C, H, W]
        # output: [B, n_patches, C, patchsize, patchsize]
        padding = int((patchsize - 1) / 2)
        # 滑窗裁剪为patch
        unfolder = torch.nn.Unfold(
            kernel_size=patchsize, stride=stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(feature)
        # 计算可以分割多少个patch
        number_of_total_patches = []
        for s in feature.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (patchsize-1) -1) / stride + 1
            number_of_total_patches.append(int(n_patches))
        # 进行reshape操作: [B, C*patchsize*patchsize, n_patches] -> [B, C, patchsize, patchsize, n_patches]
        unfolded_features = unfolded_features.reshape(*feature.shape[:2], patchsize, patchsize, -1)
        # 进行permute操作: [B, C, patchsize, patchsize, n_patches] -> [B, n_patches, C, patchsize, patchsize]
        unfolded_features = unfolded_features.permute([0, 4, 1, 2, 3])
        return unfolded_features, number_of_total_patches
    

    def Align_dim(feature:torch.Tensor, target_dim:int):
        """
            feature: 经过patchify后的feature [B, n_patches, C, patchsize, patchsize]
            target_dim : 最后的目标维度
        """
        # [B, n_patches, C, ph, pw] -> [B*n_patches, C, ph, pw]
        _feature = feature.reshape(-1, *feature.shape[2:])
        # [B*n_patches, C, ph, pw] -> [B*n_patches, C*ph*pw]
        _feature = _feature.reshape(len(_feature), -1)
        # 增加维度用于1D池化: [B*n_patches, C*ph*pw] -> [B*n_patches, 1, C*ph*pw] # C*ph*pw = 3456
        _feature = _feature.unsqueeze(1)
        # 1D自适应平均池化到target_dim
        _feature = F.adaptive_avg_pool1d(_feature, target_dim) # 384 * 4 = 1536
        # 去除中间维度: [B*n_patches, 1, target_dim] -> [B*n_patches, target_dim]
        return _feature.squeeze(1)

    patch_shapes = []  # 用于存储不同layer的分辨率
    Align_dim_features = []  # 用于存储经过对齐dim后的特征
    
    for layer_idx in layers:
        feature = features[layer_idx] # feature[layer_idx]是所有样本(batch_size个)的对应layer_idx层的patch特征，形状为[B, C, H, W]
        # 切分patch
        Unfolded_feature, number_of_total_patches = patchify(feature, patchsize, stride)
        patch_shapes.append(number_of_total_patches)
        # 直接对齐dim，跳过Align_patches（假设所有层分辨率相同）
        Align_dim_feature = Align_dim(Unfolded_feature, target_dim=target_dim)
        Align_dim_features.append(Align_dim_feature) # 最终37 x 37 x 8 x 4 = 4352 这是所有样本不同layer的总的patch数量，布局为[B*n_patches, target_dim] * num_layers，其中n_patches是每层的patch数量（假设所有层相同），target_dim是对齐后的维度
    
    # 合并多个layer的特征
    # [[B*n_patches, target_dim] * num_layers] -> [B*n_patches, num_layers, target_dim]
    _features = torch.stack(Align_dim_features, dim=1) # 堆叠后每列都是对应的layer层特征
    # 展平层维度: [B*n_patches, num_layers, target_dim] -> [B*n_patches, 1, num_layers*target_dim] = [37x37x8,1,6144]，每个patch的特征被展平为一个长向量4*1536=6144维包含了所有层的信息
    _features = _features.reshape(len(_features), 1, -1)
    # 1D池化到目标输出维度
    _features = F.adaptive_avg_pool1d(_features, output_size=output_size) # 1536每个patch聚合了所有的layer的特征
    # 去除中间维度: [B*n_patches, 1, output_size] -> [B*n_patches, output_size]
    _features = _features.reshape(len(_features), -1)
    return _features, patch_shapes

