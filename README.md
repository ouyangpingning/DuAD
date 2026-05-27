# DuAD — Dual-Branch Unsupervised Anomaly Detection with DINOv2

基于冻结 DINOv2 特征 + 双分支 GAN 的无监督工业视觉异常检测，支持**全样本** 与**少样本**场景。

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心特性](#核心特性)
- [训练流程](#训练流程)
- [可视化](#可视化)
- [结果汇总](#结果汇总)
- [参数配置](#参数配置)
- [评估指标](#评估指标)
- [进阶使用](#进阶使用)

## 项目结构

```
.
├── config.toml                  # 统一参数配置（唯一参数源）
├── requirements.txt
├── train_all_tmux.sh            # 交互式批量训练脚本（tmux 并行）
├── visualize_all_tmux.sh        # 交互式批量可视化脚本（tmux 并行）
├── aggregate_results.sh         # 交互式结果汇总脚本
├── model_ckpt/                  # 模型检查点（按类别分目录）
│   ├── bottle/
│   ├── screw/
│   └── ...
├── model_onnx/                  # ONNX 导出模型（端到端推理）
│   ├── bottle_k2_s0_full.onnx
│   └── ...
├── model_log/                   # 训练日志（按类别分目录）
│   ├── bottle/
│   ├── screw/
│   └── ...
├── outputs/                     # 可视化 / 实验输出
├── results/                     # 汇总 CSV 输出
├── docs/
│   ├── ONNX_export_report.md    # ONNX 导出详细报告
│   └── pca_student.md           # PCA Student 设计与训练流程
├── scripts/
│   └── aggregate_results.py     # 日志汇总统计脚本
├── facebookresearch_dinov2_main/  # DINOv2 本地源码（torch.hub 加载）
└── src/
    ├── main.py                  # 训练入口（支持断点续训）
    ├── visualize_feature.py     # 可视化 / 推理入口
    ├── export_onnx.py           # ONNX 模型导出（端到端）
    ├── myAD.py                  # 核心模型（ModelConfig / Trainer / Predictor / 组件）
    ├── dataset.py               # MVTec AD 数据加载器
    ├── utils.py                 # 评估指标、日志、特征聚合
    ├── perlin.py                # Perlin 噪声掩码生成
    ├── config.py                # TOML → ModelConfig 解析器
    └── commen_import.py         # 共享第三方导入
```

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：`torch>=2.0`, `scikit-learn`, `opencv-python`, `matplotlib`, `tomli`

ONNX 推理（导出后）：`onnxruntime`（~10MB，无需 torch）

### 准备数据

下载 [MVTec AD](https://www.mvtefactory.com/annotated-dataset) 数据集，修改 `config.toml` 中 `[paths]` 的 `base_dir`：

```toml
[paths]
base_dir = "/path/to/mvtec_anomaly_detection"
```

### 训练

```bash
# 全样本训练（默认 15 个类别）
python src/main.py

# 指定类别
python src/main.py --categories "bottle screw capsule"

# 少样本训练（每类 4 张正常图像）
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 42
```

### 交互式批量训练

```bash
bash train_all_tmux.sh
```

交互式选择：全样本 / 少样本 → 类别 → K 值和种子，按 GPU 显存自动分配 tmux 会话并行训练。

### 可视化

```bash
# 命令行调用
python src/visualize_feature.py --categories "bottle screw"

# 少样本模型
python src/visualize_feature.py --categories "bottle" --k_shot 4 --shot_seed 42

# 交互式批量（多 seed 并行）
bash visualize_all_tmux.sh
```

### 结果汇总

```bash
# 交互式汇总
bash aggregate_results.sh

# 或直接调用 Python 脚本
python scripts/aggregate_results.py                     # 默认日志目录
python scripts/aggregate_results.py /path/to/logs       # 指定日志目录
python scripts/aggregate_results.py --csv               # CSV 格式输出
```

## 核心特性

### 1. 双分支对抗训练 (Dual-Branch GAN)

| 分支 | 噪声定位 | 损失函数 | 作用 |
|------|---------|---------|------|
| **Perlin 分支** | PCA 掩模 ∩ Perlin 噪声 | BCE Loss | 精确定位噪声位置 |
| **PCA 分支** | 整个 PCA 前景区域 | Hinge Loss | 全局判别能力 |

两个分支共享同一个判别器，权重通过 `perlin_branch_weight` / `pca_branch_weight` 调节。PCA 掩模关闭时自动回退为单分支 Hinge Loss。

### 2. PCA 前景掩模

- 对 DINOv2 特征做 PCA（GPU 加速 SVD），取第一主成分投影值
- **PCA Student（推理加速）**：训练一个小型 MLP 直接预测 SVD 二值掩模，推理时 sigmoid(logits) > 0.5 替代完整 SVD 流程，加速约 **343×**
- 自适应阈值分离前景 / 背景，中心区域保护避免反转（SVD 路径）
- 支持类别特定阈值（见 `config.toml` `[category_pca.threshold]`）
- 纹理/网格类物体自动跳过
- PCA Student 由 `Trainer.train_pca_student()` 按需训练（~1 分钟），不持久化文件，每次训练/可视化时自动训练

### 3. Perlin 噪声掩模

- 在 PCA 前景基础上用 Perlin 噪声约束噪声区域
- 多尺度混合（`perlin_min` / `perlin_max`）
- 形态学腐蚀让噪声更集中于前景中心

### 4. 余弦退火噪声强度

训练早期大噪声探索异常空间，后期减小噪声精细收敛。支持 `linear`、`cosine`、`exponential` 三种类型。

### 5. 少样本学习 (Few-Shot)

- `--k_shot N` 仅用每类 N 张正常样本
- 带放回 RandomSampler 保证 batch 填满
- 少样本自动启数据增强（翻转、旋转），多颜色类别启用颜色增强
- 多 `--shot_seed` 训练取平均降低采样方差
- 检查点和日志均独立命名：`model_ckpt/{cat}/{cat}_k{K}_s{seed}_best_ckpt.pth`，`model_log/{cat}/{cat}_k{K}_s{seed}_full.log`
- 全样本对应 `{cat}_best_ckpt.pth` / `{cat}_full.log`，互不覆盖

### 6. ONNX 模型导出

支持将训练好的模型导出为 ONNX 格式进行部署推理：

```bash
# 导出全样本模型
python src/export_onnx.py --category bottle

# 导出少样本模型并验证
python src/export_onnx.py --category bottle --k_shot 2 --shot_seed 0 --verify
```

导出模型包含完整的端到端推理流程（DINOv2 → 特征聚合 → Projection → Discriminator → 后处理），仅需 `onnxruntime` 即可推理，不依赖 PyTorch 环境。

详细说明见 [`docs/ONNX_export_report.md`](docs/ONNX_export_report.md)。

### 7. 断点续训

训练过程中如果服务器中断，重新运行相同命令即可从断点继续：

```bash
# 训练中断后，直接重新运行同一命令
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0
# 自动检测 latest_ckpt，恢复模型/优化器状态，从中断 epoch 继续
```

每个 epoch 结束后自动保存 `latest_ckpt`，训练正常完成后自动删除。断点续训无需额外参数。

## 训练流程

```
[Pre-training] PCA Student (可选，启用 use_pca_student 时)
  训练图像 → DINOv2 特征 → SVD 掩模 (ground truth) → BCE 训练 MLP

[GAN Training]
  输入图像 [B, 3, 518, 518]
    ↓ 冻结 DINOv2 ViT-S/14 (layers [2,5,8,11])
  多层特征 [B, 384, H, W] × 4
    ↓ _embed_legacy 聚合
  特征 patches [B*H*W, 1536]
    ↓ PCA 前景掩模（可选，PCA Student 或 SVD）
  前景特征 [N, 1536]
    ↓ Projection MLP
  投影特征 [N, 1536]
    ↓ + 高斯噪声
  真假特征 → Discriminator MLP → 异常分数
```

### 推理

```
测试图像 → DINOv2 → 聚合 → PCA Student (sigmoid > 0.5) 或 SVD → Projection → Discriminator → 负分数
                                                                              ↓
                                                                    上采样 + 高斯平滑 → 像素级热力图
```

## 可视化

运行 `visualize_feature.py` 或 `visualize_all_tmux.sh`，输出至 `outputs/`：

| 路径 | 内容 |
|------|------|
| `{category}_test.png` | 测试样本异常热力图 |
| `augmented/{category}_augmented.png` | 数据增强效果（少样本） |
| `pca_mask/{category}_pca_mask.png` | PCA 前景掩模 |
| `perlin_mask/{category}_perlin_mask.png` | Perlin 掩模叠加 |
| `feature_map/{category}_feature_map.png` | DINOv2 特征激活热力图 |

## 结果汇总

`scripts/aggregate_results.py` 递归扫描 `model_log/` 子目录下所有日志，自动解析：

- 按 (类别, k_shot) 分组统计均值 ± 标准差
- 跨类别平均汇总
- 支持 `--csv` 输出便于导入 Excel / LaTeX

## 参数配置

所有参数集中在 `config.toml`：

```toml
[architecture]     # 输入尺寸、层索引、特征维度
[training]         # meta_epochs, gan_epochs, batch_size, 学习率
[noise]            # 高斯噪声标准差、余弦退火参数
[pca_mask]         # PCA 掩模阈值、跳过类别
[pca_student]      # PCA Student: MLP 加速 PCA 掩模推理（~343×）
[perlin_mask]      # Perlin 掩模、双分支权重
[augment]          # 几何/颜色增强类别
[category_pca]     # 类别特定 PCA 阈值和边界
[paths]            # 数据、模型、日志、输出路径
```

## 评估指标

| 指标 | 含义 |
|------|------|
| **Image AUROC** | 图像级异常检测 |
| **Pixel AUROC** | 像素级异常分割 |
| **Pixel PRO** | Per-Region Overlap — 异常区域定位精度 |
| **AP** | Average Precision |
| **F1** | 基于最优 PR 阈值的 F1 Score |

## 进阶使用

### 多种子少样本取平均

```bash
bash train_all_tmux.sh  # 选择少样本模式，种子输入 "0 42 123 666 2024"

# 训练完成后汇总
python scripts/aggregate_results.py
```

### 添加新类别

1. 在 MVTec AD 数据集中加入新类别文件夹
2. 如需 PCA 掩模，在 `config.toml` 的 `[category_pca.threshold]` 添加阈值
3. 纹理类物体加入 `skip_categories`

### 调参建议

| 现象 | 调整方向 |
|------|---------|
| 图像 AUROC 低 | 提高 `meta_epochs`、调大 `dsc_margin` |
| 像素 AUROC 低 | 调优 `pca_threshold`、`pca_border`（按类别） |
| 少样本过拟合 | 增大 `noise_std_max` 或 `noise_std_min` |
| 训练太慢 | 减少 `meta_epochs`、增大 `batch_size` |

## 许可与引用

本项目为在读研究生研究课题的一部分，使用 MVTec AD 数据集需遵循其[许可条款](https://www.mvtefactory.com/licenses)。
