# DuAD — Dual-Branch Unsupervised Anomaly Detection with DINOv2

[English](README.md) | **简体中文**

基于冻结 DINOv2 (`dinov2_vits14_reg`) 特征 + 双分支 GAN 的无监督工业视觉异常检测，支持**全样本**与**少样本**场景。

## 快速开始

### 克隆项目

```bash
# 含子模块（DINOv2）
git clone --recurse-submodules git@github.com:ouyangpingning/DuAD.git

# 如果已克隆但缺少 facebookresearch_dinov2_main/
git submodule init && git submodule update
```

### 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：`torch>=2.0`, `scikit-learn`, `opencv-python`, `matplotlib`, `tomli`

### 准备数据

下载 [MVTec AD](https://www.mvtefactory.com/annotated-dataset) 或 [VisA](https://amazon-visual-anomaly.s3.amazonaws.com/VisA.tar.gz) 数据集，修改 `config.toml` 中 `[paths]`：

```toml
[paths]
mvtec_base_dir = "/path/to/mvtec_anomaly_detection"
visa_base_dir = "/path/to/VisA"
```

### 训练

```bash
# 全样本训练（指定类别）
python src/main.py --categories "bottle screw"

# 少样本训练（每类 4 张正常图像）
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0

# VisA 数据集
python src/main.py --categories "candle" --dataset visa

# 交互式批量训练（按 GPU 显存自动分配 tmux 会话）
bash train_all_tmux.sh
```

### 可视化

```bash
# 命令行（默认随机抽取 4 张测试图）
python src/viz/visualize_feature.py --categories "bottle screw"

# 抽取更多样本 / 仅分析模式（无需 .pth）
python src/viz/visualize_feature.py --categories "bottle" --num_samples 8
python src/viz/visualize_feature.py --categories "bottle" --skip_inference

# 交互式批量（多 seed 并行）
bash visualize_all_tmux.sh
```

### ONNX 导出

```bash
# SVD 模式（默认）
python src/deploy/export_onnx.py --category bottle

# PCA Student 模式（端到端）
python src/deploy/export_onnx.py --category bottle --pca_mode student --verify

# 交互式导出
bash export_onnx_all_tmux.sh
```

### 结果汇总

```bash
# 交互式
bash aggregate_results.sh

# 直接调用
python src/analysis/aggregate_results.py
python src/analysis/aggregate_results.py --csv
```

## 项目结构

```
.
├── config.toml                      # 统一参数配置（唯一参数源）
├── requirements.txt
├── train_all_tmux.sh                # 交互式批量训练
├── visualize_all_tmux.sh            # 交互式批量可视化
├── export_onnx_all_tmux.sh          # 交互式 ONNX 导出
├── aggregate_results.sh             # 交互式结果汇总
├── .gitmodules                      # Git 子模块配置
├── src/
│   ├── main.py                      # 训练入口
│   ├── viz/
│   │   ├── visualize_feature.py     # 可视化入口
│   ├── deploy/
│   │   ├── export_onnx.py           # ONNX 模型导出
│   ├── analysis/
│   │   ├── aggregate_results.py     # 日志汇总统计
│   ├── myAD.py                      # 核心模型
│   ├── dataset/                     # 数据集统一抽象层（Facade）
│   │   ├── __init__.py              #   统一 API：get_dataloader()
│   │   ├── mvtec.py                 #   MVTec AD 数据集
│   │   └── visa.py                  #   VisA 数据集
│   ├── utils.py                     # 评估指标、日志、特征聚合、DINOv2 加载
│   ├── perlin.py                    # Perlin 噪声掩码生成
│   ├── config.py                    # TOML → ModelConfig 解析器
│   └── commen_import.py             # 共享第三方导入
├── facebookresearch_dinov2_main/    # Git 子模块：DINOv2 源码
├── model_ckpt/                      # 模型检查点（gitignore）
├── model_log/                       # 训练日志（gitignore）
├── model_onnx/                      # ONNX 模型（gitignore）
├── outputs/                         # 可视化输出（gitignore）
└── results/                         # 汇总 CSV（gitignore）
```

## 核心特性

### 1. 双分支对抗训练 (Dual-Branch GAN)

| 分支 | 噪声定位 | 损失函数 | 作用 |
|------|---------|---------|------|
| **Perlin 分支** | PCA 掩模 ∩ Perlin 噪声 | BCE Loss | 精确定位噪声位置 |
| **PCA 分支** | 整个 PCA 前景区域 | Hinge Loss | 全局判别能力 |

两分支共享判别器，权重通过 `perlin_branch_weight` / `pca_branch_weight` 调节。PCA 掩模关闭时自动回退单分支 Hinge Loss。

### 2. PCA 前景掩模

- 对 DINOv2 特征做 PCA（GPU 加速 SVD），取第一主成分投影值
- **PCA Student（推理加速）**：训练小型 MLP 直接预测 SVD 二值掩模，推理时 `sigmoid > 0.5` 替代 SVD 流程，加速约 **343×**
- 自适应阈值 + 中心区域保护，支持类别特定阈值
- 纹理类物体（carpet、grid 等）自动跳过
- PCA Student 由 `Trainer.train_pca_student()` 每次按需训练，不持久化

### 3. Perlin 噪声掩模

- 在 PCA 前景基础上用 Perlin 噪声进一步约束噪声区域
- 多尺度混合 + 形态学腐蚀，噪声更集中于前景中心

### 4. 噪声退火

训练早期大噪声探索异常空间，后期减小噪声精细收敛。支持 `linear`、`cosine`、`exponential` 三种退火类型。少样本模式自动禁用。

### 5. 少样本学习 (Few-Shot)

- `--k_shot N` 仅用每类 N 张正常样本，`--shot_seed S` 控制采样
- 带放回 RandomSampler 保证 batch 填满
- 自动启用几何增强 + 颜色增强（多颜色类别）
- 多 seed 训练取平均降低采样方差
- 检查点/日志独立命名：`{cat}_k{K}_s{seed}_best_ckpt.pth`

### 6. 数据集 Facade 模式

`src/dataset/__init__.py` 提供统一入口 `get_dataloader(root_dir, category, dataset_type, ...)`，根据 `dataset_type` 自动分发到 MVTec 或 VisA 加载器。新增数据集只需在 `dataset/` 下添加模块并注册。

### 7. 可视化工具

`CategoryVisualizer` 类（Strategy + Template Method 模式）编排 5 种可视化：

| 输出 | 内容 |
|------|------|
| `{category}_heatmap.png` | N×3 网格：原图 + GT Mask + plasma 热力叠加（随机抽样、百分位归一化、F1 阈值过滤） |
| `augmented/{category}_augmented.png` | 数据增强预览（少样本） |
| `pca_mask/{category}_pca_mask.png` | PCA 掩模 SVD vs Student 对比 |
| `perlin_mask/{category}_perlin_mask.png` | Perlin 掩模叠加 |
| `feature_map/{category}_feature_map.png` | DINOv2 特征 L2 范数激活图 |

### 8. ONNX 部署

支持导出端到端 ONNX 模型，无需 PyTorch 即可推理：

| PCA 模式 | 产物 | 输入 |
|---------|------|------|
| SVD | `{category}_full.onnx` | image + mask |
| PCA Student | `{category}_full_student.onnx` | image only |

## 训练流程

```
[Pre-training] PCA Student (可选)
  训练图像 → DINOv2 特征 → SVD 掩模 (GT) → BCE 训练 MLP

[GAN Training]
  输入图像 [B, 3, 518, 518]
    ↓ 冻结 DINOv2 ViT-S/14 (layers [2,5,8,11])
  多层特征 × 4
    ↓ _embed_legacy 聚合
  特征 patches [B*H*W, 1536]
    ↓ PCA 前景掩模（PCA Student 或 SVD）
  前景特征 [N, 1536]
    ↓ Projection MLP
  投影特征 [N, 1536]
    ↓ + 高斯噪声
  真假特征 → Discriminator MLP → 异常分数
```

推理：

```
测试图像 → DINOv2 → 聚合 → PCA mask → Projection → Discriminator → 负分数
                                                                          ↓
                                                          上采样 + 高斯平滑 → 热力图
```

## 参数配置

所有参数集中在 `config.toml`：

```toml
[architecture]     # 输入尺寸、层索引、特征维度
[training]         # meta_epochs, gan_epochs, batch_size, 学习率
[noise]            # 噪声标准差、退火参数
[pca_mask]         # PCA 掩模阈值、跳过类别
[pca_student]      # PCA Student MLP 配置
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

训练时快速评估仅计算 AUROC；完成后完整评估计算全部指标（含交叉归一化集成）。

## 添加快捷命令

```bash
# 训练
python src/main.py --categories "bottle screw"
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0

# 可视化
python src/viz/visualize_feature.py --categories "bottle screw"
python src/viz/visualize_feature.py --categories "bottle" --skip_inference

# 导出
python src/deploy/export_onnx.py --category bottle
python src/deploy/export_onnx.py --category bottle --pca_mode student --verify

# 汇总
python src/analysis/aggregate_results.py
python src/analysis/aggregate_results.py --csv
```

## 许可

Apache License 2.0
