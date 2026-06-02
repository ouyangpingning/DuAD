# visualize_feature.py — 可视化工具完整说明

## 概述

`src/visualize_feature.py` 是本项目的**推理可视化入口**，用于生成训练后模型的各类分析和诊断图表。

核心设计采用 **Strategy + Template Method** 模式，通过 `CategoryVisualizer` 类将 5 种可视化策略封装为独立方法，`run_all()` 模板方法按固定顺序依次调用。

## CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--categories` | `str` | 全部 15 类 | 要可视化的类别，空格分隔 |
| `--k_shot` | `int` | `None` | 少样本 K 值，`None` 表示全样本 |
| `--shot_seed` | `int` | `0` | 少样本采样种子 |
| `--dataset` | `choice` | `mvtec` | 数据集：`mvtec` 或 `visa` |
| `--num_samples` | `int` | `4` | 随机抽取的测试样本数（优先异常样本） |
| `--skip_inference` | `flag` | `False` | 跳过模型推理，仅生成分析可视化 |
| `--help` | — | — | 显示帮助信息 |

### 使用示例

```bash
# 基本用法：全样本模型，随机抽 4 张测试图
python src/visualize_feature.py --categories "bottle screw"

# 少样本模型
python src/visualize_feature.py --categories "bottle" --k_shot 4 --shot_seed 0

# 自定义抽取样本数
python src/visualize_feature.py --categories "bottle screw" --num_samples 8

# 仅分析模式（无需 .pth 文件，不跑推理）
python src/visualize_feature.py --categories "bottle" --skip_inference

# VisA 数据集
python src/visualize_feature.py --categories "candle" --dataset visa

# 交互式批量（多 seed 并行）
bash visualize_all_tmux.sh
```

## 架构

```
CategoryVisualizer                    ← 单个类别的可视化编排器
│
├── run_all()                         ← 模板方法：固定顺序调度
│
├── visualize_augmented()             ← Strategy 1: 数据增强预览
├── visualize_anomaly_heatmap()       ← Strategy 2: 异常热力图（核心）
├── visualize_pca_mask()              ← Strategy 3: PCA 掩模对比
├── visualize_perlin_mask(data)       ← Strategy 4: Perlin 掩模（依赖 3）
└── visualize_feature_map()           ← Strategy 5: 特征激活图

共享工具:
├── _denormalize()                    ← ImageNet 反归一化
├── _setup_chinese_font()             ← matplotlib 中文字体
├── _get_train_sample()               ← 取训练样本（缓存）
├── _create_pca_generator()           ← PCA 生成器工厂
└── _apply_f1_threshold()             ← F1 阈值计算与过滤
```

## 5 种可视化策略详解

### 1. 数据增强预览 (`augmented/`)

**触发条件**：少样本模式 + 类别启用增强

2×4 网格展示训练数据增强效果（翻转、旋转、颜色抖动）。

### 2. 异常热力图 (`{category}_heatmap.png`)

**触发条件**：`--skip_inference` 未设置（需要 .pth 文件）

这是核心可视化，处理流程：

```
1. 收集全部测试样本 → 分离异常/正常
2. 随机抽取 num_samples 张（优先异常，不足时正常补齐）
3. 逐张推理:
   a. DINOv2 特征提取
   b. PCA 前景掩模（可选）
   c. Projection → Discriminator → 负分数 = 异常分数
   d. 背景填充 + 上采样 + 高斯平滑
   e. 背景区域设为 NaN（可视化透明）
   f. 百分位归一化（2%~98%，仅前景）
   g. F1 阈值过滤（仅异常样本，低于最优阈值的分数→NaN）
4. N×3 网格渲染: [原图 | GT Mask | plasma 热力叠加]
```

**关键设计决策**：

| 特性 | 说明 |
|------|------|
| **背景 NaN** | PCA 背景区域透明，不干扰 colormap |
| **百分位归一化** | 裁剪 2% 极端离群值，放大中间区域对比度 |
| **F1 阈值** | 基于 GT mask 计算最优 F1 阈值，仅保留高置信度异常区域 |
| **plasma colormap** | 与 GT mask 的 gray 形成视觉对比 |
| **100% 叠加** | 热力图全不透明覆盖前景，背景原图全可见 |

### 3. PCA 掩模对比 (`pca_mask/`)

**触发条件**：`use_pca_mask = True`

- **有 PCA Student**：4 列布局（原图 / SVD PC 值 / SVD 掩模 / MLP 掩模），标注 IoU
- **无 PCA Student**：3 列布局（回退 `Visualizer.visualize_pca_mask`）

返回值 `pca_data` dict 供 `visualize_perlin_mask` 复用（`svd_mask`, `svd_mask_up`, `H`, `W`, `sample_image`, `img_np`）。

### 4. Perlin 掩模 (`perlin_mask/`)

**触发条件**：`use_perlin_mask = True` 且 `use_pca_mask = True`

依赖 `visualize_pca_mask` 的输出数据。4 列布局：原图 / PCA 掩模 / Perlin 掩模 / 叠加。

### 5. 特征激活图 (`feature_map/`)

**触发条件**：始终执行

DINOv2 特征 L2 范数激活热力图（jet colormap），展示模型"关注"的区域。

## 输出目录结构

```
outputs/
├── {category}_heatmap.png            # 异常热力图 N×3 网格（随机抽样）
├── augmented/
│   └── {category}_augmented.png      # 数据增强效果（仅少样本）
├── pca_mask/
│   └── {category}_pca_mask.png       # PCA SVD vs Student 对比
├── perlin_mask/
│   └── {category}_perlin_mask.png    # Perlin 噪声掩模
└── feature_map/
    └── {category}_feature_map.png    # DINOv2 特征激活图
```

少样本模式下文件名带 `_k{K}_s{seed}` 后缀（如 `bottle_k4_s0_heatmap.png`）。

## 与 main.py 的关系

| 维度 | `main.py` | `visualize_feature.py` |
|------|-----------|----------------------|
| 目的 | 训练模型 | 可视化分析 |
| 输入 | 训练集 | 训练集（分析用）+ 测试集（推理用） |
| 输出 | checkpoint (.pth) + 日志 | 图表 (.png) |
| PCA Student | 训练后挂接到 Trainer | 按需训练，挂接到临时 `PCAMaskGenerator` |
| 推理 | `Predictor.predict()` (全量，评估指标) | 内联推理 (单张，仅可视化) |

## 新增可视化类型

在 `CategoryVisualizer` 中添加：

```python
def visualize_xxx(self):
    """新可视化策略"""
    ...

def run_all(self):
    ...
    # 在合适位置插入
    self.visualize_xxx()
```

遵循现有方法的模式：获取数据 → 推理/计算 → 绘图 → 保存。
