# DuAD — 基于 DINOv2 的合成异常判别方法

[English](README.md) | **简体中文**

基于冻结 DINOv2 (`dinov2_vits14_reg`) 特征 + 双分支合成异常判别器的无监督工业视觉异常检测，支持**全样本**与**少样本**场景。

## 方法概览

![DuAD Method](Method.png)

## 快速开始

### 克隆项目

```bash
git clone --recurse-submodules https://github.com/ouyangpingning/DuAD.git

# 如果已克隆但缺少 facebookresearch_dinov2_main/
git submodule init && git submodule update
```

### 环境安装

```bash
# 1. 先安装 PyTorch（必须用 CUDA 索引，否则可能装成 CPU 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. 再安装其余依赖
pip install -r requirements.txt
```

> **CUDA 兼容性**: 驱动版本不同时，在 https://pytorch.org/get-started/locally/ 选择对应的 `--index-url`。当前仓库使用 CUDA 12.4（兼容驱动 ≥ 525）。

核心依赖：`torch>=2.0`, `scikit-learn`, `opencv-python-headless`, `matplotlib`, `tomli`

> **服务器用户**: 使用 `opencv-python-headless`（无 GUI 依赖）。如需 `cv2.imshow`，替换为 `opencv-python` 并安装 `apt install libgl1-mesa-glx`。

### 准备数据

下载 [MVTec AD](https://www.mvtefactory.com/annotated-dataset) 或 [VisA](https://amazon-visual-anomaly.s3.amazonaws.com/VisA.tar.gz) 数据集，在 `config.toml` 中配置路径：

```toml
[paths]
mvtec_base_dir = "/path/to/mvtec_anomaly_detection"
visa_base_dir = "/path/to/VisA"
```

### 训练

```bash
# 全样本
python src/main.py --categories "bottle screw"

# 少样本 (K=4)
python src/main.py --categories "bottle screw" --k_shot 4 --shot_seed 0

# VisA 数据集
python src/main.py --categories "candle" --dataset visa

# 消融实验 (交互式，支持 MVTec AD 和 VisA)
bash train_ablation.sh

# 交互式批量训练 (tmux, 按 GPU 显存自动分配)
bash train_all_tmux.sh
```

### 可视化

```bash
# 随机抽取测试图 + 热力图叠加
python src/viz/visualize_feature.py --categories "bottle screw"

# 更多样本 / 仅分析模式 (无需 checkpoint)
python src/viz/visualize_feature.py --categories "bottle" --num_samples 8
python src/viz/visualize_feature.py --categories "bottle" --skip_inference

# 交互式批量可视化 (tmux)
bash visualize_all_tmux.sh
```

### ONNX 导出

```bash
# SVD 模式（默认）
python src/deploy/export_onnx.py --category bottle

# PCA Student 模式（端到端，推理时无需 SVD）
python src/deploy/export_onnx.py --category bottle --pca_mode student --verify

# 交互式批量导出
bash export_onnx_all_tmux.sh
```

### 结果汇总

```bash
bash aggregate_results.sh
python src/analysis/aggregate_results.py
python src/analysis/aggregate_results.py --csv
```

## 项目结构

```
.
├── config.toml                      # 统一参数配置（唯一参数源）
├── requirements.txt
├── train_all_tmux.sh                # 交互式批量训练
├── train_ablation.sh                # 交互式消融实验
├── visualize_all_tmux.sh            # 交互式批量可视化
├── export_onnx_all_tmux.sh          # 交互式 ONNX 导出
├── aggregate_results.sh             # 交互式结果汇总
├── src/
│   ├── main.py                      # 训练入口
│   ├── DuAD.py                      # 核心模型 (Trainer, Predictor 等)
│   ├── benchmark_pca_mask.py        # PCA 掩模速度基准测试 (SVD vs MLP)
│   ├── config.py                    # TOML → ModelConfig 解析器
│   ├── utils.py                     # 评估指标、日志、DINOv2 加载
│   ├── commen_import.py             # 共享第三方导入
│   ├── dataset/                     # 数据集抽象层 (Facade 模式)
│   │   ├── __init__.py              #   统一 API: get_dataloader()
│   │   ├── mvtec.py                 #   MVTec AD 数据集
│   │   └── visa.py                  #   VisA 数据集
│   ├── viz/
│   │   └── visualize_feature.py     # 可视化入口
│   ├── deploy/
│   │   └── export_onnx.py           # ONNX 模型导出
│   └── analysis/
│       └── aggregate_results.py     # 日志汇总统计
├── simplenet/                       # SimpleNet 基线 (独立项目)
├── facebookresearch_dinov2_main/    # Git 子模块: DINOv2 源码
├── model_ckpt/                      # 模型检查点
├── model_log/                       # 训练日志
├── model_onnx/                      # ONNX 模型
├── outputs/                         # 可视化输出
└── results/                         # 汇总 CSV
```

## 许可

Apache License 2.0
