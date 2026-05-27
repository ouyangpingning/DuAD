# PCA Student — 用可学习模型替代推理时 SVD 分解

## 动机

在 DuAD 方法中，推理时 `PCAMaskGenerator._compute_first_pc_torch()` 对每张测试图像的 DINOv2 特征 `[1369, 1536]` 执行 `torch.linalg.svd`，复杂度 O(N·C·min(N,C)) ≈ 2.9B 次运算。实测在 RTX 4090 上，单张图像的 SVD 耗时约 **386 ms**，加上阈值+形态学后完整 PCA 掩模生成耗时约 **540 ms**，占总推理时间（~563 ms）的 **96%**，成为主要推理瓶颈。

## 核心思路

PCA 是线性运算：

```
first_pc = X_centered @ v1 = X @ w + b
```

其中 `v1` 是第一主成分方向向量，`w = v1`, `b = -μ @ v1`。因此一个 `Linear(1536, 1, bias=True)` 即可精确表达 PCA 投影。

在类别训练前，用原始 SVD 产出的 first PC 值作为监督信号（ground truth），训练一个小型神经网络（"PCA Student"）来预测 first PC 值。训练好后，推理时用 Student 的矩阵乘法替换 SVD，剩余流程（阈值、自适应方向校正、形态学处理）保持不变。

## 架构

### 模型

MLP(1536, H1) → ReLU → MLP(H1, H2) → ReLU → MLP(H2, 1)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dim` | 1536 | DINOv2 多尺度聚合特征维度 |
| `hidden_dims` | [512, 128] | 隐藏层维度，可在 config.toml 中配置 |

输出 raw logits，训练用 `BCEWithLogitsLoss`（内部 sigmoid + BCE），推理时经 sigmoid 转为前景概率 [0, 1]。

### 训练流程

入口方法为 `DINOv2AnomalyDetector.train_pca_student(train_dataloader)`（`src/myAD.py:1088`），分为前导检查和三个阶段。

**前导检查：** 若 `use_pca_student=False` 直接跳过；若开启了 Student 但未开启 `use_pca_mask`，打印警告并跳过（PCA Student 替换的是 PCA mask 流程中的 SVD 步骤，因此依赖 PCA mask）。

#### Phase 1 — 收集 SVD 掩模 target

```
创建临时 PCAMaskGenerator (pca_student=None, 回退 SVD)
遍历 train_dataloader:
  images → feature_extractor(images) → features [B*H*W, 1536]
  features → temp_pca_gen(features, (H, W)) → binary_mask [B*H*W]
  将 features 和 binary_mask.float() 转为 CPU 后分别 append 到列表

all_features = concat(所有 batch 的 features)   # [N_total, 1536]
all_targets = concat(所有 batch 的 masks)        # [N_total]
```

关键细节：
- target 是 SVD 生成的完整二值掩模（已含阈值、中心检测、形态学处理），非中间 PC 投影值
- `feature_extractor` 处于 `eval()` 模式，全程 `torch.no_grad()`
- 统计信息记录前景比例 `fg_ratio`
- 此阶段仅在全部训练数据上执行一次，输出固定的 `all_features` / `all_targets` 作为后续训练的数据集
```

关键细节：
- `feature_extractor` 处于 `eval()` 模式，全程 `torch.no_grad()`
- `compute_first_pc_svd()` 内部对 `[N, 1536]` 做 `torch.linalg.svd`，得到第一主成分方向 `Vh[0,:]` 并计算 `X_centered @ v1` 作为 target
- SVD 回退：若 GPU SVD 异常，自动回退到 CPU 上的 `sklearn.decomposition.PCA`
- 所有特征和 target 先转到 CPU 再保存，避免占用 GPU 显存
- 此阶段仅在全部训练数据上执行一次，输出固定的 `all_features` / `all_targets` 作为后续训练的数据集

#### Phase 2 — 训练 PCA Student（行 1155–1212）

```
TensorDataset(all_features, all_targets)
  → DataLoader(batch_size=pca_student_batch_size, shuffle=True, drop_last=False)

for epoch in 1..pca_student_epochs:
  student.train()
  for batch_feat, batch_target in loader:
    batch_feat, batch_target → device
    pred = student(batch_feat).squeeze(-1)       # [B]
    loss = MSELoss(pred, batch_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

  # 每 10 个 epoch 或首个 epoch 做一次验证：
  student.eval()
  with no_grad:
    取前 4096 个 patch 的特征和目标
    pred, target → 去均值后计算 cosine similarity (Corr)
    log MSE 和 Corr

best_loss = min(best_loss, avg_loss)
```

关键细节：
- 优化器：`Adam(lr=pca_student_lr, weight_decay=1e-5)`
- 损失函数：`BCEWithLogitsLoss`（logits + sigmoid 内部计算，数值更稳定）
- `pca_student_batch_size` 默认 4096（patch 级 batch，远大于图像级 batch_size，因为每张图产出 1369 个 patch）
- 验证指标：IoU（掩模重叠率）和 Acc（像素级准确率），直接衡量掩模质量
- 不保存"最佳模型"——训练完成后直接使用最后一步的 `student` 参数
- 训练结束后将 `student` 设为 `eval()` 模式

#### Phase 3 — 挂接到 PCA 生成器（行 1215）

```
_set_pca_student_on_generators():
  if trainer is not None and trainer.pca_generator is not None:
    trainer.pca_generator.set_pca_student(self.pca_student)
  if predictor is not None and predictor.pca_generator is not None:
    predictor.pca_generator.set_pca_student(self.pca_student)
```

关键细节：
- 如果 `trainer` 尚未创建（首次调用 `train_pca_student` 在 `fit()` 之前），此时只挂接 `pca_student` 到实例的 `self.pca_student` 属性
- 之后 `fit()` / `predict()` 创建 `Trainer` / `Predictor` 时会检查 `self.pca_student` 并自动挂接到新创建的 PCA generator（见 `src/myAD.py:1242` 和 `1263`）
- 挂接后，`PCAMaskGenerator._compute_first_pc_torch()` 的推理路径从 `compute_first_pc_svd(features)` 切换为 `self.pca_student(features).squeeze(-1)`，实现 ~3800× 加速

### 推理替换

在 `PCAMaskGenerator._compute_first_pc_torch()` 中：

```
if self.pca_student is not None:
    return torch.sigmoid(self.pca_student(features).squeeze(-1))  # [0,1] 前景概率, ~0.1 ms
else:
    return compute_first_pc_svd(features)                          # PC 投影值, ~386 ms
```

在 `PCAMaskGenerator.compute_background_mask()` 中，PCA Student 路径：
- 概率 > 0.5 即为前景，跳过阈值比较和中心区域自适应反转
- 仅保留形态学后处理（膨胀+闭运算）

## 代码集成位置

| 文件 | 改动 |
|------|------|
| `config.toml` | 新增 `[pca_student]` 配置段 |
| `src/config.py` | `build_model_config()` 读取新字段 |
| `src/myAD.py` | `ModelConfig` 新增 PCA Student 字段；`compute_first_pc_svd()` 独立函数；`PCAStudent` 类（MLP + raw logits 输出）；`PCAMaskGenerator` 增加 `pca_student` 参数及快捷路径；`DINOv2AnomalyDetector` 增加 `train_pca_student()`、`_set_pca_student_on_generators()`、`save_pca_student()`、`load_pca_student()` |
| `src/main.py` | `train_category()` 中在 GAN 训练前调用 `model.train_pca_student(train_loader)` |

### 配置示例

```toml
[pca_student]
use_pca_student = false          # 主开关
hidden_dims = [512, 128]         # MLP 隐藏层维度
lr = 0.001                       # Adam 学习率
epochs = 50                      # 训练轮数
batch_size = 4096                # mini-batch 大小 (patch 级别)
```

## 实验结果

### VisA capsules（100 张训练图像）

| 模型 | Corr（与 PCA 目标） | IoU（与 PCA 掩模） |
|------|-------------------|-------------------|
| Linear(1536, 1) | 0.428 | 0.651 |
| Linear(1536+pos, 1) | 0.426 | 0.670 |
| **MLP(1536+pos)** | **0.991** | **0.766** |
| K-Means K=2 center | — | 0.828 |

### MVTec AD screw（100 张训练图像）

| 模型 | Corr（与 PCA 目标） | IoU（与 PCA 掩模） |
|------|-------------------|-------------------|
| Linear(1536, 1) | **0.777** | **0.703** |
| Linear(1536+pos, 1) | 0.778 | 0.703 |
| MLP(1536+pos) | 0.991 | 0.688 |
| K-Means K=2 center | — | 0.499 |

### 推理速度对比（RTX 4090）

| 方法 | 耗时/张 | 加速比 |
|------|---------|--------|
| SVD (原始) | 386 ms | 1× |
| PCA Student (Linear) | ~0.1 ms | **~3800×** |
| 完整推理 (特征提取+PCA) | 563 ms | — |
| 完整推理 (特征提取+Student) | ~178 ms | **3.2×** |

## 关键发现

1. **PCA 是逐图计算的**——每张图像有自己的均值 μ 和方向 v1。一个全局 Linear 层试图用一个方向拟合所有图像，当类别内图像差异大时（如多物体的 VisA capsules）Corr 仅 0.43；当图像一致性高时（如中心化螺丝的 MVTec screw）Corr 可达 0.78。

2. **MLP + 位置编码可达 Corr=0.99**——MLP 的非线性能力结合空间位置信息，可以逼近逐图变化的 PCA 方向。但位置编码对纯 Linear 层无帮助（缺少非线性来利用位置信息）。

3. **所有模型都无法超越 PCA**——因为训练目标是拟合 PCA 输出，天花板就是 PCA 本身。PCA 粗粒度分割（包含背景 patch）的问题会被 Student 原样继承。

4. **若要去掉 PCA 推理，需要同时去掉训练时的 PCA**——实验表明训练带 PCA 但推理不带 PCA 会导致 Image AUROC 从 0.90 降至 0.81。

## 使用方式

```bash
# 1. 编辑 config.toml，设置 use_pca_student = true
# 2. 正常训练（PCA Student 会在每个类别训练前自动训练）
python src/main.py --categories "screw bottle"

# 3. 推理自动使用 PCA Student（无需额外参数）
python src/visualize_feature.py --categories "screw bottle"

# 4. 测试去掉 PCA 推理的效果
python src/visualize_feature.py --categories "screw" --no_pca
```
