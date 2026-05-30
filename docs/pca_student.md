# PCA Student — 用可学习 MLP 替代推理时 SVD 分解

## 动机

在 DuAD 方法中，推理时 `PCAMaskGenerator._compute_first_pc_torch()` 对每张测试图像的 DINOv2 特征 `[1369, 1536]` 执行 `torch.linalg.svd`，复杂度 O(N·C·min(N,C)) ≈ 2.9B 次运算。实测在 RTX 4090 上，单张图像的 SVD 耗时约 **108 ms**，完整 PCA 掩模生成（含阈值+形态学）耗时约 **109 ms**。

PCA Student 将这个流程替换为 MLP 前向传播，**完整掩模生成仅需 0.32 ms（约 343× 加速）**。

## 核心思路

不用回归 PCA 投影值，而是直接学习 SVD 掩模的最终输出——即**二值前景/背景分割**。用 SVD 掩模作为 ground truth，BCE loss 训练 MLP 输出前景概率，推理时 `sigmoid(logits) > 0.5` 直接得到掩模。

```
SVD 路径:  DINOv2特征 → SVD → PC投影值 → 阈值 → 中心检测 → 形态学 → 掩模  (~109 ms)
MLP 路径:  DINOv2特征 → MLP → logits → sigmoid → >0.5 → 形态学 → 掩模      (~0.32 ms)
```

优势：
- **端到端学习**：Student 直接学习最终掩模，跳过阈值调参和中心区域自适应反转
- **方向确定**：SVD 的符号歧义（前景可能是正或负 PC 值）由 target 自动消解
- **速度快**：343× 加速完整掩模生成

## 架构

### 模型

MLP(1536, H1) → ReLU → MLP(H1, H2) → ReLU → MLP(H2, 1)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dim` | 1536 | DINOv2 多尺度聚合特征维度 |
| `hidden_dims` | [512, 128] | 隐藏层维度，可在 `config.toml` 中配置 |

852K 参数，输出 raw logits。训练用 `BCEWithLogitsLoss`（内部 sigmoid + BCE），推理时经 sigmoid 转为前景概率 [0, 1]。

### 配置（config.toml）

```toml
[pca_student]
use_pca_student = true          # 主开关
hidden_dims = [512, 128]        # MLP 隐藏层维度
lr = 0.001                      # Adam 学习率
epochs = 100                    # 训练轮数
batch_size = 1369               # mini-batch 大小（patch 级别）
```

## 训练流程

入口为 `Trainer.train_pca_student(train_dataloader)`（`src/myAD.py`），由 `DINOv2AnomalyDetector.train_pca_student()` 委托调用。

### Phase 1 — 收集 SVD 掩模 target

```
创建临时 PCAMaskGenerator (pca_student=None, 回退 SVD, 含完整阈值+中心检测+形态学)
遍历 train_dataloader:
  images → feature_extractor(images) → features [B*H*W, 1536]
  features → temp_pca_gen(features, (H, W)) → binary_mask [B*H*W]  (True=前景)
  将 features 和 binary_mask.float() 转为 CPU 后收集

all_features = concat(所有 batch)   # [N_total, 1536]
all_targets = concat(所有 batch)    # [N_total]  (0/1)
```

关键细节：
- target 是 SVD 生成的完整二值掩模（已含阈值比较、中心区域检测、自适应反转、形态学处理），非中间 PC 投影值
- `feature_extractor` 处于 `eval()` 模式，全程 `torch.no_grad()`
- 所有数据先转到 CPU 再存储，避免占用 GPU 显存
- 此阶段仅执行一次，输出固定的 `all_features` / `all_targets` 作为后续训练数据集

### Phase 2 — 训练 MLP

```
TensorDataset(all_features, all_targets)
  → DataLoader(batch_size=pca_student_batch_size, shuffle=True, drop_last=False)

for epoch in 1..pca_student_epochs:
  student.train()
  for batch_feat, batch_target in loader:
    pred = student(batch_feat).squeeze(-1)             # raw logits [B]
    loss = BCEWithLogitsLoss(pred, batch_target)        # sigmoid + BCE
    optimizer.zero_grad(); loss.backward(); optimizer.step()

  # 每 10 epoch 验证一次:
  student.eval()
  取前 4096 个 patch → 计算 IoU 和 Accuracy
  log: BCE={loss}, IoU={iou}, Acc={acc}
```

关键细节：
- 优化器：`Adam(lr=pca_student_lr, weight_decay=1e-5)`
- 损失函数：`BCEWithLogitsLoss`（比 sigmoid + BCELoss 数值更稳定）
- 验证指标：IoU（掩模重叠率）和 Acc（像素级准确率），直接衡量掩模质量
- 不保存中间最佳模型——训练完成后使用最终参数
- 训练结束后将 student 设为 `eval()` 模式

### Phase 3 — 挂接到 PCA 生成器

训练完成后，`Trainer` 直接将 PCA Student 挂接到自己的 `pca_generator`：

```python
self.pca_generator.set_pca_student(self.pca_student)
```

之后 `DINOv2AnomalyDetector` 在创建 `Predictor` 时自动将 `pca_student` 传递给 Predictor 的 `pca_generator`。

挂接后，`PCAMaskGenerator` 的推理路径从 SVD 切换为 MLP：

```python
# _compute_first_pc_torch():
if self.pca_student is not None:
    return torch.sigmoid(self.pca_student(features).squeeze(-1))  # [0,1] 前景概率
else:
    return compute_first_pc_svd(features)                          # SVD 回退

# compute_background_mask(): PCA Student 路径
probs > 0.5 → 形态学处理 → 二值掩模
# (跳过 SVD 路径的阈值比较和中心区域自适应反转)
```

## 训练策略

PCA Student **默认不持久化**——训练和可视化时按需训练（~1 分钟），用完即弃。

- **训练 (`main.py`)**：在 GAN 训练前调用 `model.train_pca_student(train_loader)`，训练完成后 student 常驻 Trainer 的 `pca_generator`
- **可视化 (`visualize_feature.py`)**：在生成掩模前调用 `detector.train_pca_student(train_loader)`，用完即弃
- **ONNX 导出 (`export_onnx.py`)**：当 `--pca_mode student` 时，**仅此时持久化**——训练 PCA Student → 保存 `{category}_pca_student.pth` → 导出内嵌 ONNX

ONNX 导出流程（`--pca_mode student`）：

```bash
python src/export_onnx.py --category bottle --pca_mode student
```

```
Step 1/2: detector.train_pca_student(train_loader)  ← 复用 myAD.py 训练逻辑
          → 保存 {category}_pca_student.pth
Step 2/2: export_full_student_onnx() → {category}_full_student.onnx
```

导出的 ONNX 模型内嵌 PCA Student，仅需 `image` 输入，单次 DINOv2 前向完成端到端推理。

## 代码结构

| 类/方法 | 文件 | 职责 |
|---------|------|------|
| `PCAStudent` | `src/myAD.py` | 模型定义（MLP + raw logits 输出） |
| `PCAMaskGenerator` | `src/myAD.py` | 掩模生成：MLP 路径（sigmoid→0.5→形态学）或 SVD 回退 |
| `Trainer.train_pca_student()` | `src/myAD.py` | Phase 1/2/3 完整训练逻辑 |
| `DINOv2AnomalyDetector.train_pca_student()` | `src/myAD.py` | 薄封装：检查开关→创建 Trainer→委托训练 |
| `main.py` `train_category()` | `src/main.py` | 在 GAN 训练前调用一次 |
| `visualize_feature.py` | `src/visualize_feature.py` | 在可视化前调用一次 |
| `FullAnomalyDetectorWithStudentONNX` | `src/export_onnx.py` | ONNX 可导出的端到端模型（内嵌 PCA Student） |
| `export_full_student_onnx()` | `src/export_onnx.py` | 将 PyTorch 模型 + PCA Student 导出为 ONNX |
| `main() [--pca_mode student]` | `src/export_onnx.py` | 训练 PCA Student → 保存 .pth → 导出 ONNX |

## 实验结果

### MVTec AD screw — 推理时掩模质量（K=4, 500 epochs 训练）

| 指标 | 训练集 (4 images) | 测试集 (160 images) |
|------|-------------------|---------------------|
| IoU (MLP vs SVD) | 0.73–0.80 | **0.75 ± 0.03** |
| IoU < 0.3 | 0 | **0 张** |
| IoU > 0.7 | 全部 | **153/160 (95.6%)** |
| Acc (像素级) | — | **0.92 ± 0.01** |

MLP 掩模与 SVD 掩模高度一致，正常和异常图像的 IoU 无显著差异（0.750 vs 0.749）。

### 推理速度对比（RTX 4090）

| 方法 | 耗时/张 | 加速比 |
|------|---------|--------|
| SVD PC 计算 | 108 ms | 1× |
| MLP PC 计算 | 0.09 ms | **~1171×** |
| SVD 完整掩模 | 109 ms | 1× |
| MLP 完整掩模 | 0.32 ms | **~343×** |

## 关键发现

1. **Binary 模式优于 Regression**：直接学习 SVD 掩模（BCE）比回归 PC 投影值（MSE）更稳定，IoU 从 0.50±0.35 提升到 0.75±0.03。

2. **MLP 泛化良好**：K=4 时用 4 张训练图像训练的 MLP，在 160 张测试图像上 IoU 全部 > 0.67，95.6% > 0.7。

3. **不受异常影响**：正常和异常测试图像的 IoU 几乎相同，说明 MLP 掩模质量不受图像内容（缺陷）干扰。

4. **不需要位置编码**：当前架构纯 MLP(1536→512→128→1)，不拼接空间坐标，IoU 已达 0.75。位置编码可进一步探索但非必需。

5. **训练目标决定上限**：PCA Student 学习的是 SVD 掩模，天花板就是 SVD 本身。SVD 掩模的粗粒度分割问题会被 Student 原样继承。

## 使用方式

```bash
# ── 训练 (config.toml 中设置 use_pca_student = true) ──
# PCA Student 在 GAN 训练前自动训练，但不保存 .pth
python src/main.py --categories "screw bottle"

# ── 可视化 (按需训练 PCA Student, 不保存 .pth) ──
python src/visualize_feature.py --categories "screw" --k_shot 4 --shot_seed 0
python src/visualize_feature.py --categories "screw" --k_shot 4 --shot_seed 0 --skip_inference

# ── ONNX 导出 (仅此时保存 .pth + 导出 .onnx) ──
# 全样本
python src/export_onnx.py --category bottle --pca_mode student --verify
# 少样本
python src/export_onnx.py --category bottle --k_shot 4 --shot_seed 0 --pca_mode student --verify
```
