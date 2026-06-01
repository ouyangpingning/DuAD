# PCA 掩模参数调优指南

## 概述

`PCAMaskGenerator.compute_background_mask()` 负责从 DINOv2 特征中分离前景/背景区域，生成二值掩模。本文档解释其参数含义及调优方法。

**代码位置**: `src/myAD.py:262-322`

## 处理流程

```
DINOv2 特征 [N, C]
      │
      ├─── PCA Student 路径 (use_pca_student=true) ──────────┐
      │     sigmoid(MLP(features)) → 概率 [0, 1]              │
      │     > 0.5 → 二值掩模 → 形态学后处理 → 输出              │
      │                                                       │
      └─── SVD 路径 (默认) ──────────────────────────────────┐ │
            first_pc = SVD 第1主成分投影值 (无界标量)           │ │
            > threshold → 初始二值掩模                         │ │
            检查中心区域前景占比                                │ │
            < 35%? ──YES──→ 反转掩模 (auto-reverse)           │ │
            形态学后处理 → 输出                                │ │
```

## 参数说明

### 1. `threshold` — SVD 路径的前景判定阈值

| 属性 | 说明 |
|------|------|
| 默认值 | `10.0` (config.toml `[pca_mask] pca_threshold`) |
| 作用 | `first_pc > threshold` 判定为前景 |
| 数据类型 | float，无上限 |
| 调大 | 掩模更严格，更少像素被判定为前景 |
| 调小 | 掩模更宽松，更多像素被判定为前景 |

**PC 值含义**: SVD 第一主成分投影值反映每个 patch 在主要变化方向上的偏离程度。物体类（如 bottle）通常前景 patches 变化大，PC 值集中在较高范围；纹理类（如 carpet）变化均匀，PC 值分布窄。

**典型值范围**:
- 物体类 (bottle, screw, metal_nut): `10 ~ 80`
- 纹理类: 通常直接 `skip_categories` 跳过，不调 threshold

### 2. `border_ratio` — 中心区域边界比例（auto-reverse 用）

| 属性 | 说明 |
|------|------|
| 默认值 | `0.2` (config.toml `[pca_mask] pca_border`) |
| 作用 | 决定"中心区域"的大小，用于 auto-reverse 判断 |
| 范围 | `0.0 ~ 0.5` |
| 计算方式 | 中心区域 = `[H*border, H*(1-border)] × [W*border, W*(1-border)]` |

举例 (H=W=37):
```
border=0.0 → 中心区域 = 37×37 (整个图)
border=0.2 → 中心区域 = 22×22 (去掉四周各 7 个 patch)
border=0.4 → 中心区域 = 8×8  (去掉四周各 14 个 patch)
```

| 调大 | 中心区域变小，auto-reverse 更容易触发（因为检查的区域小，前景少时反转更敏感） |
| 调小 | 中心区域变大，auto-reverse 更难触发 |

### 3. `kernel_size` — 形态学处理核大小

| 属性 | 说明 |
|------|------|
| 默认值 | `3` (config.toml `[pca_mask] pca_kernel_size`) |
| 作用 | 膨胀 + 闭运算的核大小 |
| 调大 | 掩模更平滑，小孔洞被填平，但细节丢失 |
| 调小 | 保留更多细节，但可能有噪声碎片 |
| 建议 | 一般不需要改，3 适用于大多数情况 |

### 4. `skip_categories` — 跳过 PCA 掩模的类别

| 属性 | 说明 |
|------|------|
| 默认值 | `["cable", "carpet", "grid", "leather", "tile", "transistor", "wood"]` |
| 作用 | 列表中的类别直接返回全 1 掩模（所有 patches 都是前景），跳过 SVD 计算 |
| 使用场景 | 纹理类（carpet, grid, tile 等）特征分布均匀，PCA 无法有效区分前景/背景 |

### 5. `use_pca_student` — 是否使用 PCA Student 替代 SVD

| 属性 | 说明 |
|------|------|
| 默认值 | `false` |
| 作用 | 用训练好的 MLP 替代 SVD 推理，速度约 343× 更快 |
| 注意 | 启用后 SVD 路径的 threshold/border 参数不再生效，改为 sigmoid(MLP) > 0.5 |

## Auto-reverse 机制

SVD 路径唯一依赖的自动纠错机制：

```python
# 第 314-317 行
center_mask = mask_2d[h_start:h_end, w_start:w_end]
if center_mask.sum() <= center_mask.numel() * 0.35:  # 中心前景 < 35%
    mask = (-first_pc) > self.threshold               # 反转！
```

**逻辑**: 大多数工业检测场景中，物体位于图像中心。如果中心区域前景占比小于 35%，说明掩模把背景当成了前景 → 自动反转。

**失效场景**: 
- 物体不在中心（如某些 VisA 类别）
- 物体占比太小或太大
- 中心 35% 阈值对特定类别不合适（硬编码，无法通过 config 修改）

**遇到 auto-reverse 失效时**，有以下选择：
1. 调整 `border_ratio` 改变中心区域大小
2. 调整 `threshold` 改变初始掩模
3. 将该类别加入 `skip_categories`，放弃 PCA 掩模

## 调优工作流

### Step 1: 生成可视化，诊断问题

```bash
python src/visualize_feature.py --dataset visa --categories "candle" --skip_inference
```

查看 `outputs/pca_mask/candle_pca_mask.png`：
- **掩模全是前景**（一片红）→ threshold 太小，调大
- **掩模全是背景**（几乎无红）→ threshold 太大，调小；或 auto-reverse 失效
- **前景/背景颠倒**（背景区域标红，物体区域透明）→ auto-reverse 失效

### Step 2: 调整 threshold

在 `outputs/pca_mask/` 图中会显示 "SVD 第一主成分投影值" 热力图，观察：
- 物体区域的颜色（暖色=高 PC 值，冷色=低 PC 值）
- 物体区域大致数值范围 → 设 threshold 为该范围的中间偏下值

```toml
[category_pca.threshold]
candle = 50.0     # 如果物体区域 PC 值在 40~80，设 50 合适
```

### Step 3: 调整 border_ratio（如 auto-reverse 失效）

```toml
[category_pca.border]
candle = 0.1      # 调小 border → 中心区域变大 → 更难反转
# candle = 0.3    # 调大 border → 中心区域变小 → 更容易反转
```

### Step 4: 如果仍然无法解决

将该类别加入 `skip_categories`，放弃 PCA 掩模：

```toml
[pca_mask]
skip_categories = ["cable", "carpet", ..., "candle"]
```

对于纹理密集型类别（PCB、fryum 等），通常直接 skip 是更好的选择。

## 常见问题速查

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| 掩模全红（全是前景） | threshold 太小 | 调大 threshold |
| 掩模几乎无红 | threshold 太大 | 调小 threshold |
| 物体变背景、背景变物体 | auto-reverse 判断错误 | 调 border_ratio，或 skip |
| 掩模碎片化 | kernel_size 太小 | 调大 kernel_size |
| 掩模太模糊 | kernel_size 太大 | 调小 kernel_size |
| 纹理类效果差 | PCA 不适合纹理 | 加入 skip_categories |
