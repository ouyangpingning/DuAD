# ONNX 模型导出报告：DuAD 异常检测模型

## 一、ONNX 原理概述

### 1.1 什么是 ONNX

ONNX（Open Neural Network Exchange）是一种开放的神经网络模型交换格式，由微软和 Facebook 于 2017 年联合推出。它的核心思想是定义一套**统一的计算图中间表示（IR）**，使得不同深度学习框架训练的模型可以互相转换和部署。

### 1.2 为什么要转换为 ONNX

| 优势 | 说明 |
|------|------|
| **框架无关** | 导出后不再依赖 PyTorch 环境，用 ONNX Runtime 即可推理 |
| **推理加速** | ONNX Runtime 内置图优化（算子融合、常量折叠、内存规划），通常比原生 PyTorch 快 1.3~2 倍 |
| **轻量部署** | `onnxruntime` 包仅 ~10MB，而 PyTorch 超过 2GB |
| **跨平台** | 支持 C++/C#/Java/JavaScript 等多语言绑定，可部署到服务器、移动端、嵌入式设备 |
| **硬件兼容** | 支持 CPU、CUDA、TensorRT、OpenVINO、CoreML 等多种执行后端 |

### 1.3 ONNX 转换原理

PyTorch 模型转 ONNX 的核心是 `torch.onnx.export()`，其内部执行两个步骤：

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: Tracing（追踪）                                 │
│  用 example input 跑一次模型 forward，记录所有张量操作     │
│  → 生成一个静态计算图 (FX Graph 或 TorchScript Graph)    │
├─────────────────────────────────────────────────────────┤
│  Step 2: Translation（翻译）                             │
│  将 PyTorch 算子逐一映射为 ONNX 算子                      │
│  例如: aten::linear → onnx::Gemm                        │
│       aten::conv2d  → onnx::Conv                        │
│       F.interpolate → onnx::Resize                      │
└─────────────────────────────────────────────────────────┘
```

**关键约束**：模型 forward 中不能有依赖于输入值的动态控制流（如 `if tensor_value > 0:`），因为 tracing 只会记录实际执行的那条路径。

---

## 二、DuAD 模型 ONNX 导出架构

### 2.1 原 PyTorch 推理流程

```
                    ┌─────────── Python 端 ───────────┐
图像                  DINOv2          PCA mask
 │                      │                │
 ▼                      ▼                ▼
[3,518,518] ──→ features [N,1536] ──→ mask [N] bool
                                           │
                    ┌──────────────────────┘
                    ▼
              features[mask] (只保留前景 patch)
                    │
                    ▼
              Projection (2层 MLP)
                    │
                    ▼
              Discriminator (2层 MLP) → patch_scores [M,1]
                    │
                    ▼
              背景填充 (非前景 patch 用最低分填充)
                    │
                    ▼
              空间重塑 [B,H,W] → 上采样 → 高斯模糊
                    │
                    ▼
              heatmaps [B,518,518] + image_scores [B]
```

### 2.2 ONNX 导出策略

由于 PCA mask 生成依赖 `sklearn.decomposition.PCA` 和 `cv2`（非 PyTorch 算子），无法直接导出。因此采用**混合架构**：

```
┌──────── Python 预处理 ────────┐    ┌──────── ONNX 推理 ──────────────────────┐
                                  │    │                                          │
  图像 resize + Normalize         │    │  image [1,3,518,518]                     │
      │                           │    │      │                                   │
      ▼                           │    │      ▼                                   │
  DINOv2 特征提取 ──→ features ──┐│    │  DINOv2 特征提取 (get_intermediate_layers)│
      │                          ││    │      │                                   │
      ▼                          ││    │      ▼                                   │
  PCA mask 生成 ──→ mask ────────┘│    │  _embed_legacy 特征聚合                   │
                                  │    │      │                                   │
                                  │    │      ▼                                   │
                                  │    │  Projection → Discriminator → 负分数      │
                                  │    │      │                                   │
                                  │    │      ▼                                   │
                                  │    │  背景填充 → 空间重塑 → 上采样 → 高斯模糊    │
                                  │    │      │                                   │
                                  │    │      ▼                                   │
                                  │    │  heatmaps [1,518,518] + image_scores [1] │
                                  │    │                                          │
└─────────────────────────────────┘    └──────────────────────────────────────────┘
```

这样**所有可导出的部分**（DINOv2 + 特征聚合 + MLP + 后处理）都在 ONNX 中，只有 PCA mask 留在外部。

### 2.3 模型文件信息

| 项目 | 值 |
|------|-----|
| 输入 | `image [1, 3, 518, 518]`（ImageNet 归一化）+ `mask [1, 1369] bool`（PCA 前景） |
| 输出 | `heatmaps [1, 518, 518]` + `image_scores [1]` |
| 计算图节点数 | 581 |
| ONNX 文件 | ~860 KB（仅计算图） |
| 权重文件 | ~112 MB（DINOv2 + Projection + Discriminator 权重） |
| 总大小 | ~113 MB |
| Opset 版本 | 17 |

---

## 三、代码逐段解析

### 3.1 高斯核生成 (`make_gaussian_kernel`，第39-46行)

```python
def make_gaussian_kernel(sigma: float = 4.0):
    ksize = int(2 * (4 * sigma) + 1)  # =33，匹配 OpenCV 行为
    x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g1d = torch.exp(-0.5 * (x / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d.view(1, 1, ksize, ksize), ksize // 2
```

**说明**：原始推理中使用 `cv2.GaussianBlur(m, (0,0), sigmaX=4)` 做后处理平滑。由于 `cv2` 不是 PyTorch 算子，这里用三角函数公式手动构造一个 33×33 的 2D 高斯核（通过两个 1D 高斯向量的外积），然后用 `F.conv2d` 实现等价的高斯滤波。此操作完全可被 ONNX 追踪。

### 3.2 端到端导出模型 (`FullAnomalyDetectorONNX`，第51-182行)

这是核心类，包装了完整的推理流程。

#### (a) 初始化 (`__init__`，第64-87行)

```python
self.encoder = dino_encoder          # DINOv2 ViT-Small/14 + registers
self.projection = projection          # 2层 MLP: 1536→1536→1536
self.discriminator = discriminator    # 2层 MLP: 1536→1024→1
self.layer_indices = [2, 5, 8, 11]   # 要提取的 Transformer 层索引
self.H = self.W = target_size // 14   # 518//14 = 37 (特征图尺寸)
self.input_planes = 384 * 4 = 1536   # 聚合后的特征维度
```

将预先训练好的 PyTorch 模型组件赋值到导出模型中，同时预计算不可训练的参数（高斯核）。

#### (b) DINOv2 特征提取 (`_extract_intermediate_layers`，第89-103行)

```python
outputs = self.encoder.get_intermediate_layers(
    image, n=self.layer_indices,   # n=[2,5,8,11]
    reshape=True,                   # 输出 [B,384,37,37] 空间格式
    return_class_token=False,       # 不要 class token
    norm=True,                      # 应用 LayerNorm
)
return list(outputs)                # 4 个 [B,384,37,37] 张量
```

**原理**：DINOv2 的 `get_intermediate_layers` 内部使用显式 for 循环遍历 12 层 Transformer blocks，在索引 [2,5,8,11] 处收集输出。该循环在 tracing 时被展开（unroll）为静态计算图，因此可以被 ONNX 追踪。

**为什么用 `get_intermediate_layers` 而不是手动迭代 blocks**：PyTorch 2.x 的 Dynamo 导出器在嵌套 `nn.ModuleList` 迭代上有兼容性问题，直接调用 DINOv2 自身的方法避免了这个问题。

#### (c) 特征聚合 (`_embed_legacy`，第105-144行)

此方法将原始 `utils._embed_legacy` 内联重写，确保 ONNX 可追踪。流程：

```
对每一层特征 [B,384,37,37]:
  │
  ├─ F.unfold(kernel=3, padding=1, stride=1)  → [B, 384×9, 1369]
  │     (3×3 滑窗展开每个 patch 的邻域)
  │
  ├─ reshape + permute → [B, 1369, 384, 3, 3]
  │     (按空间位置重排)
  │
  ├─ reshape → [1369, 3456]
  │     (展平每个 patch 的局部特征)
  │
  └─ F.adaptive_avg_pool1d(target=1536) → [1369, 1536]
        (自适应池化到统一维度)

4 层特征堆叠:
  │
  ├─ torch.stack → [1369, 4, 1536]
  ├─ reshape → [1369, 1, 6144]
  └─ F.adaptive_avg_pool1d(output=1536) → [1369, 1536]
       (跨层聚合)
```

**数据形状变化**：

| 步骤 | 输入形状 | 输出形状 |
|------|---------|---------|
| 单层 Unfold | `[B, 384, 37, 37]` | `[B, 3456, 1369]` |
| Reshape | `[B, 3456, 1369]` | `[B, 1369, 384, 3, 3]` |
| 展平 | `[B, 1369, 384, 3, 3]` | `[1369, 3456]` |
| Pool1d 降维 | `[1369, 3456]` | `[1369, 1536]` |
| 4 层堆叠 | 4×`[1369, 1536]` | `[1369, 4, 1536]` |
| 跨层池化 | `[1369, 1, 6144]` | `[1369, 1536]` |

#### (d) 主 forward 流程（第146-182行）

```python
def forward(self, image, mask):
    # Step 1: DINOv2 特征提取
    layer_features = self._extract_intermediate_layers(image)
    # 返回 4 个 [B, 384, 37, 37] 张量

    # Step 2: 特征聚合
    features = self._embed_legacy(layer_features)
    # [B*37*37, 1536] = [1369, 1536]

    # Step 3: Projection → Discriminator → 负分数
    projected = self.projection(features)       # [1369, 1536]
    scores = -self.discriminator(projected)     # [1369, 1]
    scores = scores.squeeze(-1)                 # [1369]

    # Step 4: PCA 背景填充
    # 非前景 patch 用前景最低分填充（避免背景影响异常检测）
    large = torch.full_like(scores, 1e10)
    fg_scores = torch.where(mask, scores, large)  # 背景设为极大值
    min_fg = fg_scores.min()                      # 前景最低分
    scores = torch.where(mask, scores, min_fg.expand_as(scores))

    # Step 5: 后处理
    scores_2d = scores.reshape(B, 1, H, W)        # [1, 1, 37, 37]
    upsampled = F.interpolate(                     # [1, 1, 518, 518]
        scores_2d, size=(518,518), mode='bilinear'
    )
    blurred = F.conv2d(                            # 高斯平滑
        F.pad(upsampled, [16]*4, mode='reflect'),
        self.gaussian_kernel,                       # 33×33 高斯核
    )                                              # [1, 1, 518, 518]

    # Step 6: 输出
    heatmaps = blurred.squeeze(1)                  # [1, 518, 518]
    image_scores = heatmaps.reshape(1,-1).max(dim=1).values  # [1]
    return heatmaps, image_scores
```

**关键设计决策**：
- **`torch.where` 替代索引赋值**：`scores[mask] = val` 是 Python 索引操作，ONNX 不支持；改用 `torch.where(mask, scores, fill_val)`，它是标准 ONNX 算子
- **`F.conv2d` 替代 `cv2.GaussianBlur`**：用预计算的高斯核做卷积，完全等价且可追踪
- **背景填充的 `1e10` 技巧**：将背景 patch 分数设为极大值后取 `min()`，确保只在前景范围内找最小值

### 3.3 导出函数 (`export_onnx`，第187-234行)

```python
def export_onnx(ckpt_path, onnx_path, config, target_size=518, opset_version=17):
    # 1. 加载训练好的 PyTorch 模型
    detector = DINOv2AnomalyDetector(...)
    detector.load(ckpt_path)  # 恢复 Projection + Discriminator 权重

    # 2. 构建 ONNX 导出模型 (复用恢复的组件)
    model = FullAnomalyDetectorONNX(
        dino_encoder=detector.feature_extractor.encoder,  # 直接引用已加载的 DINOv2
        projection=detector.projection,
        discriminator=detector.discriminator,
        ...
    )

    # 3. 导出
    torch.onnx.export(
        model,
        (dummy_image, dummy_mask),    # example inputs (B=1)
        onnx_path,
        input_names=['image', 'mask'],
        output_names=['heatmaps', 'image_scores'],
        opset_version=opset_version,
    )
```

**关于动态 Batch**：当前版本导出时使用固定的 `B=1`。PyTorch 2.x 的 Dynamo 导出器在处理跨输入的 shared dynamic dimension 时存在兼容性问题（`torch.export.Dim` 同名约束在 `B=1` 的边界情况下失效）。后续 PyTorch 版本更新后可添加 `dynamic_shapes` 参数以支持变长 batch。

### 3.4 验证函数 (`verify_onnx`，第239-300行)

导出后自动验证 ONNX 模型与 PyTorch 模型的输出一致性：

```python
# 用相同随机输入分别跑 PyTorch 和 ONNX Runtime
with torch.no_grad():
    pt_heatmaps, pt_scores = pt_model(images, mask)

session = ort.InferenceSession(onnx_path)
onnx_heatmaps, onnx_scores = session.run(None, {
    'image': images.numpy(), 'mask': mask.numpy()
})

# 逐元素对比
hm_diff = |pt_heatmaps - onnx_heatmaps|.max()
```

验证通过阈值：`atol=1e-3`（实测差异均在 `~6e-04` 以下，主要来自浮点精度差异）。

### 3.5 CLI 入口 (`main`，第305-346行)

```bash
# 全样本模型
python src/export_onnx.py --category bottle

# 少样本模型 (k=2, seed=0)
python src/export_onnx.py --category bottle --k_shot 2 --shot_seed 0

# 导出并验证
python src/export_onnx.py --category bottle --verify

# 自定义图像尺寸
python src/export_onnx.py --category bottle --target_size 288
```

参数说明：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--category` | str | 必填 | MVTec AD 类别名称 |
| `--k_shot` | int | None | 少样本数量，None=全样本 |
| `--shot_seed` | int | 0 | 少样本随机种子 |
| `--target_size` | int | 518 | 输入图像尺寸（从 config.toml 读取） |
| `--verify` | flag | False | 导出后用 ONNX Runtime 验证 |
| `--opset` | int | 17 | ONNX 算子集版本 |

---

## 四、ONNX 模型使用示例

### 4.1 安装依赖

```bash
pip install onnxruntime  # CPU 版本 (~10MB)
# 或
pip install onnxruntime-gpu  # GPU 版本
```

### 4.2 Python 推理

```python
import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
import cv2

# 1. 加载 ONNX 模型
session = ort.InferenceSession("model_onnx/bottle_full.onnx")

# 2. 图像预处理 (与训练时一致)
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
image = transform(pil_image).unsqueeze(0).numpy()  # [1, 3, 518, 518]

# 3. 生成 PCA 前景 mask (需要在 Python 端完成)
#    使用训练好的 PCA 模型计算前景区域
mask = generate_pca_mask(image)  # [1, 1369] bool

# 4. ONNX 推理
heatmaps, image_scores = session.run(
    ['heatmaps', 'image_scores'],
    {'image': image.astype(np.float32), 'mask': mask}
)

# heatmaps: [1, 518, 518] 异常热图
# image_scores: [1] 图像级异常分数
```

### 4.3 C++ 推理（概念示意）

```cpp
#include <onnxruntime_cxx_api.h>

Ort::Session session(env, "model.onnx", Ort::SessionOptions{});

// 准备输入
std::vector<float> image_data(1 * 3 * 518 * 518);
std::vector<bool> mask_data(1 * 1369);
// ... 填充数据 ...

Ort::Value input_image = Ort::Value::CreateTensor<float>(...);
Ort::Value input_mask = Ort::Value::CreateTensor<bool>(...);

auto outputs = session.Run(Ort::RunOptions{},
    input_names, &input_image, 1, &input_mask, 1,
    output_names, 2);
```

---

## 五、当前局限与改进方向

| 局限 | 说明 | 解决方向 |
|------|------|---------|
| **固定 Batch=1** | 每次只能推理一张图 | PyTorch 更新后启用 `dynamic_shapes` |
| **PCA mask 仍在 Python** | 需要 sklearn + cv2 | 将 PCA 重写为纯 PyTorch 实现（`torch.linalg.svd`） |
| **target_size 固定** | 导出时的尺寸不可变 | 使用动态尺寸输入（需修改 DINOv2 位置编码逻辑） |
| **外部数据文件** | 权重存储在 .onnx.data | 设置 `external_data=False` 可合并为单文件 |

---

## 六、ONNX 算子映射表

导出模型中使用的主要 PyTorch → ONNX 算子映射：

| PyTorch 操作 | ONNX 算子 | 用途 |
|-------------|----------|------|
| `torch.nn.Linear` | `Gemm` | MLP 全连接层 |
| `torch.nn.LayerNorm` | `LayerNormalization` | Transformer 归一化 |
| `F.unfold` | `Im2Col` | 滑窗展开 |
| `F.adaptive_avg_pool1d` | `GlobalAveragePool` | 特征维度对齐 |
| `F.interpolate(bilinear)` | `Resize` | 上采样到原图尺寸 |
| `F.conv2d` | `Conv` | 高斯平滑 |
| `F.pad(reflect)` | `Pad` | 边界填充 |
| `torch.where` | `Where` | PCA 背景填充 |
| `torch.min` | `ReduceMin` | 前景最低分计算 |
| `torch.reshape` | `Reshape` | 张量形状变换 |
| `torch.stack` | `Concat` + `Unsqueeze` | 多层特征堆叠 |

---

*报告生成日期：2026-05-18*
*对应代码文件：`src/export_onnx.py`*
