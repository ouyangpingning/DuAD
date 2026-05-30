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

### 2.2 两种 ONNX 导出模式

`export_onnx.py` 支持两种 PCA 推理模式，通过 `--pca_mode` 选择。

#### 模式 A: SVD (默认, `--pca_mode svd`)

PCA mask 由 Python 端 SVD 计算后传入 ONNX。与原有流程一致：

```
┌──────── Python 预处理 ────────┐    ┌──────── ONNX 推理 ──────────────────────┐
                                  │    │                                          │
  图像 resize + Normalize         │    │  image [1,3,518,518]                     │
      │                           │    │      │                                   │
      ▼                           │    │      ▼                                   │
  DINOv2 特征提取 ──→ features ──┐│    │  DINOv2 特征提取 (get_intermediate_layers)│
      │                          ││    │      │                                   │
      ▼                          ││    │      ▼                                   │
  PCA mask (SVD) ──→ mask ──────┘│    │  _embed_legacy 特征聚合                   │
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

**局限**：DINOv2 在 Python 端和 ONNX 端各运行一次（共两次），且 SVD 计算慢（~109 ms/张）。

#### 模式 B: PCA Student (`--pca_mode student`)

将 PCA Student MLP 内嵌到 ONNX 模型中，实现**端到端单次推理**：

```
┌─── Python 预处理 ───┐    ┌─────── ONNX 推理 (单次前向) ───────────────────┐
                        │    │                                                 │
  图像 resize+Normalize │    │  image [1,3,518,518]                            │
      │                 │    │      │                                           │
      ▼                 │    │      ▼                                           │
  输入 ONNX ────────────┘    │  DINOv2 特征提取 → _embed_legacy 聚合            │
                              │      │                                           │
                              │      ├── PCA Student MLP → sigmoid → mask       │
                              │      │                                           │
                              │      └── Projection → Discriminator → 负分数     │
                              │                     │                            │
                              │                     ▼                            │
                              │         背景填充 (PCA Student mask)               │
                              │                     │                            │
                              │                     ▼                            │
                              │         上采样 → 高斯平滑                         │
                              │                     │                            │
                              │                     ▼                            │
                              │         heatmaps + image_scores                  │
└─────────────────────────────┘    └─────────────────────────────────────────────┘
```

**优势**：DINOv2 只运行一次，PCA mask 生成仅需 ~0.09 ms（MLP 前向），比 SVD 快 ~1200×。

**导出前自动训练**：PCA Student 不在训练时持久化，而是在 ONNX 导出时**独立训练**（加载训练数据 → DINOv2 提取特征 → SVD 生成 targets → 训练 MLP → 保存 `.pth`）。

### 2.3 模型文件信息

| 项目 | SVD 模式 | Student 模式 |
|------|---------|-------------|
| 输入 | `image [1,3,518,518]` + `mask [1,1369] bool` | `image [1,3,518,518]` |
| 输出 | `heatmaps [1,518,518]` + `image_scores [1]` | 同左 |
| 计算图节点数 | ~581 | ~600 |
| ONNX 文件大小 | ~113 MB | ~113 MB |
| 导出产物 | `{category}_full.onnx` | `{category}_pca_student.pth` + `{category}_full_student.onnx` |
| Opset 版本 | 17 | 17 |

---

## 三、代码逐段解析

### 3.0 代码结构

`export_onnx.py` 通过基类 **`_BaseONNXModel`** 消除重复代码：

```
_BaseONNXModel (共享方法)
├── _extract_intermediate_layers()   # DINOv2 中间层特征提取
├── _embed_legacy()                  # 多尺度特征聚合
├── _post_process()                  # 上采样 + 高斯平滑
│
├── FullAnomalyDetectorONNX          # SVD 模式 (image + mask → heatmaps)
└── FullAnomalyDetectorWithStudentONNX  # Student 模式 (image → heatmaps)
```

PCA Student 训练复用 `myAD.py` 的 `DINOv2AnomalyDetector.train_pca_student()`，不在 `export_onnx.py` 中重复实现。

### 3.1 高斯核生成 (`make_gaussian_kernel`)

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

### 3.2 基类 `_BaseONNXModel`

两个 ONNX 模型类的共享基类，提供 DINOv2 特征提取、特征聚合和后处理：

- **`_extract_intermediate_layers(image)`**: 调用 `encoder.get_intermediate_layers()` 提取 [2,5,8,11] 层特征。该循环在 tracing 时被展开为静态计算图
- **`_embed_legacy(layer_features)`**: 将原始 `utils._embed_legacy` 内联重写，每层经 `F.unfold → reshape → adaptive_avg_pool1d`，最后堆叠跨层聚合。数据形状：4×[B,384,37,37] → [B*1369, 1536]
- **`_post_process(scores_flat, B)`**: 空间重塑 → 双线性上采样 → 高斯平滑 → (heatmaps, image_scores)

### 3.3 `FullAnomalyDetectorONNX` (SVD 模式)

继承 `_BaseONNXModel`，输入 `(image, mask)`，forward 流程：

```
image → _extract_intermediate_layers → _embed_legacy → features
  ↓
features → Projection → Discriminator → -scores
  ↓
mask 背景填充 (torch.where) → _post_process → (heatmaps, scores)
```

### 3.4 `FullAnomalyDetectorWithStudentONNX` (Student 模式)

继承 `_BaseONNXModel`，输入 `image`，forward 流程：

```
image → _extract_intermediate_layers → _embed_legacy → features
  ↓
features → PCA Student MLP → sigmoid → >0.5 → mask
features → Projection → Discriminator → -scores
  ↓
PCA Student mask 背景填充 → _post_process → (heatmaps, scores)
```

**关键区别**：PCA Student 在 ONNX 图内部生成 mask，部署时无需 Python 端 SVD。

### 3.5 导出函数

**`export_onnx()`** — SVD 模式导出:

```python
def export_onnx(ckpt_path, onnx_path, config, target_size=518, opset_version=17):
    detector, model = _build_detector_and_onnx_model(ckpt_path, config, target_size)
    model.to(device).eval()
    torch.onnx.export(model, (dummy_image, dummy_mask), onnx_path,
                      input_names=['image', 'mask'],
                      output_names=['heatmaps', 'image_scores'],
                      opset_version=opset_version)
```

**`export_full_student_onnx()`** — Student 模式导出（需先训练 PCA Student）：

```python
def export_full_student_onnx(detector, pca_student, onnx_path, config, ...):
    model = FullAnomalyDetectorWithStudentONNX(
        dino_encoder=detector.feature_extractor.encoder,
        projection=detector.projection,
        discriminator=detector.discriminator,
        pca_student=pca_student,  # 已训练的 PCA Student
        ...
    )
    torch.onnx.export(model, dummy_image, onnx_path,
                      input_names=['image'],
                      output_names=['heatmaps', 'image_scores'],
                      opset_version=opset_version)
```

### 3.6 验证函数

**`verify_onnx()`** / **`verify_full_student_onnx()`**：导出后用 ONNX Runtime 与 PyTorch 输出逐元素对比，验证通过阈值 `atol=1e-3`（实测差异 < 6e-4，主要来自浮点精度）。

### 3.7 CLI 入口 (`main`)

```bash
# SVD 模式 (默认)
python src/export_onnx.py --category bottle

# PCA Student 模式 (自动训练 + 导出)
python src/export_onnx.py --category bottle --pca_mode student --verify

# 少样本 + Student
python src/export_onnx.py --category bottle --k_shot 4 --shot_seed 0 --pca_mode student --verify
```

**Student 模式的内部流程**：

```
main() [--pca_mode student]
  ├─ 1. 加载 config.toml → build_model_config()
  ├─ 2. 加载 GAN checkpoint (proj + dsc)
  ├─ 3. 准备训练 dataloader (复用 dataset.py)
  ├─ 4. 临时设置 config.use_pca_mask/use_pca_student = True
  ├─ 5. detector.train_pca_student(train_loader)  ← 复用 myAD.py 现有训练逻辑
  ├─ 6. 保存 detector.pca_student → .pth
  └─ 7. export_full_student_onnx() → .onnx
```

参数说明：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--category` | str | 必填 | MVTec AD 类别名称 |
| `--k_shot` | int | None | 少样本数量，None=全样本 |
| `--shot_seed` | int | 0 | 少样本随机种子 |
| `--target_size` | int | 518 | 输入图像尺寸（从 config.toml 读取） |
| `--pca_mode` | str | `svd` | `svd` (外部 SVD mask) 或 `student` (内嵌 PCA Student) |
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
| **SVD 模式仍需 Python** | 需要 sklearn + cv2 做 PCA mask | 使用 `--pca_mode student` 导出端到端 ONNX |
| **Student 模式无形态学后处理** | ONNX 内 mask 直接用 sigmoid>0.5，无 dilate+close | 在 Python/C++ 调用端做后处理，或用 ONNX 实现形态学算子 |
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

*报告生成日期：2026-05-30*
*对应代码文件：`src/export_onnx.py` · `src/myAD.py`*
