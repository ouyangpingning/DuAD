# DuAD ONNX 部署契约（客户端对齐协议）

> 本文件是服务器端（训练/导出）与客户端（上位机推理/显示）之间的**接口契约**。
> 服务器端实现见 `12-训练期阈值标定与ONNX导出方案.md`；本文件只描述客户端必须遵守的
> 输入输出口径、阈值语义与回退逻辑。两端以本文为准，如有改动必须双方同步。

---

## 1. ONNX 模型：输入输出契约

### 1.1 输入

| 输入 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `image` | `[B, 3, 518, 518]` | float32 | **已归一化**的图：`(pixel/255 - mean) / std`，mean=`[0.485, 0.456, 0.406]`，std=`[0.229, 0.224, 0.225]`（ImageNet）。**不要传原始像素**。B 为动态，任意 batch 可用 |

- PCA 掩码（前景/背景）在**模型内部**计算（`PCA inline` 模式，新 checkpoint 导出）。
- 旧模型回退模式需要额外输入 `mask [B, 1369] bool`（见第 5 节兼容性），按 `duad.deploy` metadata 是否存在判断，一般客户端只面向新模型。

### 1.2 输出

| 输出 | 形状 | 说明 |
|---|---|---|
| `heatmaps` | `[B, 37, 37]` | **patch 级**异常分数图（37 = 518/14）。**未上采样、未高斯平滑** |
| `image_scores` | `[B]` | 图像级异常分数（patch 分数的最大值） |

### 1.3 客户端必须补做的后处理（服务器端有意不包含）

客户端拿到 `heatmaps` 后，按以下顺序得到与服务器端 `Predictor._upsample_masks` 一致的像素热力图 `hm_smooth [518, 518]`：

1. 双线性上采样到 518×518（`align_corners=False`，即 OpenCV `INTER_LINEAR` 语义）
2. 高斯平滑：`kernel_size=25, sigma=4`（等价于 `cv2.GaussianBlur((0,0), sigmaX=4)` 对 float32 图的自动计算）

⚠ 这两步不做，阈值语义（第 3 节）就不成立，检测结果会错。

---

## 2. 部署阈值：ONNX metadata 契约

训练结束时服务器端在**验证集**上标定两个阈值，随模型写入 ONNX `metadata_props`。
客户端用 `session.get_modelmeta().custom_metadata_map` 读取，key 如下：

| key | 类型 | 含义 |
|---|---|---|
| `duad.image_threshold` | str(float) | 图像级部署阈值（原始 patch-max 尺度，**与 `image_scores` 同尺度**） |
| `duad.image_threshold_method` | str | 标定方法：`"youden"` / `"best_f1"` / `"p99"` |
| `duad.pixel_threshold` | str(float) | 像素级 F1-max 分割阈值（原始 amap 尺度，**与 `hm_smooth` 同尺度**） |
| `duad.pixel_f1_max` | str(float) | 像素级最优 F1 值（参考，非阈值） |
| `duad.category` | str | 类别名（如 `bottle`） |
| `duad.calibrated_at` | str | 标定时间（ISO 字符串） |
| `duad.deploy` | str(JSON) | 完整 deploy dict（含 `image_thresholds` 阈值表，诊断/扩展用） |

**读取优先级：ONNX metadata > `*.threshold.json` > 默认值 1.7。**

---

## 3. 两个阈值的语义（最容易出错的地方）

### 3.1 图像级阈值 `duad.image_threshold`

- **用途**：判定整图是否有缺陷；同时也是**初始阈值设定**（免人工标定）。
- **判定方式**：`image_scores > duad.image_threshold` → 缺陷。
- **尺度**：原始 patch-max 分数（即 ONNX `image_scores` 的数值尺度），
  **不需要任何归一化**。

### 3.2 像素级阈值 `duad.pixel_threshold`

- **用途**：异常区域的精确定位显示（`hm_smooth > threshold` 的像素为异常区域）。
- **尺度**：原始 amap（上采样 + 高斯平滑后、**未归一化**的 `hm_smooth`）。
- **判定方式**：`hm_smooth > duad.pixel_threshold` → 异常像素。

### 3.3 ⚠ 百分位归一化只能用于颜色映射，不能用于二值化

服务器端可视化（`visualize_feature.py`）的百分位归一化
（clip 到 p2/p98，再 min-max 到 [0,1]）是**逐图自适应**的：
每张图的 [0,1] 空间都不同，因此归一化后的固定阈值**跨图不可比**，
`duad.pixel_threshold` 在归一化空间里没有意义。

客户端显示热力图时的正确顺序：

```
1. hm_smooth 原始值 → 与 duad.pixel_threshold 比较 → 得到异常区域二值图（定位）
2. 颜色显示：可对 hm_smooth 做百分位归一化后映射颜色（仅为好看）
   —— 但第一步的二值化判断必须发生在归一化之前
```

---

## 4. 客户端标准推理流程（伪代码）

```python
import onnxruntime as ort
import cv2, numpy as np

# 1. 加载模型 + 读 metadata 阈值
sess = ort.InferenceSession("model_onnx/bottle_full.onnx")
meta = sess.get_modelmeta().custom_metadata_map
image_thr = float(meta.get("duad.image_threshold", fallback_json_or_1.7))
pixel_thr = float(meta.get("duad.pixel_threshold", fallback_json_or_0.5))

# 2. 预处理：resize 到 518 + ImageNet 归一化
img = cv2.resize(raw, (518, 518)).astype(np.float32) / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img = img.transpose(2, 0, 1)[None]          # [1, 3, 518, 518]

# 3. 推理
heatmaps, image_scores = sess.run(None, {"image": img})
score = float(image_scores[0])              # 图像级分数

# 4. 后处理：上采样 + 高斯平滑（契约 1.3）
hm = cv2.resize(heatmaps[0], (518, 518), interpolation=cv2.INTER_LINEAR)
hm_smooth = cv2.GaussianBlur(hm, (0, 0), sigmaX=4)

# 5. 判定
is_defect = score > image_thr              # 图像级：初始阈值免人工
anomaly_region = hm_smooth > pixel_thr     # 像素级：异常区域定位

# 6. 颜色显示（可选）：百分位归一化仅作用于颜色，不影响第 5 步
```

---

## 5. 兼容性与兜底

| 情形 | 客户端行为 |
|---|---|
| ONNX 无 `duad.*` metadata（旧模型） | 回退读 `*.threshold.json`；再没有则用默认值 1.7（图像级） |
| `image_scores` 只有正常样本可用（现场无缺陷标定） | 阈值在服务器端已按 p99 回退（`image_threshold_method="p99"`） |
| `duad.pixel_f1_max == -1` | 像素级无法定义（标定集无缺陷），客户端定位功能退化为手动阈值 |
| skip 类别（如 `macaroni2`） | PCA 掩码在模型内部自动为全前景，客户端无需感知 |

---

## 6. 客户端对齐自检清单（改完代码后逐项确认）

1. [ ] 输入图做了 ImageNet 归一化（不是原始像素）
2. [ ] 后处理做了上采样（INTER_LINEAR）+ 高斯平滑（kernel 25 / sigma 4）
3. [ ] 图像级判定用 `image_scores` 与 `duad.image_threshold` 比较，未做归一化
4. [ ] 像素级定位用 `hm_smooth`（原始尺度）与 `duad.pixel_threshold` 比较
5. [ ] 百分位归一化只用于颜色显示，未参与二值化
6. [ ] metadata 缺失时走 `*.threshold.json` → 默认 1.7 兜底
7. [ ] 与服务器端 PyTorch 输出做过数值一致性抽查
   （同图同权重，heatmaps/score 差异应 < 1e-3）
