#!/bin/bash
# 交互式 ONNX 模型导出脚本 (单类别)
# 用法: bash export_onnx_all_tmux.sh

work_env="${CONDA_DEFAULT_ENV:-base}"
if [ "$work_env" = "base" ]; then
    echo "⚠ 当前 conda 环境为 base，建议激活 pytorch 环境后再运行"
    read -p "是否继续使用 base 环境？[y/N]: " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消。请先执行: conda activate <你的环境名>"
        exit 1
    fi
fi

work_path="$(cd "$(dirname "$0")" && pwd)"

# ==================== 交互式配置 ====================
echo "================================================"
echo "  ONNX 模型导出 - 交互式配置"
echo "================================================"
echo ""

# --- 1. PCA 模式 ---
echo "请选择 PCA 推理模式:"
echo "  [1] SVD 模式 (默认)"
echo "      mask 在 Python 端用 SVD 计算, 导出标准 ONNX (image+mask → heatmaps)"
echo "  [2] PCA Student 模式"
echo "      自动训练 PCA Student MLP → 导出端到端 ONNX (image → heatmaps)"
read -p "输入 [1/2] (默认=1): " pca_choice
pca_choice=${pca_choice:-1}

if [ "$pca_choice" == "2" ]; then
    pca_mode="student"
    mode_label="PCA Student (端到端)"
else
    pca_mode="svd"
    mode_label="SVD (mask 外部传入)"
fi

echo ""

# --- 2. 选择 checkpoint ---
echo "选择要使用的 checkpoint (对应训练时的配置):"
echo "  [1] 全样本    → model_ckpt/{category}/{category}_best_ckpt.pth"
echo "  [2] 少样本    → model_ckpt/{category}/{category}_k{K}_s{seed}_best_ckpt.pth"

read -p "输入 [1/2] (默认=1): " ckpt_choice
ckpt_choice=${ckpt_choice:-1}

if [ "$ckpt_choice" == "2" ]; then
    read -p "  K 值 (默认=4): " k_shot
    k_shot=${k_shot:-4}
    read -p "  随机种子 (默认=0): " shot_seed
    shot_seed=${shot_seed:-0}
    ckpt_label="model_ckpt/{cat}/{cat}_k${k_shot}_s${shot_seed}_best_ckpt.pth"
    k_shot_arg="--k_shot ${k_shot} --shot_seed ${shot_seed}"
else
    k_shot_arg=""
    ckpt_label="model_ckpt/{cat}/{cat}_best_ckpt.pth"
fi

echo ""

# --- 3. 类别 ---
echo "可用的 MVTec AD 类别:"
echo "  bottle cable capsule carpet grid hazelnut leather metal_nut"
echo "  pill screw tile toothbrush transistor wood zipper"
echo ""
read -p "输入类别名: " category

if [ -z "$category" ]; then
    echo "[ERROR] 必须指定一个类别"
    exit 1
fi

echo ""

# --- 4. 验证 ---
echo "导出后验证 (--verify):"
echo "  用 ONNX Runtime 和 PyTorch 分别推理同一随机输入, 逐元素对比输出,"
echo "  确保 ONNX 模型与 PyTorch 模型一致 (需要 pip install onnxruntime)。"
read -p "是否验证？[y/N] (默认=N): " verify_choice
if [ "$verify_choice" == "y" ] || [ "$verify_choice" == "Y" ]; then
    verify_flag="--verify"
    verify_label="是"
else
    verify_flag=""
    verify_label="否"
fi

# ==================== 摘要 + 确认 ====================
echo ""
echo "================================================"
echo "  配置摘要"
echo "================================================"
echo "  PCA 模式:     ${mode_label}"
echo "  Checkpoint:   ${ckpt_label}"
echo "  类别:         ${category}"
echo "  导出后验证:   ${verify_label}"
echo "  输出目录:     ${work_path}/model_onnx/"
echo "================================================"
echo ""

read -p "按回车开始导出，或 Ctrl+C 取消... "
echo ""

# ==================== 执行 ====================
cmd="python src/export_onnx.py --category ${category} ${k_shot_arg} --pca_mode ${pca_mode} ${verify_flag}"

echo "执行: ${cmd}"
echo ""

cd "$work_path"
conda activate "$work_env" 2>/dev/null

$cmd

echo ""
echo "================================================"
if [ $? -eq 0 ]; then
    echo "  导出完成！"
    echo ""
    echo "  产物:"
    if [ "$pca_mode" == "student" ]; then
        if [ -n "$k_shot_arg" ]; then
            base="${category}_k${k_shot}_s${shot_seed}"
        else
            base="${category}"
        fi
        echo "    model_onnx/${base}_pca_student.pth"
        echo "    model_onnx/${base}_full_student.onnx"
    else
        if [ -n "$k_shot_arg" ]; then
            base="${category}_k${k_shot}_s${shot_seed}"
        else
            base="${category}"
        fi
        echo "    model_onnx/${base}_full.onnx"
    fi
else
    echo "  导出失败！请检查错误信息。"
fi
echo "================================================"
