#!/bin/bash
# 交互式 ONNX 模型导出脚本 (单类别)
# 用法: bash export_onnx_all_tmux.sh
#
# 导出模式由 checkpoint 自动决定:
#   [PCA inline] ckpt 含 pca_mean/pca_component (新格式) → 输入仅 image,
#                掩码在 ONNX 图内计算 (推荐)
#   [外部 mask]  旧格式 ckpt → 输入 image + mask, 掩码由部署端计算

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

# --- 1. 选择 checkpoint ---
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

# --- 2. 类别 ---
echo "可用类别:"
echo "  MVTec AD: bottle cable capsule carpet grid hazelnut leather metal_nut"
echo "            pill screw tile toothbrush transistor wood zipper"
echo "  VisA:     candle capsules cashew chewinggum fryum macaroni1 macaroni2"
echo "            pcb1 pcb2 pcb3 pcb4 pipe_fryum"
echo ""
read -p "输入类别名，空格分隔多个 (如: bottle screw): " categories_input

if [ -z "$categories_input" ]; then
    echo "[ERROR] 必须指定至少一个类别"
    exit 1
fi
categories=($categories_input)

echo ""

# --- 3. 验证 ---
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
echo "  Checkpoint:   ${ckpt_label}"
echo "  类别 (${#categories[@]}):  ${categories[@]}"
echo "  导出后验证:   ${verify_label}"
echo "  导出模式:     自动 (ckpt 含 PCA 参数 → 内联; 否则外部 mask)"
echo "  输出目录:     ${work_path}/model_onnx/"
echo "================================================"
echo ""

read -p "按回车开始导出，或 Ctrl+C 取消... "
echo ""

# ==================== 执行 ====================
# 注意: 不能拼成字符串后 $cmd 执行, 变量里的引号不会重新解析,
# 多类别会被 word splitting 拆散 (argparse 报 unrecognized arguments)
echo "执行: python src/deploy/export_onnx.py --category \"${categories_input}\" ${k_shot_arg} ${verify_flag}"
echo ""

cd "$work_path"
conda activate "$work_env" 2>/dev/null

# "${categories_input}" 加引号 → 整个类别列表作为一个参数;
# ${k_shot_arg} / ${verify_flag} 不加引号 → 按空格拆分为独立参数
python src/deploy/export_onnx.py --category "${categories_input}" ${k_shot_arg} ${verify_flag}

echo ""
echo "================================================"
if [ $? -eq 0 ]; then
    if [ -n "$k_shot_arg" ]; then
        base_suffix="_k${k_shot}_s${shot_seed}"
    else
        base_suffix=""
    fi
    echo "  导出完成！"
    echo ""
    echo "  产物:"
    for cat in "${categories[@]}"; do
        echo "    model_onnx/${cat}${base_suffix}_full.onnx"
    done
else
    echo "  导出失败！请检查错误信息。"
fi
echo "================================================"
