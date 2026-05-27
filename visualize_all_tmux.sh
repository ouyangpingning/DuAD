#!/bin/bash
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
echo "  MVTec AD 可视化脚本 - 交互式配置"
echo "================================================"
echo ""

# --- 1. 模式选择 ---
echo "请选择模型类型:"
echo "  [1] 全样本模型可视化"
echo "  [2] 少样本模型可视化 (K-shot)"
read -p "输入选项 [1/2] (默认=1): " mode_choice
mode_choice=${mode_choice:-1}

if [ "$mode_choice" == "2" ]; then
    read -p "请输入 K 值（少样本数量，默认=4）: " k_shot
    k_shot=${k_shot:-4}
    read -p "请输入随机种子，空格分隔多个seed（默认=0）: " seeds_input
    seeds_input=${seeds_input:-0}
    seeds=($seeds_input)
    extra_args="--k_shot ${k_shot}"
    mode_label="少样本 K=${k_shot}"
else
    k_shot=""
    seeds=(0)
    extra_args=""
    mode_label="全样本"
fi

echo ""

# --- 1.5 推理跳过选项 ---
echo "是否包含模型推理可视化（异常热力图）？"
echo "  推理大约需要 1-2 分钟/类别，跳过则只生成 PCA掩模 / Perlin掩模 / 特征图 / 数据增强图"
read -p "包含推理？[Y/n] (默认=Y): " include_inference
include_inference=${include_inference:-Y}

echo ""

# --- 2. 类别选择 ---
echo "可用的 MVTec AD 类别 (共15类):"
echo "  bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
echo ""
read -p "输入要可视化的类别，空格分隔（回车=全部15类）: " categories_input
if [ -z "$categories_input" ]; then
    categories="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
else
    categories="$categories_input"
fi

num_seeds=${#seeds[@]}

echo ""
echo "================================================"
echo "  配置摘要"
echo "================================================"
echo "  推理可视化:   $([[ "$include_inference" =~ ^[Nn] ]] && echo '跳过' || echo '包含')"
echo "  模型类型:     ${mode_label}"
echo "  类别列表:     ${categories}"
if [ "$mode_choice" == "2" ]; then
    echo "  K 值:         ${k_shot}"
    echo "  种子数量:     ${num_seeds}"
    echo "  种子列表:     ${seeds[@]}"
fi
echo "================================================"
echo ""

echo "代码执行路径: $work_path"
echo "虚拟环境: $work_env"
echo ""

# ==================== 创建 tmux 会话 ====================
# 每个 seed 一个 session，可视化比较轻量，不需要按显存拆分
session_num=0

for seed in "${seeds[@]}"; do
    session_num=$((session_num + 1))

    if [ "$mode_choice" == "2" ]; then
        cmd_args="--categories \"${categories}\" --k_shot ${k_shot} --shot_seed ${seed}"
        session_name="vis_k${k_shot}_s${seed}"
    else
        cmd_args="--categories \"${categories}\""
        session_name="vis_full_s${session_num}"
    fi

    if [[ "$include_inference" =~ ^[Nn] ]]; then
        cmd_args="${cmd_args} --skip_inference"
    fi

    echo "创建 tmux 会话: ${session_name}"
    echo "  种子: ${seed}"
    echo "  类别: ${categories}"
    echo "  命令: python src/visualize_feature.py ${cmd_args}"

    tmux new -d -s "$session_name"
    tmux send-keys -t "$session_name" "cd $work_path" C-m
    tmux send-keys -t "$session_name" "conda activate $work_env" C-m
    tmux send-keys -t "$session_name" "python src/visualize_feature.py ${cmd_args}" C-m

    echo "  会话 ${session_name} 已创建并启动"
    echo ""
done

# ==================== 完成提示 ====================
echo "======================================================="
echo "所有 tmux 会话:"
tmux list-sessions 2>/dev/null || echo "  (无活跃会话)"
echo ""
echo "完成！共创建 ${session_num} 个 tmux 会话"
echo ""
echo "输出目录: ${work_path}/outputs/"
if [[ ! "$include_inference" =~ ^[Nn] ]]; then
    echo "  - {category}_test.png           异常热力图"
fi
echo "  - pca_mask/{category}_pca_mask.png   PCA掩模 (SVD + MLP 对比)"
echo "  - perlin_mask/{category}_perlin_mask.png  Perlin掩模"
echo "  - feature_map/{category}_feature_map.png  特征激活图"
echo "  - augmented/{category}_augmented.png    数据增强效果 (少样本)"
echo ""
echo "使用以下命令管理会话:"
echo "  tmux attach -t <会话名>      # 进入指定会话"
echo "  tmux list-sessions           # 列出所有会话"
echo "  tmux kill-session -t <名>    # 关闭指定会话"
echo "  tmux kill-server             # 关闭所有会话"
