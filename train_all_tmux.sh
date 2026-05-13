#!/bin/bash
# 自动检测当前 conda 环境，不再硬编码环境名
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
now=$(date)
gpu_total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
gpu_used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
gpu_free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
used_memory=3072 # mb per process

# ==================== 交互式配置 ====================
echo "================================================"
echo "  MVTec AD 训练脚本 - 交互式配置"
echo "================================================"
echo ""

# --- 1. 训练模式选择 ---
echo "请选择训练模式:"
echo "  [1] 全样本训练"
echo "  [2] 少样本训练 (K-shot)"
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

# --- 2. 类别选择 ---
echo "可用的 MVTec AD 类别 (共15类):"
echo "  bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
echo ""
read -p "输入要训练的类别，空格分隔（回车=全部15类）: " categories_input
if [ -z "$categories_input" ]; then
    categories=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")
else
    categories=($categories_input)
fi

num_categories=${#categories[@]}
num_seeds=${#seeds[@]}

echo ""
echo "================================================"
echo "  配置摘要"
echo "================================================"
echo "  训练模式:     ${mode_label}"
echo "  类别数量:     ${num_categories}"
echo "  类别列表:     ${categories[@]}"
if [ "$mode_choice" == "2" ]; then
    echo "  K 值:         ${k_shot}"
    echo "  种子数量:     ${num_seeds}"
    echo "  种子列表:     ${seeds[@]}"
fi
echo "================================================"
echo ""

# ==================== GPU 信息 ====================
echo "username: $USER"
echo "虚拟环境: $work_env"
if [ -d "$work_path" ]; then
    echo "代码执行路径: $work_path"
else
    echo "错误: 不存在该文件夹 $work_path"
    exit 1
fi

if [ $gpu_free_memory -eq $gpu_total_memory ]; then
    echo "显存未被占用"
else
    echo "显存已使用: $gpu_used_memory MB"
fi
echo "当前剩余显存: $gpu_free_memory MB"
echo "当前时间: $now"

# ==================== 计算会话分配 ====================
# 根据剩余显存决定最多并行 session 数，每个 session 合并多个类别
num_session=$(($gpu_free_memory / used_memory))
if [ $num_session -eq 0 ]; then
    echo "显存不足，无法启动任何进程"
    exit 1
fi
echo "当前显存最多支持 ${num_session} 个并行进程"

total_tasks=$((num_categories * num_seeds))
echo "总训练任务数: ${total_tasks} (${num_categories} 类别 × ${num_seeds} 种子)"

# 每个 session 分多少任务：总任务数 ÷ session 数，至少 1 个
tasks_per_session=$(( (total_tasks + num_session - 1) / num_session ))
echo "每个 session 分配约 ${tasks_per_session} 个类别"
echo ""

# ==================== 构建任务列表 ====================
# 将 (category, seed) 组合展开为任务数组
task_idx=0
for seed in "${seeds[@]}"; do
    for cat in "${categories[@]}"; do
        task_cats[$task_idx]="$cat"
        task_seeds[$task_idx]="$seed"
        task_idx=$((task_idx + 1))
    done
done

# ==================== 创建 tmux 会话 ====================
session_num=0
task_cursor=0

while [ $task_cursor -lt $total_tasks ]; do
    session_num=$((session_num + 1))

    # 收集当前 session 要处理的类别（同一 seed 下合并多个类别）
    current_seed=${task_seeds[$task_cursor]}
    session_cats=""
    session_task_count=0

    while [ $task_cursor -lt $total_tasks ] && [ $session_task_count -lt $tasks_per_session ]; do
        t_seed=${task_seeds[$task_cursor]}
        t_cat=${task_cats[$task_cursor]}

        # 不同 seed 不能合并到同一 session
        if [ -n "$session_cats" ] && [ "$t_seed" != "$current_seed" ]; then
            break
        fi

        current_seed=$t_seed
        if [ -z "$session_cats" ]; then
            session_cats="$t_cat"
        else
            session_cats="$session_cats $t_cat"
        fi
        session_task_count=$((session_task_count + 1))
        task_cursor=$((task_cursor + 1))
    done

    # 构建命令
    if [ "$mode_choice" == "2" ]; then
        cmd_args="--categories \"${session_cats}\" --k_shot ${k_shot} --shot_seed ${current_seed}"
        session_name="tr_k${k_shot}_s${current_seed}_g${session_num}"
    else
        cmd_args="--categories \"${session_cats}\""
        session_name="tr_g${session_num}"
    fi

    echo "创建 tmux 会话: ${session_name}"
    echo "  seed=${current_seed}  类别: ${session_cats}"

    tmux new -d -s "$session_name"
    tmux send-keys -t "$session_name" "cd $work_path" C-m
    tmux send-keys -t "$session_name" "conda activate $work_env" C-m
    tmux send-keys -t "$session_name" "python src/main.py ${cmd_args}" C-m
    sleep 2  # 错开 CUDA 初始化，避免同时抢显存

    echo "  已启动"
done

# ==================== 完成提示 ====================
echo "======================================================="
echo "所有 tmux 会话:"
tmux list-sessions
echo ""
echo "完成！共创建 ${session_num} 个 tmux 会话"
echo ""
echo "会话命名规则:"
echo "  全样本:   tr_g{N}"
echo "  少样本:   tr_k{K}_s{seed}_g{N}"
echo ""
echo "使用以下命令管理会话:"
echo "  tmux attach -t <会话名>      # 进入指定会话"
echo "  tmux list-sessions           # 列出所有会话"
echo "  tmux kill-session -t <名>    # 关闭指定会话"
echo "  tmux kill-server             # 关闭所有会话"
