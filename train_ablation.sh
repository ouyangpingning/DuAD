#!/bin/bash
# =============================================================================
# 消融实验批量训练脚本 (交互式) — MVTec AD
#
# 用法:
#   bash train_ablation.sh
#
#   运行后按提示交互选择: 消融变体 → K-Shot → 种子 → 类别
# =============================================================================

set -e

# ==================== 交互式参数 ====================

echo ""
echo "================================================"
echo "  消融实验 — 交互式配置"
echo "================================================"
echo ""

# --- 1. 选择消融变体 ---
echo "请选择消融变体:"
echo "  1) dino2_only      — DINOv2 单分支 (无 PCA, 无 Perlin)"
echo "  2) pca_only        — DINOv2 + PCA (无 Perlin)"
echo "  3) no_augment      — Full DuAD 无数据增强"
echo "  4) channel_concat  — 通道拼接聚合 (替代门控融合)"
echo "  5) neighborhood     — 邻域聚合 (替代门控融合)"
echo ""
while true; do
    read -p "输入编号 [1-5]: " variant_choice
    case "$variant_choice" in
        1) VARIANT="dino2_only"
           ABLATION_FLAG="--no_pca_mask"
           ABLATION_LABEL="DINOv2 single-branch (no PCA, no Perlin)"; break ;;
        2) VARIANT="pca_only"
           ABLATION_FLAG="--no_perlin_mask"
           ABLATION_LABEL="DINOv2 + PCA (no Perlin)"; break ;;
        3) VARIANT="no_augment"
           ABLATION_FLAG="--no_augment"
           ABLATION_LABEL="Full DuAD without augmentation"; break ;;
        4) VARIANT="channel_concat"
           ABLATION_FLAG="--aggregation channel_concat"
           ABLATION_LABEL="Channel concat aggregation (no neighborhood)"; break ;;
        5) VARIANT="neighborhood"
           ABLATION_FLAG="--aggregation neighborhood"
           ABLATION_LABEL="Neighborhood aggregation (no fusion)"; break ;;
        *) echo "无效输入，请输入 1-5" ;;
    esac
done
echo "  → 已选择: ${ABLATION_LABEL}"
echo ""

# --- 2. K-Shot ---
read -p "K-Shot 数量 [默认: 4]: " k_input
K_SHOT=${k_input:-4}
echo "  → K-Shot: ${K_SHOT}"
echo ""

# --- 3. 种子列表 ---
read -p "种子列表，空格分隔 [默认: 0 42 123 456 789]: " seeds_input
if [ -z "$seeds_input" ]; then
    SEEDS=(0 42 123 456 789)
else
    SEEDS=($seeds_input)
fi
echo "  → 种子: ${SEEDS[@]} (共 ${#SEEDS[@]} 个)"
echo ""

# --- 4. 数据集 ---
echo "选择数据集:"
echo "  1) MVTec AD (15 类)"
echo "  2) VisA (12 类)"
echo ""
while true; do
    read -p "输入编号 [1-2, 默认: 1]: " dataset_choice
    dataset_choice=${dataset_choice:-1}
    case "$dataset_choice" in
        1) DATASET="mvtec"
           DEFAULT_CATS=(bottle cable capsule carpet grid hazelnut leather
                          metal_nut pill screw tile toothbrush transistor wood zipper)
           break ;;
        2) DATASET="visa"
           DEFAULT_CATS=(candle capsules cashew chewinggum fryum macaroni1 macaroni2
                          pcb1 pcb2 pcb3 pcb4 pipe_fryum)
           break ;;
        *) echo "无效输入，请输入 1-2" ;;
    esac
done
echo "  → 数据集: ${DATASET}"
echo ""

# --- 5. 类别 ---
echo "${DATASET} 默认类别 (${#DEFAULT_CATS[@]} 类):"
echo "  ${DEFAULT_CATS[@]}"
echo ""
read -p "指定类别 (空格分隔, 回车=全部) [默认: 全部]: " cats_input
if [ -z "$cats_input" ]; then
    CATEGORIES=("${DEFAULT_CATS[@]}")
else
    CATEGORIES=($cats_input)
fi
echo "  → 类别: ${CATEGORIES[@]} (共 ${#CATEGORIES[@]} 个)"
echo ""

# --- 6. 确认 ---
num_categories=${#CATEGORIES[@]}
num_seeds=${#SEEDS[@]}
total_tasks=$((num_categories * num_seeds))

echo "================================================"
echo "  配置确认"
echo "================================================"
echo "  消融变体:     ${ABLATION_LABEL}"
echo "  Ablation flag: ${ABLATION_FLAG}"
echo "  K-Shot:       ${K_SHOT}"
echo "  种子:         ${SEEDS[@]}"
echo "  类别:         ${CATEGORIES[@]}"
echo "  总任务数:     ${num_categories} 类 × ${num_seeds} 种子 = ${total_tasks}"
echo "================================================"
echo ""
read -p "确认开始训练? [Y/n]: " confirm
if [ "$confirm" = "n" ] || [ "$confirm" = "N" ]; then
    echo "已取消"
    exit 0
fi
echo ""

# ==================== 环境检测 ====================
work_env="${CONDA_DEFAULT_ENV:-base}"
if [ "$work_env" = "base" ]; then
    echo "⚠ 当前 conda 环境为 base，建议先激活 pytorch 环境"
    read -p "是否继续？[y/N]: " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消。请先执行: conda activate <你的环境名>"
        exit 1
    fi
fi

work_path="$(cd "$(dirname "$0")" && pwd)"
now=$(date)

gpu_total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
gpu_free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
used_memory=2560  # MB per process

if [ -z "$gpu_free_memory" ]; then
    echo "错误: 无法获取 GPU 显存信息，请确认 nvidia-smi 可用"
    exit 1
fi

# ==================== 打印配置 ====================
echo "================================================"
echo "  消融实验 — 批量训练"
echo "================================================"
echo "  消融变体:     ${ABLATION_LABEL}"
echo "  Ablation flag: ${ABLATION_FLAG}"
echo "  数据集:       ${DATASET}"
echo "  训练类别:     ${CATEGORIES[@]}"
echo "  K-Shot:       ${K_SHOT}"
echo "  种子数量:     ${num_seeds} (${SEEDS[@]})"
echo "  类别数量:     ${num_categories}"
echo "  总任务数:     ${total_tasks}"
echo "  虚拟环境:     ${work_env}"
echo "  工作目录:     ${work_path}"
echo "  GPU 总显存:   ${gpu_total_memory} MB"
echo "  GPU 剩余:     ${gpu_free_memory} MB"
echo "  当前时间:     ${now}"
echo "================================================"
echo ""

# ==================== 计算 tmux 分配 ====================
num_session=$(($gpu_free_memory / $used_memory))
if [ $num_session -eq 0 ]; then
    echo "显存不足，无法启动任何进程"
    exit 1
fi
if [ $num_session -gt $total_tasks ]; then
    num_session=$total_tasks
fi
echo "最多并行 ${num_session} 个 tmux 会话"

# 均匀分配任务到 sessions
tasks_per_session_base=$((total_tasks / num_session))
tasks_remainder=$((total_tasks % num_session))
echo "任务分配: ${tasks_remainder} 个 session 各 ${tasks_per_session_base}+1 个任务，"\
"$((num_session - tasks_remainder)) 个 session 各 ${tasks_per_session_base} 个任务"
echo ""

# 预计算每个 session 的配额
for ((i=0; i<num_session; i++)); do
    if [ $i -lt $tasks_remainder ]; then
        session_quota[$i]=$((tasks_per_session_base + 1))
    else
        session_quota[$i]=$tasks_per_session_base
    fi
done

# ==================== 构建任务列表 ====================
task_idx=0
for seed in "${SEEDS[@]}"; do
    for cat in "${CATEGORIES[@]}"; do
        task_cats[$task_idx]="$cat"
        task_seeds[$task_idx]="$seed"
        task_idx=$((task_idx + 1))
    done
done

# ==================== 创建 tmux 会话 ====================
VARIANT_SHORT="${VARIANT//_/-}"
RUN_TS=$(date +%m%d_%H%M)  # 时间戳确保每次运行 session 名不冲突
session_num=0
task_cursor=0

while [ $task_cursor -lt $total_tasks ]; do
    session_num=$((session_num + 1))
    session_name="abl_${RUN_TS}_${VARIANT_SHORT}_k${K_SHOT}_g${session_num}"

    # 安全检查: 同名 session 已存在则跳过 (理论上不会，但兜底)
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "⚠ 跳过: session '${session_name}' 已存在"
        # 跳过当前 session 配额内的任务
        tasks_this_session=${session_quota[$((session_num - 1))]}
        skipped=0
        while [ $task_cursor -lt $total_tasks ] && [ $skipped -lt $tasks_this_session ]; do
            task_cursor=$((task_cursor + 1))
            skipped=$((skipped + 1))
        done
        continue
    fi

    # 收集当前 session 的任务
    declare -A seed_cats
    session_seeds=()
    session_task_count=0

    tasks_this_session=${session_quota[$((session_num - 1))]}
    while [ $task_cursor -lt $total_tasks ] && [ $session_task_count -lt $tasks_this_session ]; do
        t_seed=${task_seeds[$task_cursor]}
        t_cat=${task_cats[$task_cursor]}

        if [ -z "${seed_cats[$t_seed]}" ]; then
            session_seeds+=("$t_seed")
            seed_cats[$t_seed]="$t_cat"
        else
            seed_cats[$t_seed]="${seed_cats[$t_seed]} $t_cat"
        fi

        session_task_count=$((session_task_count + 1))
        task_cursor=$((task_cursor + 1))
    done

    # 种子范围标签
    if [ ${#session_seeds[@]} -eq 1 ]; then
        seed_label="s${session_seeds[0]}"
    else
        seed_label="s${session_seeds[0]}-s${session_seeds[-1]}"
    fi
    session_name="abl_${RUN_TS}_${VARIANT_SHORT}_k${K_SHOT}_${seed_label}_g${session_num}"

    echo "创建 tmux: ${session_name}  (${session_task_count} 任务)"
    for s in "${session_seeds[@]}"; do
        echo "  seed=${s}: ${seed_cats[$s]}"
    done

    tmux new -d -s "$session_name"
    tmux send-keys -t "$session_name" "cd $work_path" C-m
    tmux send-keys -t "$session_name" "conda activate $work_env" C-m

    # 按 seed 顺序发送命令
    first_cmd=true
    for s in "${session_seeds[@]}"; do
        cmd_args="--categories \"${seed_cats[$s]}\" --k_shot ${K_SHOT} --shot_seed ${s} --dataset ${DATASET} ${ABLATION_FLAG}"

        if [ "$first_cmd" = true ]; then
            first_cmd=false
            tmux send-keys -t "$session_name" "python src/main.py ${cmd_args}"
        else
            tmux send-keys -t "$session_name" " ; python src/main.py ${cmd_args}"
        fi
    done
    tmux send-keys -t "$session_name" C-m
    sleep 2

    echo "  已启动"
    unset seed_cats
    unset session_seeds
done

# ==================== 完成 ====================
echo ""
echo "========================================================"
echo "  消融实验 [${ABLATION_LABEL}] 已全部提交"
echo "  共创建 ${session_num} 个 tmux 会话"
echo ""
echo "  所有 tmux 会话:"
tmux list-sessions 2>/dev/null | grep "abl_" || echo "  (无)"
echo ""
echo "  管理命令:"
echo "    tmux attach -t <会话名>     # 进入指定会话"
echo "    tmux list-sessions          # 列出所有会话"
echo "    tmux kill-session -t <名>   # 关闭指定会话"
echo "========================================================"
