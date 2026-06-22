#!/bin/bash
# =============================================================================
# 消融实验批量训练脚本 — 4-shot MVTec AD, 5 seeds (0, 42, 123, 456, 789)
#
# 用法:
#   bash train_ablation.sh dino2_only    # B: DINOv2 单分支 (无 PCA, 无 Perlin)
#   bash train_ablation.sh pca_only      # C: DINOv2 + PCA (无 Perlin)
#   bash train_ablation.sh no_augment    # E: Full DuAD 无数据增强
#   bash train_ablation.sh all           # 依次运行以上三个
#
# 对应关系:
#   dino2_only  → --no_pca_mask           (自动关闭 Perlin，回退单分支 Hinge)
#   pca_only    → --no_perlin_mask         (保留 PCA，单分支 Hinge，无 Perlin)
#   no_augment  → --no_augment             (PCA+Perlin 双分支，无数据增强)
#
# checkpoint/log 命名示例:
#   bottle_k4_s0_noPCA_best_ckpt.pth     (dino2_only)
#   bottle_k4_s0_noPerlin_best_ckpt.pth  (pca_only)
#   bottle_k4_s0_noAug_best_ckpt.pth     (no_augment)
# =============================================================================

set -e

# ==================== 参数解析 ====================
if [ $# -eq 0 ]; then
    echo "用法: bash train_ablation.sh <variant>"
    echo ""
    echo "可用的消融变体:"
    echo "  dino2_only   — DINOv2 单分支 (无 PCA, 无 Perlin, 有增强)"
    echo "  pca_only     — DINOv2 + PCA (无 Perlin, 有增强)"
    echo "  no_augment   — Full DuAD 无数据增强"
    echo "  all          — 依次运行以上三个变体"
    echo ""
    echo "示例:"
    echo "  bash train_ablation.sh dino2_only"
    echo "  bash train_ablation.sh all"
    exit 1
fi

VARIANT="$1"

case "$VARIANT" in
    dino2_only)
        ABLATION_FLAG="--no_pca_mask"
        ABLATION_LABEL="DINOv2 single-branch (no PCA, no Perlin)"
        ;;
    pca_only)
        ABLATION_FLAG="--no_perlin_mask"
        ABLATION_LABEL="DINOv2 + PCA (no Perlin)"
        ;;
    no_augment)
        ABLATION_FLAG="--no_augment"
        ABLATION_LABEL="Full DuAD without augmentation"
        ;;
    all)
        echo "================================================"
        echo "  消融实验 — 依次运行全部 3 个变体"
        echo "================================================"
        echo ""
        for v in dino2_only pca_only no_augment; do
            echo ">>> 开始运行: $v"
            bash "$0" "$v"
            echo ""
            echo "<<< 完成: $v"
            echo ""
        done
        echo "================================================"
        echo "  全部消融实验完成!"
        echo "================================================"
        exit 0
        ;;
    *)
        echo "错误: 未知的消融变体 '$VARIANT'"
        echo "可用: dino2_only, pca_only, no_augment, all"
        exit 1
        ;;
esac

# ==================== 固定参数 ====================
K_SHOT=4
SEEDS=(0 42 123 456 789)
DATASET="mvtec"

# MVTec AD 全部 15 类
CATEGORIES=(
    bottle cable capsule carpet grid hazelnut leather
    metal_nut pill screw tile toothbrush transistor wood zipper
)

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
used_memory=3072  # MB per process

if [ -z "$gpu_free_memory" ]; then
    echo "错误: 无法获取 GPU 显存信息，请确认 nvidia-smi 可用"
    exit 1
fi

num_categories=${#CATEGORIES[@]}
num_seeds=${#SEEDS[@]}
total_tasks=$((num_categories * num_seeds))

# ==================== 打印配置 ====================
echo "================================================"
echo "  消融实验 — 批量训练"
echo "================================================"
echo "  消融变体:     ${ABLATION_LABEL}"
echo "  Ablation flag: ${ABLATION_FLAG}"
echo "  数据集:       ${DATASET}"
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
session_num=0
task_cursor=0

while [ $task_cursor -lt $total_tasks ]; do
    session_num=$((session_num + 1))
    session_name="abl_${VARIANT_SHORT}_k${K_SHOT}_g${session_num}"

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
    session_name="abl_${VARIANT_SHORT}_k${K_SHOT}_${seed_label}_g${session_num}"

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
