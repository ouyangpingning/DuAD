#!/bin/bash
work_path="$(cd "$(dirname "$0")" && pwd)"

echo "================================================"
echo "  DuAD 结果汇总脚本"
echo "================================================"
echo ""

# --- 日志目录 ---
echo "请选择日志来源:"
echo "  [1] 当前项目 model_log/ (默认)"
echo "  [2] 指定其他目录"
read -p "输入选项 [1/2] (默认=1): " src_choice
src_choice=${src_choice:-1}

if [ "$src_choice" == "2" ]; then
    read -p "请输入日志目录路径: " log_dir
    log_arg="$log_dir"
else
    log_arg="$work_path/model_log"
fi

echo ""

# --- 输出模式 ---
echo "请选择输出方式:"
echo "  [1] 终端打印 + 保存 CSV 到 results/ (默认)"
echo "  [2] 仅终端打印 CSV (方便管道/重定向)"
read -p "输入选项 [1/2] (默认=1): " out_choice
out_choice=${out_choice:-1}

echo ""
echo "================================================"
echo "  汇总 ${log_arg}"
echo "================================================"
echo ""

cd "$work_path"

if [ "$out_choice" == "2" ]; then
    python src/analysis/aggregate_results.py "$log_arg" --csv
else
    python src/analysis/aggregate_results.py "$log_arg"
fi

echo ""
echo "完成！"
