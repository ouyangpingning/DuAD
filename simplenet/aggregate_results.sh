#!/bin/bash
# SimpleNet 结果汇总脚本
# 解析 model_log/simplenet/ 下的所有日志并输出到 results/simplenet/

cd "$(dirname "$0")/.."
python simplenet/aggregate_results.py "$@"
