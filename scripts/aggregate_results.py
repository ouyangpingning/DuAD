#!/usr/bin/env python3
"""汇总 model_log/ 下所有 *_full.log 的最终评估指标，按类别统计均值 ± 标准差。

结果自动保存至 results/ 目录（CSV 文件），同时打印到终端。

用法:
    python scripts/aggregate_results.py                    # 当前项目 model_log/
    python scripts/aggregate_results.py /path/to/logs     # 指定日志目录
    python scripts/aggregate_results.py --csv              # stdout 输出 CSV（不保存文件）
"""
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_log(log_path: Path) -> List[dict]:
    """解析单个日志文件，返回所有训练完成块的指标列表（支持单文件多 seed/shot）。"""
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception:
        return []

    results = []
    for m in re.finditer(
        r"Training Completed for (\S+).*?"
        r"Best Epoch:\s*(\d+).*?"
        r"Full Evaluation on Best Model:\s*"
        r"(.*?)Best model saved to:\s*(\S+)",
        text, re.DOTALL,
    ):
        category = m.group(1)
        epoch = int(m.group(2))
        metrics_block = m.group(3)
        ckpt_path = m.group(4)

        k_shot = None
        seed = None
        k_match = re.search(r"_k(\d+)_s(\d+)_", ckpt_path)
        if k_match:
            k_shot = int(k_match.group(1))
            seed = int(k_match.group(2))
        elif "_k" in ckpt_path:
            k_match2 = re.search(r"_k(\d+)", ckpt_path)
            if k_match2:
                k_shot = int(k_match2.group(1))

        metrics = {
            "category": category,
            "epoch": epoch,
            "k_shot": k_shot,
            "seed": seed,
        }
        lines = metrics_block.strip().splitlines()
        for line in lines:
            key_match = re.search(r"(Image|Pixel)\s+(AUROC|AP|F1|PRO):\s*([-.\d]+)", line)
            if key_match:
                level = key_match.group(1).lower()
                metric = key_match.group(2).lower()
                value = float(key_match.group(3))
                metrics[f"{level}_{metric}"] = value

        results.append(metrics)

    return results


def discover_logs(log_dir: Path, pattern: str = "*_full.log") -> List[Path]:
    """发现日志目录下所有匹配的日志文件（递归搜索子目录）。"""
    return sorted(log_dir.rglob(pattern))


def extract_dataset(log_path: Path, log_dir: Path) -> str:
    """从日志路径中提取数据集名称。

    日志结构: {log_dir}/{dataset}/{category}/{category}_full.log
    例如: model_log/mvtec/bottle/bottle_full.log → "mvtec"
          model_log/visa/candle/candle_full.log  → "visa"
    """
    try:
        rel = log_path.relative_to(log_dir)
        return rel.parts[0]
    except ValueError:
        return "unknown"


def group_results(results: List[dict]) -> Dict[Tuple[str, str, str], List[dict]]:
    """将结果按 (dataset, category, mode) 分组，mode 为 'full' 或 'k{N}'。"""
    groups = defaultdict(list)
    for r in results:
        if r["k_shot"] is not None:
            mode = f"k{r['k_shot']}"
        else:
            mode = "full"
        groups[(r.get("dataset", ""), r["category"], mode)].append(r)
    return groups


METRIC_NAMES = [
    ("image_auroc", "Image AUROC"),
    ("image_ap",    "Image AP"),
    ("image_f1",    "Image F1"),
    ("pixel_auroc", "Pixel AUROC"),
    ("pixel_ap",    "Pixel AP"),
    ("pixel_f1",    "Pixel F1"),
    ("pixel_pro",   "Pixel PRO"),
]


def compute_stats(vals: List[float]) -> Tuple[float, float, float, float]:
    """计算 mean, std, min, max。"""
    n = len(vals)
    mean_v = sum(vals) / n
    if n > 1:
        var = sum((v - mean_v) ** 2 for v in vals) / n
        std_v = var ** 0.5
    else:
        std_v = 0.0
    return mean_v, std_v, min(vals), max(vals)


# ─── 终端输出 ───────────────────────────────────────────────

def print_table(groups: Dict):
    """打印按数据集+类别分组的汇总表。"""
    for (dataset, category, mode), items in sorted(groups.items()):
        n = len(items)
        label = f"{dataset}/{category}" if dataset else category
        print(f"\n{'─'*70}")
        print(f"  {label}  [{mode}]  ({n} seed{'s' if n > 1 else ''})")
        print(f"{'─'*70}")
        header = f"  {'Metric':<16}"
        if n > 1:
            header += f"{'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}"
        else:
            header += f"{'Value':>8}"
        print(header)
        print(f"  {'─' * 48}")

        for key, label in METRIC_NAMES:
            vals = [m.get(key) for m in items if m.get(key) is not None]
            if not vals:
                continue
            mean_v, std_v, min_v, max_v = compute_stats(vals)
            if n > 1:
                print(f"  {label:<16} {mean_v:>8.4f}  {std_v:>8.4f}  {min_v:>8.4f}  {max_v:>8.4f}")
            else:
                print(f"  {label:<16} {mean_v:>8.4f}")

        if n > 1:
            seeds = [str(m["seed"]) for m in items]
            print(f"  {'Seeds':<16} {', '.join(seeds)}")


def print_cross_category_avg(groups: Dict):
    """跨类别平均终端输出。"""
    print(f"\n{'='*70}")
    print("  跨类别平均汇总 (Cross-Category Average)")
    print(f"{'='*70}")

    for mode in ("full", "k1", "k2", "k4", "k8"):
        mode_items = [(ds, cat, items) for (ds, cat, m), items in groups.items() if m == mode]
        if not mode_items:
            continue
        print(f"\n  Mode: {mode}")
        print(f"  {'Metric':<16} {'Mean ± Std':>20}")
        print(f"  {'─' * 38}")

        for key, label in METRIC_NAMES:
            cat_means = []
            for ds, cat, items in mode_items:
                vals = [m.get(key) for m in items if m.get(key) is not None]
                if vals:
                    cat_means.append(sum(vals) / len(vals))
            if cat_means:
                overall = sum(cat_means) / len(cat_means)
                var = sum((v - overall) ** 2 for v in cat_means) / len(cat_means)
                std = var ** 0.5
                print(f"  {label:<16} {overall:>8.4f} ± {std:.4f}")


def print_overview(groups: Dict):
    """一行一个 dataset+category+mode 的总览。"""
    print(f"\n{'='*70}")
    print("  总览 (Overview)")
    print(f"{'='*70}")
    for (dataset, category, mode), items in sorted(groups.items()):
        n = len(items)
        label = f"{dataset}/{category}" if dataset else category
        img_vals = [m.get("image_auroc") for m in items if m.get("image_auroc") is not None]
        pix_vals = [m.get("pixel_auroc") for m in items if m.get("pixel_auroc") is not None]
        pro_vals = [m.get("pixel_pro") for m in items if m.get("pixel_pro") is not None]
        img_str = f"I-AUROC: {sum(img_vals)/len(img_vals):.4f}" if img_vals else ""
        pix_str = f"P-AUROC: {sum(pix_vals)/len(pix_vals):.4f}" if pix_vals else ""
        pro_str = f"PRO: {sum(pro_vals)/len(pro_vals):.4f}" if pro_vals else ""
        print(f"  {label:<25} {mode:<6}  (n={n})  {img_str}  {pix_str}  {pro_str}")


# ─── CSV 保存 ───────────────────────────────────────────────

def save_csv_files(groups: Dict, output_dir: Path):
    """将结果保存为三个 CSV 文件到 output_dir。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── per_category.csv ──
    per_cat_path = output_dir / "per_category.csv"
    with open(per_cat_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "category", "mode", "n_seeds", "metric", "mean", "std", "min", "max"])
        for (dataset, category, mode), items in sorted(groups.items()):
            n = len(items)
            for key, label in METRIC_NAMES:
                vals = [m.get(key) for m in items if m.get(key) is not None]
                if not vals:
                    continue
                mean_v, std_v, min_v, max_v = compute_stats(vals)
                writer.writerow([dataset, category, mode, n, label, f"{mean_v:.4f}", f"{std_v:.4f}", f"{min_v:.4f}", f"{max_v:.4f}"])

    print(f"\n  已保存: {per_cat_path}")

    # ── cross_category_avg.csv ──
    cross_path = output_dir / "cross_category_avg.csv"
    with open(cross_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mode", "metric", "mean", "std"])
        for mode in ("full", "k1", "k2", "k4", "k8"):
            mode_items = [(ds, cat, items) for (ds, cat, m), items in groups.items() if m == mode]
            if not mode_items:
                continue
            for key, label in METRIC_NAMES:
                # 按数据集分组计算跨类别平均
                ds_means: Dict[str, List[float]] = defaultdict(list)
                for ds, cat, items in mode_items:
                    vals = [m.get(key) for m in items if m.get(key) is not None]
                    if vals:
                        ds_means[ds].append(sum(vals) / len(vals))
                for ds, cat_means in sorted(ds_means.items()):
                    if cat_means:
                        overall = sum(cat_means) / len(cat_means)
                        var = sum((v - overall) ** 2 for v in cat_means) / len(cat_means)
                        std = var ** 0.5
                        writer.writerow([ds, mode, label, f"{overall:.4f}", f"{std:.4f}"])

    print(f"  已保存: {cross_path}")

    # ── details.csv ── 每个 dataset+category+mode+seed 的原始指标
    detail_path = output_dir / "details.csv"
    detail_headers = ["dataset", "category", "mode", "k_shot", "seed"] + [label for _, label in METRIC_NAMES]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(detail_headers)
        for (dataset, category, mode), items in sorted(groups.items()):
            for m in sorted(items, key=lambda x: (x.get("seed") is None, x.get("seed", 0))):
                row = [
                    dataset,
                    category,
                    mode,
                    m.get("k_shot", ""),
                    m.get("seed", ""),
                ]
                for key, _ in METRIC_NAMES:
                    val = m.get(key)
                    row.append(f"{val:.4f}" if val is not None else "")
                writer.writerow(row)

    print(f"  已保存: {detail_path}")


def print_csv_stdout(groups: Dict):
    """stdout 输出 CSV（兼容旧版 --csv 行为）。"""
    print("dataset,category,mode,n_seeds,metric,mean,std,min,max")
    for (dataset, category, mode), items in sorted(groups.items()):
        n = len(items)
        for key, label in METRIC_NAMES:
            vals = [m.get(key) for m in items if m.get(key) is not None]
            if not vals:
                continue
            mean_v, std_v, min_v, max_v = compute_stats(vals)
            print(f"{dataset},{category},{mode},{n},{label},{mean_v:.4f},{std_v:.4f},{min_v:.4f},{max_v:.4f}")


# ─── 主入口 ─────────────────────────────────────────────────

def main():
    csv_stdout = "--csv" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--csv"]

    if args:
        log_dir = Path(args[0])
    else:
        log_dir = Path(__file__).resolve().parent.parent / "model_log"

    if not log_dir.is_dir():
        print(f"日志目录不存在: {log_dir}")
        sys.exit(1)

    # 输出目录为项目根下的 results/
    results_dir = Path(__file__).resolve().parent.parent / "results"

    # 发现并解析日志
    log_files = discover_logs(log_dir)
    print(f"扫描目录: {log_dir}")
    print(f"发现 {len(log_files)} 个日志文件")

    results = []
    for f in log_files:
        parsed = parse_log(f)
        if parsed:
            dataset = extract_dataset(f, log_dir)
            for r in parsed:
                r["dataset"] = dataset
            results.extend(parsed)
        else:
            print(f" 跳过 (无法解析): {f.name}")

    if not results:
        print(" 未找到任何可解析的训练日志。")
        sys.exit(1)

    print(f"成功解析 {len(results)} 个训练结果\n")

    groups = group_results(results)

    if csv_stdout:
        print_csv_stdout(groups)
    else:
        print_table(groups)
        print_cross_category_avg(groups)
        print_overview(groups)
        print(f"\n{'='*70}")
        print("  保存 CSV 结果")
        print(f"{'='*70}")
        save_csv_files(groups, results_dir)


if __name__ == "__main__":
    main()
