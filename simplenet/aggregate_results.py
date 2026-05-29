#!/usr/bin/env python3
"""汇总 SimpleNet model_log/simplenet/ 下所有 *_full.log 的最终评估指标。

用法:
    python simplenet/aggregate_results.py
    python simplenet/aggregate_results.py --csv
"""

import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_log(log_path: Path) -> List[dict]:
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
        for line in metrics_block.strip().splitlines():
            key_match = re.search(r"(Image|Pixel)\s+(AUROC|AP|F1|PRO):\s*([-.\d]+)", line)
            if key_match:
                level = key_match.group(1).lower()
                metric = key_match.group(2).lower()
                value = float(key_match.group(3))
                metrics[f"{level}_{metric}"] = value

        results.append(metrics)

    return results


def discover_logs(log_dir: Path, pattern: str = "*_full.log") -> List[Path]:
    return sorted(log_dir.rglob(pattern))


def group_results(results: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups = defaultdict(list)
    for r in results:
        mode = f"k{r['k_shot']}" if r["k_shot"] is not None else "full"
        groups[(r["category"], mode)].append(r)
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
    n = len(vals)
    mean_v = sum(vals) / n
    std_v = 0.0 if n <= 1 else (sum((v - mean_v) ** 2 for v in vals) / n) ** 0.5
    return mean_v, std_v, min(vals), max(vals)


def print_table(groups: Dict):
    for (category, mode), items in sorted(groups.items()):
        n = len(items)
        print(f"\n{'─'*70}")
        print(f"  {category}  [{mode}]  ({n} seed{'s' if n > 1 else ''})")
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
    print(f"\n{'='*70}")
    print("  跨类别平均汇总 (Cross-Category Average) [SimpleNet]")
    print(f"{'='*70}")
    for mode in ("full", "k1", "k2", "k4", "k8"):
        mode_items = [(cat, items) for (cat, m), items in groups.items() if m == mode]
        if not mode_items:
            continue
        print(f"\n  Mode: {mode}")
        print(f"  {'Metric':<16} {'Mean ± Std':>20}")
        print(f"  {'─' * 38}")
        for key, label in METRIC_NAMES:
            cat_means = []
            for cat, items in mode_items:
                vals = [m.get(key) for m in items if m.get(key) is not None]
                if vals:
                    cat_means.append(sum(vals) / len(vals))
            if cat_means:
                overall = sum(cat_means) / len(cat_means)
                var = sum((v - overall) ** 2 for v in cat_means) / len(cat_means)
                std = var ** 0.5
                print(f"  {label:<16} {overall:>8.4f} ± {std:.4f}")


def print_overview(groups: Dict):
    print(f"\n{'='*70}")
    print("  总览 (Overview) [SimpleNet]")
    print(f"{'='*70}")
    for (category, mode), items in sorted(groups.items()):
        n = len(items)
        img_vals = [m.get("image_auroc") for m in items if m.get("image_auroc") is not None]
        pix_vals = [m.get("pixel_auroc") for m in items if m.get("pixel_auroc") is not None]
        pro_vals = [m.get("pixel_pro") for m in items if m.get("pixel_pro") is not None]
        img_str = f"I-AUROC: {sum(img_vals)/len(img_vals):.4f}" if img_vals else ""
        pix_str = f"P-AUROC: {sum(pix_vals)/len(pix_vals):.4f}" if pix_vals else ""
        pro_str = f"PRO: {sum(pro_vals)/len(pro_vals):.4f}" if pro_vals else ""
        print(f"  {category:<15} {mode:<6}  (n={n})  {img_str}  {pix_str}  {pro_str}")


def save_csv_files(groups: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    per_cat_path = output_dir / "per_category.csv"
    with open(per_cat_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "mode", "n_seeds", "metric", "mean", "std", "min", "max"])
        for (category, mode), items in sorted(groups.items()):
            n = len(items)
            for key, label in METRIC_NAMES:
                vals = [m.get(key) for m in items if m.get(key) is not None]
                if not vals:
                    continue
                mean_v, std_v, min_v, max_v = compute_stats(vals)
                writer.writerow([category, mode, n, label, f"{mean_v:.4f}", f"{std_v:.4f}", f"{min_v:.4f}", f"{max_v:.4f}"])
    print(f"\n  已保存: {per_cat_path}")

    cross_path = output_dir / "cross_category_avg.csv"
    with open(cross_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "metric", "mean", "std"])
        for mode in ("full", "k1", "k2", "k4", "k8"):
            mode_items = [(cat, items) for (cat, m), items in groups.items() if m == mode]
            if not mode_items:
                continue
            for key, label in METRIC_NAMES:
                cat_means = []
                for cat, items in mode_items:
                    vals = [m.get(key) for m in items if m.get(key) is not None]
                    if vals:
                        cat_means.append(sum(vals) / len(vals))
                if cat_means:
                    overall = sum(cat_means) / len(cat_means)
                    var = sum((v - overall) ** 2 for v in cat_means) / len(cat_means)
                    std = var ** 0.5
                    writer.writerow([mode, label, f"{overall:.4f}", f"{std:.4f}"])
    print(f"  已保存: {cross_path}")

    detail_path = output_dir / "details.csv"
    detail_headers = ["category", "mode", "k_shot", "seed"] + [label for _, label in METRIC_NAMES]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(detail_headers)
        for (category, mode), items in sorted(groups.items()):
            for m in sorted(items, key=lambda x: (x.get("seed") is None, x.get("seed", 0))):
                row = [category, mode, m.get("k_shot", ""), m.get("seed", "")]
                for key, _ in METRIC_NAMES:
                    val = m.get(key)
                    row.append(f"{val:.4f}" if val is not None else "")
                writer.writerow(row)
    print(f"  已保存: {detail_path}")


def print_csv_stdout(groups: Dict):
    print("category,mode,n_seeds,metric,mean,std,min,max")
    for (category, mode), items in sorted(groups.items()):
        n = len(items)
        for key, label in METRIC_NAMES:
            vals = [m.get(key) for m in items if m.get(key) is not None]
            if not vals:
                continue
            mean_v, std_v, min_v, max_v = compute_stats(vals)
            print(f"{category},{mode},{n},{label},{mean_v:.4f},{std_v:.4f},{min_v:.4f},{max_v:.4f}")


def main():
    csv_stdout = "--csv" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--csv"]

    _script_dir = Path(__file__).resolve().parent
    _proj_root = _script_dir.parent

    if args:
        log_dir = Path(args[0])
    else:
        log_dir = _proj_root / "model_log" / "simplenet"

    if not log_dir.is_dir():
        print(f"日志目录不存在: {log_dir}")
        sys.exit(1)

    results_dir = _proj_root / "results" / "simplenet"

    log_files = discover_logs(log_dir)
    print(f"扫描目录: {log_dir}")
    print(f"发现 {len(log_files)} 个日志文件")

    results = []
    for f in log_files:
        parsed = parse_log(f)
        if parsed:
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
        print("  保存 CSV 结果 [SimpleNet]")
        print(f"{'='*70}")
        save_csv_files(groups, results_dir)


if __name__ == "__main__":
    main()
