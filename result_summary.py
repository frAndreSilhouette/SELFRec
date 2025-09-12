import os
import re
import pandas as pd
from collections import defaultdict
import numpy as np

# root_dir = "./history_results/results7/"
root_dir = "./results/"

pattern_loss = re.compile(r"loss(\d+)")
pattern_metric = re.compile(r"(.+?):([\d.]+)")


def parse_performance_file(filepath):
    """解析 performance.txt 文件，返回 {TopK: {metric: value}}"""
    metrics = {}
    current_topk = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Top"):
                current_topk = line  # e.g. "Top 10"
                metrics[current_topk] = {}
            elif ":" in line and current_topk is not None:
                match = pattern_metric.match(line)
                if match:
                    key, val = match.groups()
                    metrics[current_topk][key.strip()] = float(val)
    return metrics


results = []

for model in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model)
    if not os.path.isdir(model_path):
        continue

    for campus in os.listdir(model_path):
        campus_path = os.path.join(model_path, campus)
        if not os.path.isdir(campus_path):
            continue

        # 改为支持同一 loss_id 多文件取均值
        raw_loss_metrics = defaultdict(list)
        for fname in os.listdir(campus_path):
            if fname.endswith("-performance.txt"):
                match = pattern_loss.search(fname)
                if not match:
                    continue
                loss_id = int(match.group(1))
                filepath = os.path.join(campus_path, fname)
                raw_loss_metrics[loss_id].append(parse_performance_file(filepath))

        # 取均值
        loss_metrics = {}
        for loss_id, metrics_list in raw_loss_metrics.items():
            merged = {}
            for metrics in metrics_list:
                for topk, vals in metrics.items():
                    if topk not in merged:
                        merged[topk] = defaultdict(list)
                    for metric, v in vals.items():
                        merged[topk][metric].append(v)
            # 对每个 metric 取均值
            loss_metrics[loss_id] = {
                topk: {m: np.mean(vs) for m, vs in mdict.items()}
                for topk, mdict in merged.items()
            }

        # 基于均值做比较
        for topk in next(iter(loss_metrics.values())).keys():
            baseline = loss_metrics.get(0, {}).get(topk, {})
            if not baseline:
                continue

            for loss_id, metrics_dict in loss_metrics.items():
                if loss_id == 0:
                    continue
                compare = metrics_dict.get(topk, {})
                for metric, value in compare.items():
                    base_val = baseline.get(metric, None)
                    if base_val is not None and base_val != 0:
                        abs_diff = value - base_val
                        rel_diff = abs_diff / base_val * 100
                        results.append({
                            "Model": model,
                            "Campus": campus,
                            "Loss": f"loss{loss_id}",
                            "TopK": topk,
                            "Metric": metric,
                            "Baseline(loss0)": base_val,
                            "Value": value,
                            "AbsDiff": abs_diff,
                            "RelDiff(%)": rel_diff
                        })

df = pd.DataFrame(results, columns=[
    "Model", "Campus", "Loss", "TopK", "Metric",
    "Baseline(loss0)", "Value", "AbsDiff", "RelDiff(%)"
])

# 排序
df = df.sort_values(by=["Model", "Campus", "Loss"], ascending=[True, True, True])

# 保存 CSV
df.to_csv(f"{root_dir}result_summary.csv", index=False, encoding="utf-8-sig")
print(f"结果已保存到 {root_dir}result_summary.csv")
