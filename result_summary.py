import os
import re
import pandas as pd

root_dir = "./results"

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

        loss_metrics = {}
        for fname in os.listdir(campus_path):
            if fname.endswith("-performance.txt"):
                match = pattern_loss.search(fname)
                if not match:
                    continue
                loss_id = int(match.group(1))
                filepath = os.path.join(campus_path, fname)
                loss_metrics[loss_id] = parse_performance_file(filepath)

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
df = df.sort_values(by=["Model", "Campus", "Loss"],
                    ascending=[True, True, True])

# ====== 命令行美观打印 ======
# 打印表头
headers = ["Model", "Campus", "Loss", "TopK", "Metric",
           "Baseline(loss0)", "Value", "AbsDiff", "RelDiff(%)"]

col_widths = [max(len(str(x)) for x in [col] + df[col].astype(str).tolist())
              for col in headers]

header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
print(header_line)
print("-" * len(header_line))

# 打印数据行
for _, row in df.iterrows():
    row_strs = [
        str(row["Model"]).ljust(col_widths[0]),
        str(row["Campus"]).ljust(col_widths[1]),
        str(row["Loss"]).ljust(col_widths[2]),
        str(row["TopK"]).ljust(col_widths[3]),
        str(row["Metric"]).ljust(col_widths[4]),
        f"{row['Baseline(loss0)']:.5f}".ljust(col_widths[5]),
        f"{row['Value']:.5f}".ljust(col_widths[6]),
        f"{row['AbsDiff']:.5f}".ljust(col_widths[7]),
        f"{row['RelDiff(%)']:.2f}%".ljust(col_widths[8]),
    ]
    line = "  ".join(row_strs)
    if row["RelDiff(%)"] > 0:  # 提升 → 红色
        print(f"\033[91m{line}\033[0m")
    else:
        print(line)

# 仍然保存干净的 CSV
df.to_csv("loss_comparison_summary.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存到 loss_comparison_summary.csv")
