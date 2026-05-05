import json
import os
import time
from pathlib import Path


def save_metrics(metrics: dict, out_dir: str, filename: str = "metrics.json") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics → {path}")


def print_metrics_table(results: dict) -> None:
    configs = list(results.keys())
    metrics = list(results[configs[0]].keys())
    header = f"{'Metric':<32}" + "".join(f"{c:>16}" for c in configs)
    print("\n" + "=" * (32 + 16 * len(configs)))
    print(header)
    print("─" * (32 + 16 * len(configs)))
    for m in metrics:
        row = f"{m:<32}"
        for c in configs:
            v = results[c].get(m, float("nan"))
            row += f"{v:>16.4f}" if isinstance(v, float) else f"{str(v):>16}"
        print(row)
    print("─" * (32 + 16 * len(configs)))
