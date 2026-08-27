"""
generate_comparison.py
========================
Reads runs/detect/wtbdefectnet_yolo/results_summary.json and
runs/detect/yolo11n_baseline/results_summary.json (written automatically
by train_yolo.py at the end of each run) and writes a COMPARISON.md at
the repo root -- this is the file worth committing to git alongside your
code, since raw .pt checkpoint files are large and not diffable.

Usage
-----
    python generate_comparison.py
"""

import json
from pathlib import Path

RUN_NAMES = ["wtbdefectnet_yolo", "yolo11n_baseline"]


def load_results(run_name: str):
    path = Path("runs/detect") / run_name / "results_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    results = {name: load_results(name) for name in RUN_NAMES}
    missing = [name for name, r in results.items() if r is None]
    if missing:
        print(f"WARNING: no results_summary.json found for: {missing}")
        print("(train_yolo.py writes this automatically at the end of a run -- "
              "make sure both training commands finished before running this.)")

    lines = ["# WTB-DefectNet + YOLO vs. Stock YOLOv11n -- Results\n"]
    lines.append(f"_Generated automatically by generate_comparison.py_\n")

    lines.append("| Metric | WTB-DefectNet backbone | Stock YOLOv11n |")
    lines.append("|---|---|---|")

    def fmt(key, pct=True):
        vals = []
        for name in RUN_NAMES:
            r = results[name]
            if r is None:
                vals.append("N/A")
            else:
                v = r.get(key)
                vals.append(f"{v*100:.2f}%" if pct and v is not None else (f"{v:,}" if v is not None else "N/A"))
        return vals

    for label, key, pct in [
        ("mAP@0.5", "test_mAP50", True),
        ("mAP@0.5:0.95", "test_mAP50_95", True),
        ("Precision", "test_precision", True),
        ("Recall", "test_recall", True),
        ("Params", "params", False),
        ("Epochs", "epochs", False),
    ]:
        v1, v2 = fmt(key, pct)
        lines.append(f"| {label} | {v1} | {v2} |")

    # Per-class breakdown, if available
    r1 = results.get("wtbdefectnet_yolo")
    r2 = results.get("yolo11n_baseline")
    if r1 and r2 and r1.get("per_class_mAP50") and r2.get("per_class_mAP50"):
        lines.append("\n## Per-class mAP@0.5\n")
        lines.append("| Class | WTB-DefectNet backbone | Stock YOLOv11n |")
        lines.append("|---|---|---|")
        for cls_name in r1["per_class_mAP50"]:
            v1 = r1["per_class_mAP50"].get(cls_name)
            v2 = r2["per_class_mAP50"].get(cls_name)
            v1s = f"{v1*100:.2f}%" if v1 is not None else "N/A"
            v2s = f"{v2*100:.2f}%" if v2 is not None else "N/A"
            lines.append(f"| {cls_name} | {v1s} | {v2s} |")

    out_path = Path("COMPARISON.md")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
