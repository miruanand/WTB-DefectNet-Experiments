"""
generate_comparison.py
========================
Reads results_summary.json from all three runs (written automatically
by train_yolo.py at the end of each run) and writes a COMPARISON.md at
the repo root -- this is the file worth committing to git alongside your
code, since raw .pt checkpoint files are large and not diffable.

    runs/detect/wtbdefectnet_yolo/results_summary.json        (--arch p2, default)
    runs/detect/wtbdefectnet_yolo_noP2/results_summary.json   (--arch no_p2)
    runs/detect/yolo11n_baseline/results_summary.json         (--baseline)

Usage
-----
    python generate_comparison.py
"""

import json
from pathlib import Path

RUNS = [
    ("wtbdefectnet_yolo", "WTB-DefectNet (P2)"),
    ("wtbdefectnet_yolo_noP2", "WTB-DefectNet (no P2)"),
    ("yolo11n_baseline", "Stock YOLOv11n"),
]


def load_results(run_name: str):
    path = Path("runs/detect") / run_name / "results_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    results = {run_name: load_results(run_name) for run_name, _ in RUNS}
    missing = [run_name for run_name, r in results.items() if r is None]
    if missing:
        print(f"WARNING: no results_summary.json found for: {missing}")
        print("(train_yolo.py writes this automatically at the end of a run -- "
              "make sure all training commands finished before running this.)")

    lines = ["# WTB-DefectNet: P2 vs No-P2 vs Stock YOLOv11n -- Results\n"]
    lines.append("_Generated automatically by generate_comparison.py_\n")

    header = "| Metric | " + " | ".join(label for _, label in RUNS) + " |"
    sep = "|---|" + "---|" * len(RUNS)
    lines.append(header)
    lines.append(sep)

    def fmt(key, pct=True):
        vals = []
        for run_name, _ in RUNS:
            r = results[run_name]
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
        ("Image size", "imgsz", False),
    ]:
        vals = fmt(key, pct)
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    # Per-class breakdown, if all three have it
    per_class_available = [
        results[run_name].get("per_class_mAP50")
        for run_name, _ in RUNS
        if results[run_name] and results[run_name].get("per_class_mAP50")
    ]
    if len(per_class_available) == len(RUNS):
        lines.append("\n## Per-class mAP@0.5\n")
        header = "| Class | " + " | ".join(label for _, label in RUNS) + " |"
        sep = "|---|" + "---|" * len(RUNS)
        lines.append(header)
        lines.append(sep)
        # use the class list from the first run as the canonical ordering
        first_run_classes = list(results[RUNS[0][0]]["per_class_mAP50"].keys())
        for cls_name in first_run_classes:
            row_vals = []
            for run_name, _ in RUNS:
                v = results[run_name]["per_class_mAP50"].get(cls_name)
                row_vals.append(f"{v*100:.2f}%" if v is not None else "N/A")
            lines.append(f"| {cls_name} | " + " | ".join(row_vals) + " |")

    out_path = Path("COMPARISON.md")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
