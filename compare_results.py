"""
compare_results.py
===================
Reads one or more test_metrics.json files -- from evaluate.py,
cascade_infer.py, OR ensemble_infer.py, any mix -- and prints a single
clean comparison table. Auto-detects which kind of file each one is by
its keys, so you don't need to remember which nested field belongs to
which script.

Usage:
    python compare_results.py runs\Try_9\eval\test_metrics.json ^
                               runs\Try_9_phase2\eval\test_metrics.json ^
                               runs\Try_10\eval\test_metrics.json ^
                               runs\Try_9_cascade_v2\eval\test_metrics.json ^
                               runs\ensemble_8_9_cascade\eval\test_metrics.json

(Windows line-continuation is ^ , not the backtick used inside .ps1 files --
run this one directly with plain paths, no quoting gymnastics needed.)

This replaces the old inline `python -c "..."` calls in run_accuracy_push.ps1,
which broke because PowerShell mangles nested double-quotes inside an
already-double-quoted -c string. A real .py file sidesteps that entirely.
"""

import argparse
import json
import os


def extract_rows(path: str):
    """Returns a list of (label, accuracy, macro_f1) tuples for one JSON file.
    A single evaluate.py file yields one row; a cascade/ensemble file yields
    two (before/after, or ensemble-only/final)."""
    with open(path, "r") as f:
        d = json.load(f)

    rows = []
    tag = os.path.relpath(path)

    if "metrics_before_cascade" in d and "metrics_after_cascade" in d:
        # cascade_infer.py output
        before = d["metrics_before_cascade"]
        after = d["metrics_after_cascade"]
        rows.append((f"{tag}  [main only]", before["accuracy"], before["macro_f1"]))
        rows.append((f"{tag}  [+cascade, w={d.get('cascade_weight')}]",
                      after["accuracy"], after["macro_f1"]))

    elif "metrics_ensemble_only" in d:
        # ensemble_infer.py output
        ens = d["metrics_ensemble_only"]
        rows.append((f"{tag}  [ensemble only]", ens["accuracy"], ens["macro_f1"]))
        if "metrics_final" in d:
            fin = d["metrics_final"]
            rows.append((f"{tag}  [ensemble+cascade]", fin["accuracy"], fin["macro_f1"]))

    elif "accuracy" in d and "macro_f1" in d:
        # plain evaluate.py output
        rows.append((tag, d["accuracy"], d["macro_f1"]))

    else:
        rows.append((f"{tag}  [UNRECOGNIZED FORMAT]", float("nan"), float("nan")))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_paths", nargs="+", help="One or more test_metrics.json files")
    args = ap.parse_args()

    all_rows = []
    for path in args.json_paths:
        if not os.path.exists(path):
            print(f"[compare] WARNING: {path} does not exist, skipping.")
            continue
        all_rows.extend(extract_rows(path))

    if not all_rows:
        print("[compare] No valid result files found.")
        return

    all_rows.sort(key=lambda r: (r[1] if r[1] == r[1] else -1), reverse=True)  # NaN-safe sort desc by accuracy

    label_w = max(len(r[0]) for r in all_rows) + 2
    print(f"\n{'RESULT':<{label_w}}{'ACCURACY':>10}{'MACRO-F1':>10}")
    print("-" * (label_w + 20))
    for label, acc, f1 in all_rows:
        acc_s = f"{acc:.4f}" if acc == acc else "  n/a "
        f1_s = f"{f1:.4f}" if f1 == f1 else "  n/a "
        print(f"{label:<{label_w}}{acc_s:>10}{f1_s:>10}")
    print()
    best = max((r for r in all_rows if r[1] == r[1]), key=lambda r: r[1], default=None)
    if best:
        print(f"BEST: {best[0]}  (accuracy={best[1]:.4f}, macro_f1={best[2]:.4f})")


if __name__ == "__main__":
    main()
