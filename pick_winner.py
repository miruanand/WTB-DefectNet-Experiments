"""
pick_winner.py
===============
Compares Try_9, Try_9_phase2, and Try_10's test_metrics.json (by accuracy)
and writes the winner's identity to runs/winner.json:

    {"exp": "Try_9_phase2", "backbone": "resnet18", "img_size": 384,
     "accuracy": 0.8123, "macro_f1": 0.7601}

run_full_pipeline.ps1 reads this file with PowerShell's ConvertFrom-Json
instead of parsing python -c stdout -- that inline-quoting approach is
exactly what crashed the earlier run_accuracy_push.ps1 script.

A candidate that hasn't been trained yet (file missing) is silently
skipped, not treated as an error -- lets this run mid-pipeline as each
stage finishes.
"""

import json
import os

CANDIDATES = [
    {"exp": "Try_9",         "backbone": "resnet18", "img_size": 384},
    {"exp": "Try_9_phase2",  "backbone": "resnet18", "img_size": 384},
    {"exp": "Try_10",        "backbone": "resnet34", "img_size": 448},
]


def main():
    results = []
    for c in CANDIDATES:
        path = os.path.join("runs", c["exp"], "eval", "test_metrics.json")
        if not os.path.exists(path):
            print(f"[pick_winner] {path} not found -- skipping {c['exp']}.")
            continue
        with open(path, "r") as f:
            d = json.load(f)
        acc = d.get("accuracy")
        f1 = d.get("macro_f1")
        if acc is None:
            print(f"[pick_winner] WARNING: {path} has no 'accuracy' field -- skipping.")
            continue
        print(f"[pick_winner] {c['exp']:16s}: accuracy={acc:.4f}  macro_f1={f1:.4f}")
        results.append({**c, "accuracy": acc, "macro_f1": f1})

    if not results:
        raise SystemExit("[pick_winner] No valid result files found. Nothing to compare.")

    winner = max(results, key=lambda r: r["accuracy"])
    print(f"\n[pick_winner] WINNER: {winner['exp']} "
          f"(accuracy={winner['accuracy']:.4f}, macro_f1={winner['macro_f1']:.4f})")

    with open(os.path.join("runs", "winner.json"), "w") as f:
        json.dump(winner, f, indent=2)
    print("[pick_winner] Wrote runs/winner.json")


if __name__ == "__main__":
    main()
