"""
train_yolo.py
=============
Entry point for training YOLOv11 with the WTB-DefectNet backbone
(DSPS stem + Stage1-4: TSDB/ASA/DRFB/WGFR/MSCA) on WTBs2025.

Usage
-----
    python train_yolo.py --data /path/to/WTBs2025_yolo/data.yaml \
        --epochs 300 --batch 16 --imgsz 640

Run the plain YOLOv11n baseline for comparison (same data, same
hyperparameters, stock backbone) with:

    python train_yolo.py --data /path/to/WTBs2025_yolo/data.yaml \
        --epochs 300 --batch 16 --imgsz 640 --baseline

Prerequisites
-------------
1. `pip install ultralytics` (tested against ultralytics==8.4.x)
2. This file, wtb_yolo_modules.py, and yolo11-wtbdefectnet.yaml must sit
   in the same folder as your existing `wtb/` package (the one with
   model.py, config.py, etc. from WTB-DefectNet-Experiments) -- copy
   these 3 files into your repo root, don't move `wtb/`.
3. Your dataset must already be in YOLO train/val/test layout with a
   data.yaml (see the dataset-restructuring step discussed separately --
   the WTBs2025.zip download is organized by-class, not by-split, so it
   needs a one-time reshuffle into train/val/test folders first).
"""

import argparse

from wtb_yolo_modules import register_wtb_modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to WTBs2025 data.yaml (train/val/test layout)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Train stock yolo11n.yaml instead of the WTB-DefectNet backbone "
             "-- run this once with identical args as your real comparison point.",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="Run name under runs/detect/<name> (defaults to 'wtbdefectnet' or 'yolo11n_baseline')",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from its last checkpoint instead of starting over. "
             "Uses the same --name as the run you're resuming.",
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        # Resuming loads everything (model weights, optimizer state, epoch
        # count, hyperparameters) from the checkpoint itself -- none of
        # --epochs/--batch/--imgsz/--data need to match what you typed
        # originally, Ultralytics ignores them and continues exactly where
        # it left off.
        run_name = args.name or ("yolo11n_baseline" if args.baseline else "wtbdefectnet_yolo")
        ckpt_path = f"runs/detect/{run_name}/weights/last.pt"
        if args.baseline:
            model = YOLO(ckpt_path)
        else:
            register_wtb_modules()  # still needed: the checkpoint's architecture references our custom classes
            model = YOLO(ckpt_path)
        model.train(resume=True)
        metrics = model.val(data=args.data, split="test")
    else:
        if args.baseline:
            model = YOLO("yolo11n.yaml")  # stock Ultralytics backbone, random init
            run_name = args.name or "yolo11n_baseline"
        else:
            register_wtb_modules()  # must run BEFORE YOLO(cfg) parses the custom yaml
            model = YOLO("yolo11-wtbdefectnet.yaml")
            run_name = args.name or "wtbdefectnet_yolo"

        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=run_name,
            optimizer="auto",
            patience=50,
            plots=True,
        )

        # Also run the held-out test split (not just val) so you get the
        # exact P/R/mAP@0.5 numbers to drop into your comparison table.
        metrics = model.val(data=args.data, split="test")
    print("\n=== TEST SET RESULTS ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")

    # Write a small results.json next to the checkpoints so the numbers
    # can be committed to git alongside the code (not just weight files,
    # which are typically too large / not meaningful to diff in git).
    import json
    from pathlib import Path

    run_dir = Path(model.trainer.save_dir)  # e.g. runs/detect/wtbdefectnet_yolo
    n_params = sum(p.numel() for p in model.model.parameters())
    results = {
        "run_name": run_name,
        "model": "yolo11n_baseline (stock)" if args.baseline else "yolo11 + WTB-DefectNet backbone",
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "params": n_params,
        "test_mAP50": float(metrics.box.map50),
        "test_mAP50_95": float(metrics.box.map),
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
        "per_class_mAP50": {
            model.names[i]: float(v) for i, v in enumerate(metrics.box.ap50)
        } if hasattr(metrics.box, "ap50") else {},
    }
    results_path = run_dir / "results_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results summary to: {results_path}")


if __name__ == "__main__":
    main()
