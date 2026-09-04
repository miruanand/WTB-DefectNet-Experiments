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
    ap.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Training resolution. Defaults to 1024 (up from 640) -- WTBs2025's shipped "
             "images are already pre-resized to 640x640 by the dataset's own publishers, "
             "so raising imgsz here does NOT recover real lost detail (Ultralytics just "
             "upsamples the 640x640 source via interpolation). It still helps to a real "
             "but more limited degree than true higher-resolution source images would: "
             "at 1024 input with the P2 head's stride-4 output, a defect that was ~2px "
             "wide at 640 becomes ~3.2px in the (upsampled) 1024 image, giving the stride-4 "
             "grid more room to localize it. Try 1280 if your GPU has the memory for it "
             "(lower --batch if you hit an out-of-memory error at higher imgsz).",
    )
    ap.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader worker processes. Defaults to Ultralytics' standard 8. "
             "If training seems to hang right after 'Starting training for N epochs...' "
             "with no progress and no GPU/CPU activity in Task Manager for several "
             "minutes, this is usually either (a) first-run kernel compilation on "
             "Intel XPU, which is genuinely slow and just needs more time, or (b) a "
             "Windows multiprocessing deadlock with multiple worker processes. Try "
             "--workers 0 (single-process, slower per-batch but far more reliable on "
             "Windows) to tell which one you're hitting: if it starts immediately with "
             "--workers 0, it was (b); if it still takes a while, it's (a) and is normal.",
    )
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
    ap.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Mosaic augmentation probability (0.0-1.0). Defaults to OFF (0.0) here -- "
             "testing on WTBs2025 showed several defect classes have boxes as small as "
             "2-3 pixels (pinholes, coating detachment, erosion, localized damage, paint "
             "cracks, surface stains), and mosaic's aggressive crop/resize can shrink or "
             "cut these out entirely during training, which was a likely contributor to "
             "those classes scoring near zero mAP even though the data itself was verified "
             "correctly labeled. Pass --mosaic 1.0 to restore Ultralytics' default if you "
             "want to A/B test this specific change.",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=0.2,
        help="Random scale-jitter augmentation range (Ultralytics default is 0.5, i.e. "
             "each image can be randomly scaled +/-50%% during training). Defaulted down "
             "to 0.2 here: your smallest real defects (pinholes, hairline cracks) are "
             "already only 1-3 pixels wide, so an aggressive scale-down during "
             "augmentation can shrink them to sub-pixel size and destroy the training "
             "signal entirely. Pass --scale 0.5 to restore Ultralytics' default if you "
             "want to A/B test this specific change.",
    )
    ap.add_argument(
        "--arch",
        choices=["p2", "no_p2"],
        default="p2",
        help="Which WTB-DefectNet backbone yaml to use (ignored when --baseline is set). "
             "'p2' (default) = yolo11-wtbdefectnet.yaml, the 4-scale P2/P3/P4/P5 version. "
             "'no_p2' = yolo11-wtbdefectnet-noP2.yaml, the original 3-scale P3/P4/P5 "
             "version, kept around specifically so you can run a controlled P2-vs-no-P2 "
             "ablation with every other setting held identical -- run once with each and "
             "compare the per-class mAP for the tiny-object classes (pinholes, coating "
             "detachment, etc.) to see how much the P2 head is actually contributing.",
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
            arch_yaml = "yolo11-wtbdefectnet.yaml" if args.arch == "p2" else "yolo11-wtbdefectnet-noP2.yaml"
            model = YOLO(arch_yaml)
            run_name = args.name or ("wtbdefectnet_yolo" if args.arch == "p2" else "wtbdefectnet_yolo_noP2")

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
            mosaic=args.mosaic,
            scale=args.scale,
            workers=args.workers,
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
